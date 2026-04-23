"""Parameter-space sampling for dataset generation.

Reads the ``layers`` section of a dataset config and returns a
memory-efficient :class:`JobList` that lazily materialises one
``{layer_type, params, seed}`` dict per indexed access. At 10M samples the
list-of-dicts representation would cost ~8 GB per worker, which does not
fit 128 concurrent ranks on a Perlmutter CPU node; the compact storage
here keeps the per-rank overhead under ~1 GB even at tens of millions
of samples.

Per-parameter specs supported:

- ``{"range": [lo, hi]}``                    continuous float uniform
- ``{"range": [lo, hi], "scale": "log"}``    log-uniform in [lo, hi]
- ``{"range": [lo, hi], "type": "int"}``     integer in [lo, hi] inclusive
- ``{"choices": [...]}``                      categorical (int/str/bool)
- ``{"value": v}``                            fixed scalar
- ``{"levels": N}`` (grid sampling only)     number of grid steps on that axis

Sampling strategies:

- ``"sobol"``   — ``scipy.stats.qmc.Sobol(scramble=True)``
- ``"lhs"``     — ``scipy.stats.qmc.LatinHypercube``
- ``"grid"``    — full factorial Cartesian product of per-param levels
- ``"uniform"`` — IID uniform via ``numpy.random.default_rng``
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc


@dataclass
class _LayerState:
    name: str
    variable_names: list[str]
    fixed: dict[str, Any]
    unit: np.ndarray           # (count, d) float32 unit-cube samples
    grid_combos: list[tuple] | None   # only for sampling="grid"


class JobList:
    """Shuffled list of jobs with ``O(N)`` memory and lazy materialisation.

    Indexing ``job_list[i]`` returns a dict
    ``{"layer_type": str, "params": dict, "seed": int}`` equivalent to
    what the previous eager implementation produced — same content, just
    computed on demand.
    """

    def __init__(
        self,
        layer_names: list[str],
        layer_type_ids: np.ndarray,
        within_idx: np.ndarray,
        seeds: np.ndarray,
        layer_states: dict[str, _LayerState],
        layers_cfg: dict,
    ):
        self._layer_names = layer_names
        self._layer_type_ids = layer_type_ids
        self._within_idx = within_idx
        self._seeds = seeds
        self._layer_states = layer_states
        self._layers_cfg = layers_cfg

    def __len__(self) -> int:
        return int(self._layer_type_ids.shape[0])

    def __getitem__(self, i: int) -> dict:
        lt = self._layer_names[int(self._layer_type_ids[i])]
        within = int(self._within_idx[i])
        seed = int(self._seeds[i])
        state = self._layer_states[lt]
        params_cfg = self._layers_cfg[lt]["params"]
        params = dict(state.fixed)

        if state.grid_combos is not None:
            combo = state.grid_combos[within]
            params.update(dict(zip(state.variable_names, combo)))
        else:
            unit_row = state.unit[within]
            for u, name in zip(unit_row, state.variable_names):
                params[name] = _map_unit_value(float(u), params_cfg[name])

        return {"layer_type": lt, "params": params, "seed": seed}


def build_jobs(layers_cfg: dict, master_seed: int) -> JobList:
    """Build a shuffled :class:`JobList` over every layer section."""
    layer_names = list(layers_cfg.keys())
    counts = np.array([int(cfg["count"]) for cfg in layers_cfg.values()],
                      dtype=np.int64)
    total_n = int(counts.sum())

    layer_type_ids = np.empty(total_n, dtype=np.int8)
    within_idx = np.empty(total_n, dtype=np.int32)
    seeds = np.empty(total_n, dtype=np.uint32)

    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    layer_states: dict[str, _LayerState] = {}

    for lt_id, name in enumerate(layer_names):
        cfg = layers_cfg[name]
        n = int(counts[lt_id])
        section_seed = int((master_seed + 1) * 10007 + lt_id)

        params_cfg = cfg["params"]
        variable_names = [k for k, s in params_cfg.items() if "value" not in s]
        fixed = {k: s["value"] for k, s in params_cfg.items() if "value" in s}

        sampling = cfg.get("sampling", "sobol")
        grid_combos = None
        if sampling == "grid":
            grid_combos = _grid_combos(variable_names, params_cfg, n)
            unit = np.empty((n, 0), dtype=np.float32)
        elif not variable_names:
            unit = np.empty((n, 0), dtype=np.float32)
        elif sampling == "sobol":
            unit = _sample_sobol(len(variable_names), n, section_seed)
        elif sampling == "lhs":
            unit = _sample_lhs(len(variable_names), n, section_seed)
        elif sampling == "uniform":
            unit = np.random.default_rng(section_seed).uniform(
                size=(n, len(variable_names))
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown sampling strategy {sampling!r}")

        layer_states[name] = _LayerState(
            name=name,
            variable_names=variable_names,
            fixed=fixed,
            unit=unit.astype(np.float32, copy=False),
            grid_combos=grid_combos,
        )

        a, b = int(offsets[lt_id]), int(offsets[lt_id + 1])
        layer_type_ids[a:b] = lt_id
        within_idx[a:b] = np.arange(n, dtype=np.int32)
        seeds[a:b] = np.random.default_rng(section_seed).integers(
            1, 2**31 - 1, size=n, dtype=np.uint32
        )

    # Global shuffle — mixes layer types across the index order so ranks
    # get a statistically uniform slice of costs.
    perm = np.random.default_rng(master_seed).permutation(total_n)
    layer_type_ids = layer_type_ids[perm]
    within_idx = within_idx[perm]
    seeds = seeds[perm]

    return JobList(layer_names, layer_type_ids, within_idx, seeds,
                   layer_states, layers_cfg)


def _sample_sobol(d: int, n: int, seed: int) -> np.ndarray:
    # Sobol is balanced at powers of 2; request next power up and truncate.
    m = int(np.ceil(np.log2(max(n, 2))))
    total = 2**m
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(total)[:n]


def _sample_lhs(d: int, n: int, seed: int) -> np.ndarray:
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def _map_unit_value(u: float, spec: dict):
    if "choices" in spec:
        choices = spec["choices"]
        k = min(int(u * len(choices)), len(choices) - 1)
        return choices[k]
    lo, hi = spec["range"]
    if spec.get("scale") == "log":
        val = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    else:
        val = float(lo + u * (hi - lo))
    if spec.get("type") == "int":
        return int(round(val))
    return val


def _grid_combos(names: list, params_cfg: dict, count: int) -> list[tuple]:
    axes: list[list] = []
    for name in names:
        spec = params_cfg[name]
        if "choices" in spec:
            axes.append(list(spec["choices"]))
            continue
        n_levels = spec.get("levels")
        if n_levels is None:
            raise ValueError(f"grid sampling requires 'levels' on param {name!r}")
        us = np.linspace(0, 1, int(n_levels))
        axes.append([_map_unit_value(float(u), spec) for u in us])
    full = list(itertools.product(*axes))
    if len(full) != count:
        raise ValueError(
            f"grid sampling over {names} produced {len(full)} combinations "
            f"but config declares count={count}"
        )
    return full
