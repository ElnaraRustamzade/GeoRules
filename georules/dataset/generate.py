"""Generate one reservoir sample and package outputs for storage.

Wraps a GeoRules Layer construction and ``create_geology`` call with
deterministic seeding (for LobeLayer / GaussianLayer /
MeanderingChannelLayer / BraidedChannelLayer; DeltaLayer uses its own
internal ``np.random.default_rng`` and is not bit-reproducible from the
stored seed), binarises the resulting facies / active array, and casts
porosity and permeability to float16 for on-disk compactness.
"""

import numpy as np

import georules as gr
from .captions import caption_for


_LAYER_FACTORY = {
    "lobe": gr.LobeLayer,
    "gaussian": gr.GaussianLayer,
    "meandering": gr.MeanderingChannelLayer,
    "braided": gr.BraidedChannelLayer,
    "delta": gr.DeltaLayer,
}


def generate_sample(job: dict, grid_cfg: dict):
    """Generate one sample.

    Returns ``(facies, poro, perm, meta)``:

    - ``facies`` — int8 array in {0, 1}, shape ``(nx, ny, nz)``
    - ``poro``   — float16 array, shape ``(nx, ny, nz)``
    - ``perm``   — float16 array (mD), shape ``(nx, ny, nz)``
    - ``meta``   — dict with ``layer_type``, ``seed``, ``caption`` and
      every sampled parameter.
    """
    seed = int(job["seed"])
    np.random.seed(seed)
    layer_type = job["layer_type"]
    layer = _LAYER_FACTORY[layer_type](**grid_cfg)
    layer.create_geology(**job["params"])

    facies = _binarize(layer, layer_type).astype(np.int8)
    # Clip to float16-safe ranges before casting: poro to [0, 1], perm to
    # [0, 60000] mD. This also silently coerces any residual +inf from
    # extreme perm_std draws into 60000 (float16 max is 65504).
    poro = np.clip(
        np.asarray(layer.poro_mat, dtype=np.float32), 0.0, 1.0
    ).astype(np.float16)
    perm = np.clip(
        np.asarray(layer.perm_mat, dtype=np.float32), 0.0, 6e4
    ).astype(np.float16)

    meta = {
        "layer_type": layer_type,
        "seed": seed,
        "caption": caption_for(layer_type, job["params"]),
        **job["params"],
    }
    return facies, poro, perm, meta


def _binarize(layer, layer_type: str) -> np.ndarray:
    # GaussianLayer has no ``facies`` attribute; use ``active`` (0/1).
    # All other layer types: fold any ``facies > 0`` into binary 1
    # (LobeLayer encodes the lobe generation index as int, not 0/1).
    if layer_type == "gaussian":
        return np.asarray(layer.active) > 0
    return np.asarray(layer.facies) > 0
