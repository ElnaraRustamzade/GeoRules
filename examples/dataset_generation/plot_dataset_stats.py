"""Per-layer-type statistics plot for a generated dataset shard directory.

Reads the slim parquet (``params_slim.parquet``) plus the full parquet
(``params.parquet``) from each shard and produces, per layer type, a
one-page summary figure with:

* NTG histogram + realized-vs-requested NTG scatter
* mean-poro / mean-perm histograms
* width_cells / depth_cells histograms
* layer-specific knob histograms (sinuosity / asp / probAvulInside /
  mFFCHprop / trunk_length_fraction as applicable)
* azimuth rose plot
* slim-column correlation heatmap

Reusable: ``plot_layer_stats(data_dir, output_path, layer_type)`` builds
one figure for the rows matching ``layer_type``. ``main()`` iterates
over all layer types found in the dataset.

Usage:
    python plot_dataset_stats.py SHARD_DIR
    python plot_dataset_stats.py SHARD_DIR --out X
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _load_slim_rows(data_dir: Path):
    """Read every shard's ``params_slim.parquet`` into a list of pylist rows."""
    rows = []
    shards = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("shard_"))
    for shard in shards:
        slim_path = shard / "params_slim.parquet"
        if not slim_path.exists():
            continue
        rows.extend(pq.read_table(slim_path).to_pylist())
    return rows


def _column(rows, key) -> np.ndarray:
    """Pull non-None values for ``key`` into a numpy float array."""
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return np.asarray([float(v) for v in vals], dtype=np.float64)


def _hist(ax, vals, title, bins=30, color=None):
    """Plain histogram with count + mean/median annotations."""
    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        return
    ax.hist(vals, bins=bins, color=color or "C0", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.set_title(f"{title}  (n={vals.size})")
    ax.axvline(vals.mean(), color="k", lw=1, linestyle="--",
               label=f"mean={vals.mean():.3g}")
    ax.axvline(np.median(vals), color="r", lw=1, linestyle=":",
               label=f"med={np.median(vals):.3g}")
    ax.legend(fontsize=8, loc="best", framealpha=0.7)
    ax.tick_params(labelsize=8)


def _corr_heatmap(ax, rows, slim_keys):
    """Pairwise correlation across numeric slim columns (drop None / non-numeric)."""
    numeric_keys = []
    cols = []
    for k in slim_keys:
        v = _column(rows, k)
        if v.size > len(rows) * 0.5:  # ≥50% populated
            numeric_keys.append(k)
            cols.append(v)
    if len(numeric_keys) < 2:
        ax.text(0.5, 0.5, "not enough numeric cols", ha="center",
                va="center", transform=ax.transAxes, color="grey")
        ax.set_title("correlation"); ax.set_xticks([]); ax.set_yticks([])
        return
    # Reindex per-row so columns align (some rows might miss a value)
    K = len(numeric_keys)
    matrix = np.full((len(rows), K), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(numeric_keys):
            v = r.get(k)
            if v is not None:
                try: matrix[i, j] = float(v)
                except Exception: pass
    # Drop fully-NaN rows
    keep = ~np.all(np.isnan(matrix), axis=1)
    matrix = matrix[keep]
    if matrix.shape[0] < 2:
        ax.text(0.5, 0.5, "not enough rows", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title("correlation"); ax.set_xticks([]); ax.set_yticks([])
        return
    # Pairwise corr ignoring NaNs
    corr = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            mask = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
            if mask.sum() >= 3:
                v = np.corrcoef(matrix[mask, i], matrix[mask, j])[0, 1]
                corr[i, j] = v
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(numeric_keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(numeric_keys, fontsize=7)
    ax.set_title("slim-column correlation")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(K):
        for j in range(K):
            v = corr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.5 else "white")


def _azimuth_rose(ax, vals):
    """Polar histogram of azimuth (compass-CW degrees, 0..360)."""
    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title("azimuth")
        return
    rad = np.deg2rad(vals)
    n_bins = 24
    counts, bin_edges = np.histogram(rad, bins=n_bins, range=(0, 2 * np.pi))
    width = 2 * np.pi / n_bins
    centers = bin_edges[:-1] + width / 2
    ax.bar(centers, counts, width=width, bottom=0,
           color="C2", alpha=0.85, edgecolor="white")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)        # CW
    ax.set_title(f"azimuth rose (n={vals.size})")
    ax.tick_params(labelsize=7)


def plot_layer_stats(rows, layer_type: str, output_path: Path):
    """Build one stats figure for the rows matching ``layer_type``.

    Layout:
    * Row 0: 3 universal histograms (realized NTG, poro_ave, perm_ave)
      + family-specific knob count to fill out the row.
    * Subsequent rows: width_cells, depth_cells, family knobs, azimuth
      rose, packed dynamically so there's no orphan plot.
    * Bottom: full-width correlation heatmap.
    """
    matching = [r for r in rows if r.get("layer_type") == layer_type]
    if not matching:
        print(f"  skip {layer_type}: no rows")
        return

    n_total = len(matching)
    family = layer_type.split(":", 1)[0]
    if family == "lobe":
        knobs = ["asp"]
    elif family == "channel":
        knobs = ["mCHsinu", "probAvulInside", "mFFCHprop"]
    elif family == "delta":
        knobs = ["mCHsinu", "probAvulInside", "mFFCHprop", "trunk_length_fraction"]
    else:
        knobs = []

    # Build the ordered list of (kind, key/label) plot specs:
    # 3 universal hists + width/depth + family knobs + azimuth rose.
    specs = [
        ("hist", "ntg",         "realized NTG"),
        ("hist", "poro_ave",    "poro_ave (active)"),
        ("hist", "perm_ave",    "perm_ave  log10(mD)"),
        ("hist", "width_cells", "width_cells"),
        ("hist", "depth_cells", "depth_cells"),
    ]
    for k in knobs:
        specs.append(("hist", k, k))
    specs.append(("rose", "azimuth", "azimuth"))

    n_plots = len(specs)
    # Pick n_cols ∈ {3, 4, 5} that minimises trailing-empty slots so the
    # picture has no awkward whitespace. Tie-break by preferring values
    # close to 4 (a balanced grid).
    cand = []
    for nc in (3, 4, 5):
        nr = (n_plots + nc - 1) // nc
        cand.append((nr * nc - n_plots, abs(nc - 4), nc))
    cand.sort()
    n_cols = cand[0][2]
    n_plot_rows = (n_plots + n_cols - 1) // n_cols   # ceil

    # Figure width scales with column count so each plot stays ~3.5 in
    # wide; height tracks plot-row count + heatmap.
    fig_w = 3.5 * n_cols + 1.0
    fig_h = 3.0 * n_plot_rows + 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"{layer_type}  ({n_total} samples)",
                 fontsize=14, fontweight="bold")

    # +1 row for the wide heatmap.
    gs = fig.add_gridspec(n_plot_rows + 1, n_cols,
                          height_ratios=[1] * n_plot_rows + [1.4])
    for i, (kind, key, label) in enumerate(specs):
        r, c = divmod(i, n_cols)
        if kind == "rose":
            ax = fig.add_subplot(gs[r, c], projection="polar")
            _azimuth_rose(ax, _column(matching, key))
        else:
            ax = fig.add_subplot(gs[r, c])
            _hist(ax, _column(matching, key), label)

    # Heatmap spans the full width on the final row.
    ax = fig.add_subplot(gs[n_plot_rows, :])
    numeric_slim = ["ntg", "poro_ave", "perm_ave", "width_cells",
                    "depth_cells", "azimuth"] + knobs
    _corr_heatmap(ax, matching, numeric_slim)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="output dir with shard_*/ subdirs")
    p.add_argument("--out", default=None,
                   help="override output dir (default: <data_dir>/stats_pictures/)")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else data_dir / "stats_pictures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_slim_rows(data_dir)
    if not rows:
        raise SystemExit(f"no slim parquet rows found in {data_dir}")
    layer_types = sorted({r.get("layer_type") for r in rows
                          if r.get("layer_type")})
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"layer types: {layer_types}")
    for lt in layer_types:
        # Replace ':' with '_' for filesystem-safe filename
        safe = lt.replace(":", "__")
        plot_layer_stats(rows, lt, out_dir / f"stats_{safe}.png")


if __name__ == "__main__":
    main()
