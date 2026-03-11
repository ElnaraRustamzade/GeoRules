# GeoRules

Rule-based 3D geological reservoir modeling.

## Install

```bash
pip install georules
```

For development:

```bash
git clone <repo-url>
cd GeoRules
pip install -e ".[dev]"
```

## Quick Start

```python
import georules as gr

# Create a turbidite lobe layer
lobe = gr.LobeLayer(nx=100, ny=100, nz=50, x_len=3000, y_len=3000, z_len=100, top_depth=5000)
lobe.create_geology(poro_ave=0.20, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)

# Create a Gaussian simulation layer
gauss = gr.GaussianLayer(nx=100, ny=100, nz=30, x_len=3000, y_len=3000, z_len=60, top_depth=5100)
gauss.create_geology(poro_ave=0.18, perm_ave=1.2, poro_std=0.03, perm_std=0.4, ntg=0.6)

# Stack into a reservoir
reservoir = gr.Reservoir([lobe, gauss])

# Visualize
gr.plot_cube_slices(reservoir.poro_mat, title="Porosity")
```

## Layer Types

- **LobeLayer** — Turbidite lobe deposition with compensational stacking
- **GaussianLayer** — Sequential Gaussian simulation for heterogeneous facies
- **ChannelLayer** — Fluvial channels with meandering, migration, and avulsion

## License

MIT
