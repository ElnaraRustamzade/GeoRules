# GeoRules

Rule-based 3D geological reservoir modeling library.

## What This Project Does

GeoRules generates synthetic 3D geological models using rule-based and stochastic methods. It targets subsurface reservoir modeling for oil & gas, groundwater, and carbon storage applications.

## Problem It Solves

Building realistic 3D geological models typically requires expensive commercial software and extensive manual input. GeoRules provides a Python-native, pip-installable alternative that generates geologically plausible reservoir models programmatically. Users define layer types and parameters; the library handles the physics-based geometry and property modeling.

## Supported Geology Types

- **Lobe layers** — Turbidite lobe deposition with compensational stacking, Bouma sequences, and upthinning
- **Gaussian layers** — Sequential Gaussian simulation (SGS) with spatial correlation for heterogeneous sand/shale distributions
- **Channel layers** — Fluvial channel systems with meandering, migration, avulsion, neck cutoffs, and point bar geometry

## Architecture

- `Layer` base class defines grid geometry (nx, ny, nz, dimensions, depth, dip)
- Each layer type inherits from `Layer` and implements `create_geology()` to populate 3D property arrays
- `Reservoir` stacks multiple layers vertically, validating compatibility
- All outputs are numpy arrays shaped `(nx, ny, nz)`

## Key Commands

- Install: `pip install -e ".[dev]"`
- Test: `pytest tests/`
- Tutorial: `jupyter notebook notebooks/tutorial.ipynb`

## Conventions

- Array ordering: `(nx, ny, nz)` with `meshgrid(..., indexing='ij')`
- Properties: `poro_mat` (porosity, 0-1), `perm_mat` (permeability, mD), `active` (0/1 facies mask)
- Physics parameters go in `create_geology()`, not `__init__()` — init is grid-only
- Channel internals use Numba JIT and are prefixed with `_` (private)
