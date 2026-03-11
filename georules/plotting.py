import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_cube_slices(data_3d, ix=None, iy=None, iz=None,
                     cmap='YlOrRd', vmin=None, vmax=None, title=None, ax=None):
    """Plot 3 orthogonal slices arranged as faces of a cube.

    Parameters
    ----------
    data_3d : (nx, ny, nz) array
    ix, iy, iz : int or None
        Slice indices. Defaults: ix=nx-1, iy=ny-1, iz=0 (far walls + floor).
    cmap, vmin, vmax : colormap args
    title : str
    ax : Axes3D or None

    Returns
    -------
    fig, ax
    """
    nx, ny, nz = data_3d.shape
    if ix is None:
        ix = nx - 1
    if iy is None:
        iy = ny - 1
    if iz is None:
        iz = 0

    # Mask zeros for better visualization
    masked = np.where(data_3d > 0, data_3d, np.nan)
    if vmin is None:
        vmin = np.nanmin(masked) if np.any(~np.isnan(masked)) else 0
    if vmax is None:
        vmax = np.nanmax(masked) if np.any(~np.isnan(masked)) else 1

    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    grey = np.array([0.6, 0.6, 0.6, 1.0])

    # Draw surfaces back-to-front for correct depth ordering (azim=225).
    # 1. YZ slice at x=ix (farthest from camera)
    Z_grid3, Y_grid3 = np.meshgrid(np.arange(nz), np.arange(ny))
    X_grid3 = np.full_like(Y_grid3, ix, dtype=float)
    slice_data = masked[ix, :, :]
    colors3 = cmap_obj(norm(np.nan_to_num(slice_data, nan=vmin)))
    colors3[np.isnan(slice_data)] = grey
    ax.plot_surface(X_grid3, Y_grid3, Z_grid3, facecolors=colors3, shade=False)

    # 2. XZ slice at y=iy (middle distance)
    Z_grid2, X_grid2 = np.meshgrid(np.arange(nz), np.arange(nx))
    Y_grid2 = np.full_like(X_grid2, iy, dtype=float)
    slice_data = masked[:, iy, :]
    colors2 = cmap_obj(norm(np.nan_to_num(slice_data, nan=vmin)))
    colors2[np.isnan(slice_data)] = grey
    ax.plot_surface(X_grid2, Y_grid2, Z_grid2, facecolors=colors2, shade=False)

    # 3. XY slice at z=iz (closest to camera — floor)
    Y_grid, X_grid = np.meshgrid(np.arange(ny), np.arange(nx))
    Z_grid = np.full_like(X_grid, iz, dtype=float)
    slice_data = masked[:, :, iz]
    colors = cmap_obj(norm(np.nan_to_num(slice_data, nan=vmin)))
    colors[np.isnan(slice_data)] = grey
    ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=colors, shade=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=225)
    if title:
        ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label=title or '')

    return fig, ax


def plot_slices(data_3d, axis=2, indices=None, ncols=4,
                cmap='YlOrRd', vmin=None, vmax=None, title=None):
    """Plot 2D slices along one axis as a subplot grid.

    Parameters
    ----------
    data_3d : (nx, ny, nz) array
    axis : int
        Axis to slice along (0=x, 1=y, 2=z).
    indices : list of int or None
        Slice indices. None = 8 evenly spaced.
    ncols : int
        Columns in subplot grid.
    cmap, vmin, vmax : colormap args
    title : str

    Returns
    -------
    fig, axes
    """
    n = data_3d.shape[axis]
    if indices is None:
        indices = np.linspace(0, n - 1, min(8, n), dtype=int)

    masked = np.where(data_3d > 0, data_3d, np.nan)
    if vmin is None:
        vmin = np.nanmin(masked) if np.any(~np.isnan(masked)) else 0
    if vmax is None:
        vmax = np.nanmax(masked) if np.any(~np.isnan(masked)) else 1

    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    axis_labels = {0: ('Y', 'Z'), 1: ('X', 'Z'), 2: ('X', 'Y')}
    xlabel, ylabel = axis_labels[axis]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='grey')

    for i, idx in enumerate(indices):
        sl = np.take(masked, idx, axis=axis)
        im = axes[i].imshow(sl.T, origin='lower', cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect='auto')
        axes[i].set_title(f'{"XYZ"[axis]}={idx}')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)

    # Hide unused axes
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    fig.colorbar(im, ax=axes[:len(indices)], shrink=0.8, label=title or '')
    if title:
        fig.suptitle(title, fontsize=14)
    return fig, axes


def plot_layer(layer, prop='poro_mat', **kwargs):
    """Convenience: plot_cube_slices on a Layer's property."""
    data = getattr(layer, prop)
    return plot_cube_slices(data, **kwargs)


def plot_reservoir(reservoir, prop='poro_mat', **kwargs):
    """Convenience: plot_cube_slices on a Reservoir's property."""
    data = getattr(reservoir, prop)
    return plot_cube_slices(data, **kwargs)
