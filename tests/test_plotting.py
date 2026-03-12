import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from georules.plotting import plot_cube_slices, plot_slices, GEORULES_CMAP


def test_plot_cube_slices_returns_fig():
    data = np.random.rand(10, 10, 5)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)


def test_plot_slices_returns_fig():
    data = np.random.rand(10, 10, 5)
    fig, axes = plot_slices(data, axis=2)
    assert fig is not None
    plt.close(fig)


def test_plot_cube_slices_with_zeros():
    """Handles data with many zeros (inactive cells)."""
    data = np.zeros((10, 10, 5))
    data[3:7, 3:7, 1:4] = np.random.rand(4, 4, 3)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)


def test_plot_slices_axis_0():
    data = np.random.rand(10, 10, 5)
    fig, axes = plot_slices(data, axis=0, indices=[0, 5, 9])
    assert fig is not None
    plt.close(fig)


def test_plot_mask_zeros_false():
    """mask_zeros=False keeps zero values visible (for facies data)."""
    data = np.zeros((10, 10, 5))
    data[2:8, 2:8, 1:4] = 1.0
    data[4:6, 4:6, 1:4] = 2.0

    fig1, ax1 = plot_cube_slices(data, mask_zeros=False)
    assert fig1 is not None
    plt.close(fig1)

    fig2, axes2 = plot_slices(data, axis=2, mask_zeros=False)
    assert fig2 is not None
    plt.close(fig2)


def test_georules_cmap_registered():
    """GEORULES_CMAP should be importable and registered with matplotlib."""
    assert GEORULES_CMAP is not None
    assert GEORULES_CMAP.name == 'georules'
    # Should also be fetchable by name
    cmap = plt.get_cmap('georules')
    assert cmap is not None


def test_default_cmap_is_georules():
    """plot_cube_slices and plot_slices should default to georules cmap."""
    data = np.random.rand(10, 10, 5)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)

    fig2, axes2 = plot_slices(data, axis=2)
    assert fig2 is not None
    plt.close(fig2)
