import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from georules.plotting import plot_cube_slices, plot_slices


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
