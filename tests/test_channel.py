import numpy as np
import pytest
from georules.layers.channel import ChannelLayer


def test_channel_shapes():
    layer = ChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)


def test_channel_has_nonzero_facies():
    layer = ChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=5)
    assert layer.facies.max() > 0, "Channel model should produce at least some channel facies"


def test_channel_poro_in_bounds():
    layer = ChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)
