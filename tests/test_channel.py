import numpy as np
import pytest
from georules.layers.channel import MeanderingChannelLayer, BraidedChannelLayer, ChannelLayerBase


# === Existing tests (unchanged logic) ===

def test_channel_shapes():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)


def test_channel_has_nonzero_facies():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=5)
    assert layer.facies.max() > 0, "Channel model should produce at least some channel facies"


def test_channel_poro_in_bounds():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_meandering_explicit_name():
    layer = MeanderingChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(channel_width=40, n_channels=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.facies.max() > 0


def test_base_class_not_directly_usable():
    """ChannelLayerBase.create_geology should raise NotImplementedError."""
    base = ChannelLayerBase(nx=4, ny=4, nz=4, x_len=100, y_len=100,
                            z_len=10, top_depth=500)
    with pytest.raises(NotImplementedError):
        base.create_geology()


# === Braided channel tests ===

def test_braided_shapes():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=3, n_threads=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)
    assert layer.facies.shape == (64, 32, 16)


def test_braided_has_channel_facies():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=5, n_threads=3)
    assert (layer.facies == 2).any(), "Should have channel fill facies (2)"


def test_braided_has_bar_facies():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=5, n_threads=3)
    assert (layer.facies == 1).any(), "Should have braid bar facies (1)"


def test_braided_poro_in_bounds():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=3, n_threads=3)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_braided_active_matches_facies():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=3, n_threads=3)
    expected_active = (layer.facies > 0).astype(int)
    np.testing.assert_array_equal(layer.active, expected_active)


def test_braided_perm_positive_where_active():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=3, n_threads=3)
    active_perm = layer.perm_mat[layer.active == 1]
    assert np.all(active_perm > 0)


def test_braided_in_reservoir():
    """BraidedChannelLayer should be stackable in a Reservoir."""
    from georules.layers.gaussian import GaussianLayer
    from georules.reservoir import Reservoir

    g = GaussianLayer(nx=64, ny=32, nz=8, x_len=1024, y_len=512,
                      z_len=24, top_depth=1000)
    g.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                     perm_std=0.5, ntg=0.7)

    b = BraidedChannelLayer(nx=64, ny=32, nz=8, x_len=1024, y_len=512,
                            z_len=24, top_depth=1024)
    b.create_geology(braidplain_width=200, n_channels=3, n_threads=3)

    res = Reservoir([g, b])
    assert res.poro_mat.shape == (64, 32, 16)


# === Braided BBC engine unit tests ===

def _make_braided_engine(**kwargs):
    """Helper: create a braided engine with sensible defaults for testing."""
    from georules.layers._braided import braided
    defaults = dict(
        braidplain_width=200, n_threads=3,
        nx=64, ny=32, nz=16,
        xmn=8, ymn=8, xsiz=16, ysiz=16, zsiz=3,
        dwratio=0.15,
    )
    defaults.update(kwargs)
    return braided(**defaults)


def test_build_braided_path_returns_segments_and_bars():
    """_build_braided_path must return at least one segment and one bar."""
    np.random.seed(0)
    eng = _make_braided_engine()
    center_y = (eng.ymin + eng.ymax) / 2.0
    cx, cy = eng._generate_thread_streamline(y0=center_y, s=0.05)
    assert cx is not None, "Streamline generation failed"

    cy = eng._constrain_to_braidplain(cy, center_y)
    # Keep only in-grid portion
    in_grid = ((cx > eng.xmin) & (cx < eng.xmax) &
               (cy > eng.ymin) & (cy < eng.ymax))
    first = int(np.argmax(in_grid))
    last = len(in_grid) - 1 - int(np.argmax(in_grid[::-1]))
    cx, cy = cx[first:last + 1], cy[first:last + 1]

    segments, bars = eng._build_braided_path(cx, cy)
    assert len(segments) >= 1, "Should produce at least one segment"
    assert len(bars) >= 1, "Should produce at least one bar region"

    # Each segment is a (cx, cy, width) tuple
    for seg_cx, seg_cy, seg_w in segments:
        assert seg_cx.size == seg_cy.size
        assert seg_w > 0

    # Each bar is a (left_cx, left_cy, right_cx, right_cy) tuple
    for lcx, lcy, rcx, rcy in bars:
        assert lcx.size == rcx.size


def test_build_braided_path_short_trunk():
    """A trunk shorter than ~30 points should still return a valid segment."""
    eng = _make_braided_engine()
    short_cx = np.linspace(0, 100, 20)
    short_cy = np.ones(20) * 200
    segments, bars = eng._build_braided_path(short_cx, short_cy)
    # Too short for BBC → single trunk segment, no bars
    assert len(segments) == 1
    assert len(bars) == 0


def test_build_braided_path_sub_threads_narrower():
    """Sub-thread segments from BBC units should be narrower than trunk."""
    np.random.seed(1)
    eng = _make_braided_engine()
    center_y = (eng.ymin + eng.ymax) / 2.0
    cx, cy = eng._generate_thread_streamline(y0=center_y, s=0.05)
    cy = eng._constrain_to_braidplain(cy, center_y)
    in_grid = ((cx > eng.xmin) & (cx < eng.xmax) &
               (cy > eng.ymin) & (cy < eng.ymax))
    first = int(np.argmax(in_grid))
    last = len(in_grid) - 1 - int(np.argmax(in_grid[::-1]))
    cx, cy = cx[first:last + 1], cy[first:last + 1]

    segments, bars = eng._build_braided_path(cx, cy)
    widths = [w for _, _, w in segments]
    trunk_w = eng.thread_width
    # Should have at least some sub-thread segments narrower than trunk
    assert any(w < trunk_w for w in widths), \
        "BBC sub-threads should be narrower than the trunk"


def test_braided_bars_adjacent_to_channels():
    """Bar cells should appear near channel cells, not in isolation."""
    np.random.seed(2)
    layer = BraidedChannelLayer(
        nx=128, ny=64, nz=16, x_len=2048, y_len=1024, z_len=48,
        top_depth=1000,
    )
    layer.create_geology(braidplain_width=400, n_channels=5, n_threads=3)

    f = layer.facies
    bar_locs = np.argwhere(f == 1)
    if bar_locs.size == 0:
        pytest.skip("No bars generated (stochastic)")

    # For a sample of bar cells, check that at least one neighbour
    # within ±2 cells is a channel cell (facies==1)
    n_check = min(50, len(bar_locs))
    rng = np.random.default_rng(0)
    sample = rng.choice(len(bar_locs), n_check, replace=False)
    adjacent_count = 0
    for idx in sample:
        ix, iy, iz = bar_locs[idx]
        # 5x5 neighbourhood in x-y
        x_lo, x_hi = max(0, ix - 2), min(f.shape[0], ix + 3)
        y_lo, y_hi = max(0, iy - 2), min(f.shape[1], iy + 3)
        neighbourhood = f[x_lo:x_hi, y_lo:y_hi, iz]
        if (neighbourhood == 2).any():
            adjacent_count += 1

    # Most bar cells should be near a channel
    assert adjacent_count > n_check * 0.5, \
        f"Only {adjacent_count}/{n_check} bar cells are near channels"


def test_braided_z_coverage():
    """Channels should span most of the z-range, not cluster in a few layers."""
    np.random.seed(3)
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=5, n_threads=3)

    f = layer.facies
    z_with_channels = sum(1 for z in range(f.shape[2]) if (f[:, :, z] == 2).any())
    # Should occupy at least half the z-levels
    assert z_with_channels >= f.shape[2] // 2, \
        f"Channels only in {z_with_channels}/{f.shape[2]} z-layers"


def test_braided_only_valid_facies_codes():
    """Facies array should contain only codes 0, 1, and 2."""
    np.random.seed(4)
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=200, n_channels=3, n_threads=3)
    unique = set(np.unique(layer.facies))
    assert unique.issubset({0, 1, 2}), f"Unexpected facies codes: {unique}"
