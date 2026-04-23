"""Natural-language description of one generated reservoir sample.

Each caption is a short sentence summarising the key physics parameters of
the sample. Used as the ``caption`` column in the per-sample parquet
metadata table and as a conditioning signal for future
text-conditioned generative models.
"""


def caption_for(layer_type: str, params: dict) -> str:
    """Return one sentence describing the reservoir parameters."""
    fn = _DISPATCH.get(layer_type)
    if fn is None:
        raise ValueError(f"no caption template registered for {layer_type!r}")
    return fn(params)


def _lobe(p):
    # perm_ave is log10(mD) per LobeLayer.create_geology docstring.
    perm_mD = 10.0 ** float(p["perm_ave"])
    return (
        f"Turbidite lobe deposit with mean porosity {p['poro_ave']:.2f}, "
        f"mean permeability {perm_mD:.0f} mD (log10-std {p['perm_std']:.2f}), "
        f"net-to-gross {p['ntg']:.2f}, "
        f"lobe radius {p['rmin']:.0f}-{p['rmax']:.0f} m, "
        f"aspect ratio {p['asp']:.1f}, "
        f"thickness range {p['dhmin']}-{p['dhmax']} m, "
        f"bouma factor {p.get('bouma_factor', 0):.1f}, "
        f"upthinning {'enabled' if p.get('upthinning', True) else 'disabled'}."
    )


def _gaussian(p):
    # perm_ave is log10(mD) per GaussianLayer.create_geology docstring.
    perm_mD = 10.0 ** float(p["perm_ave"])
    return (
        f"Heterogeneous sand-shale body from sequential Gaussian simulation "
        f"with mean porosity {p['poro_ave']:.2f}, "
        f"mean permeability {perm_mD:.0f} mD (log10-std {p['perm_std']:.2f}), "
        f"net-to-gross {p['ntg']:.2f}, "
        f"nugget {p.get('nugget', 0.05):.3f}."
    )


def _meandering(p):
    return (
        f"Meandering fluvial channel belt with {int(p['n_channels'])} channels "
        f"of width {p['channel_width']:.0f} m, "
        f"meander scale {p.get('meander_scale', 1.2):.2f}, "
        f"slope {p.get('slope', 0.008):.4f}, "
        f"migration distance ratio {p.get('migration_distance_ratio', 1.0):.2f}."
    )


def _braided(p):
    return (
        f"Braided fluvial channel system with braidplain width "
        f"{p['braidplain_width']:.0f} m, "
        f"{int(p.get('n_channels', 24))} channels, "
        f"slope {p.get('slope', 0.008):.4f}."
    )


def _delta(p):
    return (
        f"River-mouth delta with {int(p.get('n_generations', 8))} bifurcation "
        f"generations, fan angle {p.get('fan_angle_deg', 95):.0f} deg, "
        f"feeder width {p.get('feeder_width', 60):.0f} m, "
        f"azimuth {p.get('azimuth', 0):.0f} deg, "
        f"progradation {p.get('progradation_fraction', 0):.2f}."
    )


_DISPATCH = {
    "lobe": _lobe,
    "gaussian": _gaussian,
    "meandering": _meandering,
    "braided": _braided,
    "delta": _delta,
}
