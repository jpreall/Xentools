from __future__ import annotations

import matplotlib.axes


def _bounds():
    return (0.0, 1000.0, 0.0, 1000.0)


def _genes(xdata, n=3):
    genes = xdata.trans["feature_name"].dropna().astype(str).unique().tolist()
    assert len(genes) >= n
    return genes[:n]


def test_render_image_splat_points_cells_composite(xdata):
    genes = _genes(xdata)

    ax = xdata.render(
        image={"channel": "DAPI", "level": 4},
        splat={"genes": genes, "pixel_size_um": 100, "sigma_um": 1, "gains": [2, 2, 2]},
        points={"genes": genes[:1], "max_points": 10, "show_legend": False, "warn_on_sample": False},
        cells=True,
        bounds=_bounds(),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) >= 2
    assert len(ax.collections) >= 1
    assert ax.images[-1].get_array().shape[-1] == 4


def test_render_multiple_images_uses_transparent_overlay(xdata):
    ax = xdata.render(
        images=[
            {"channel": "DAPI", "level": 4},
            {"channel": "DAPI", "level": 4, "color": "cyan", "alpha": 0.5},
        ],
        bounds=_bounds(),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 2
    overlay = ax.images[-1].get_array()
    assert overlay.ndim == 3
    assert overlay.shape[-1] == 4
    assert overlay[..., 3].max() <= 0.5


def test_render_background_image_alpha_dims_scalar_image(xdata):
    bounds = _bounds()
    full = xdata.render(
        image={"channel": "DAPI", "level": 4, "alpha": 1.0},
        bounds=bounds,
    )
    dim = xdata.render(
        image={"channel": "DAPI", "level": 4, "alpha": 0.25},
        bounds=bounds,
    )

    full_vmin, full_vmax = full.images[-1].get_clim()
    dim_vmin, dim_vmax = dim.images[-1].get_clim()
    assert dim_vmin == full_vmin
    assert dim_vmax > full_vmax


def test_render_out_of_image_bounds_keeps_layer_extents_aligned(xdata):
    genes = _genes(xdata)
    bounds = (-1000.0, 2000.0, -600.0, 2400.0)

    ax = xdata.render(
        image={"channel": "DAPI", "level": 2, "alpha": 0.3},
        splat={"genes": genes, "pixel_size_um": 10, "sigma_um": 1, "gains": [2, 2, 2]},
        cells={"edge_alpha": 0.1},
        bounds=bounds,
    )

    image_extent = tuple(float(v) for v in ax.images[0].get_extent())
    splat_extent = tuple(float(v) for v in ax.images[-1].get_extent())
    assert image_extent == bounds
    assert splat_extent == bounds
    assert tuple(float(v) for v in ax.get_xlim()) == bounds[:2]
    assert tuple(float(v) for v in ax.get_ylim()) == bounds[2:]


def test_pl_render_matches_xendata_method(xdata):
    import xentools

    ax = xentools.pl.render(
        xdata,
        splat={"genes": _genes(xdata), "pixel_size_um": 100, "sigma_um": 1},
        bounds=_bounds(),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
