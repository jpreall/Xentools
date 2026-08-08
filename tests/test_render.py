from __future__ import annotations

import matplotlib.axes
import matplotlib.colors as mcolors
import pytest


def _bounds():
    return (0.0, 1000.0, 0.0, 1000.0)


def _genes(xdata, n=3):
    genes = xdata.trans["feature_name"].dropna().astype(str).unique().tolist()
    assert len(genes) >= n
    return genes[:n]


def _genes_in_bounds(xdata, bounds, n=1):
    xmin, xmax, ymin, ymax = bounds
    trans = xdata.trans
    in_bounds = trans[
        (trans["x_location"] >= xmin)
        & (trans["x_location"] <= xmax)
        & (trans["y_location"] >= ymin)
        & (trans["y_location"] <= ymax)
    ]
    if "cell_id" in in_bounds.columns:
        in_bounds = in_bounds[in_bounds["cell_id"].astype(str) != "UNASSIGNED"]
    genes = in_bounds["feature_name"].dropna().astype(str).unique().tolist()
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


def test_render_image_binned_splat_composite(xdata):
    genes = _genes(xdata)
    xdata.create_binned_adata(bin_size=100, include_features=genes)

    ax = xdata.render(
        image={"channel": "DAPI", "level": 4},
        binned_splat={"genes": genes, "gains": [2, 2, 2]},
        bounds=_bounds(),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) >= 2
    assert ax.images[-1].get_array().shape[-1] == 4
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text().strip() for text in legend.get_texts()]
    assert "Binned transcript density" in labels


def test_render_builds_single_combined_legend(xdata):
    genes = _genes(xdata)

    ax = xdata.render(
        image={"channel": "DAPI", "level": 4, "alpha": 0.5},
        splat={"genes": genes, "pixel_size_um": 100, "sigma_um": 1, "gains": [2, 2, 2]},
        points={"genes": genes[:1], "max_points": 10, "warn_on_sample": False},
        cells={"color_by": "Cluster", "face_alpha": 0.2, "edge_alpha": 0.1},
        bounds=_bounds(),
    )

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text().strip() for text in legend.get_texts()]
    assert "Images" in labels
    assert "Transcript density" in labels
    assert "Transcript points" in labels
    assert "Cells: Cluster" in labels
    assert "DAPI" in labels
    assert "DAPI (alpha=0.5)" not in labels
    assert genes[0] in labels


def test_render_combined_legend_supports_cell_type_labels(xdata):
    genes = _genes(xdata)
    original = xdata.adata.obs["Cell_Type"].copy() if "Cell_Type" in xdata.adata.obs.columns else None
    labels = ["ductal", "stromal"]
    xdata.adata.obs["Cell_Type"] = [labels[i % 2] for i in range(xdata.adata.n_obs)]

    try:
        ax = xdata.render(
            splat={"genes": genes, "pixel_size_um": 100, "sigma_um": 1},
            cells={"color_by": "Cell_Type", "face_alpha": 0.3},
            bounds=_bounds(),
        )
    finally:
        if original is None:
            del xdata.adata.obs["Cell_Type"]
        else:
            xdata.adata.obs["Cell_Type"] = original

    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text().strip() for text in legend.get_texts()]
    assert "Cells: Cell_Type" in legend_labels
    assert "ductal" in legend_labels
    assert "stromal" in legend_labels


def test_render_cell_legend_uses_centroid_fallback_when_boundary_ids_do_not_match(xdata):
    genes = _genes(xdata)
    original_boundaries = xdata.cell_boundaries
    original_cell_type = xdata.adata.obs["Cell_Type"].copy() if "Cell_Type" in xdata.adata.obs.columns else None
    labels = ["ductal", "stromal"]
    xdata.adata.obs["Cell_Type"] = [labels[i % 2] for i in range(xdata.adata.n_obs)]

    try:
        gdf = xdata.cell_boundaries.query_bounds(_bounds()).copy()
        gdf.index = [f"boundary_{i}" for i in range(len(gdf))]
        xdata.cell_boundaries = gdf

        ax = xdata.render(
            splat={"genes": genes, "pixel_size_um": 100, "sigma_um": 1},
            cells={"color_by": "Cell_Type", "face_alpha": 0.3},
            bounds=_bounds(),
        )
    finally:
        xdata.cell_boundaries = original_boundaries
        if original_cell_type is None:
            del xdata.adata.obs["Cell_Type"]
        else:
            xdata.adata.obs["Cell_Type"] = original_cell_type

    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text().strip() for text in legend.get_texts()]
    assert "Cells: Cell_Type" in legend_labels
    assert "ductal" in legend_labels
    assert "stromal" in legend_labels


def test_render_combined_legend_can_be_disabled(xdata):
    ax = xdata.render(
        image={"channel": "DAPI", "level": 4},
        cells=True,
        bounds=_bounds(),
        legend=False,
    )

    assert ax.get_legend() is None


def test_render_missing_cell_color_by_raises_clear_error(xdata):
    with pytest.raises(ValueError, match="color_by='NotAColumn'.*xdata.adata.obs"):
        xdata.render(
            splat={"genes": _genes(xdata), "pixel_size_um": 100, "sigma_um": 1},
            cells={"color_by": "NotAColumn", "face_alpha": 0.8},
            bounds=_bounds(),
        )


def test_render_points_color_controls_plot_and_legend(xdata):
    genes = _genes_in_bounds(xdata, _bounds(), n=1)
    ax = xdata.render(
        image={"channel": "DAPI", "level": 4},
        points={"genes": genes, "color": "yellow", "max_points": 20, "warn_on_sample": False},
        bounds=_bounds(),
    )

    yellow = mcolors.to_rgba("yellow", alpha=0.75)
    facecolors = ax.collections[-1].get_facecolors()
    assert len(facecolors) > 0
    assert tuple(facecolors[0]) == pytest.approx(yellow)

    legend = ax.get_legend()
    assert legend is not None
    point_handles = [h for h in legend.legend_handles if hasattr(h, "get_markerfacecolor")]
    assert point_handles
    assert mcolors.to_rgba(point_handles[-1].get_markerfacecolor(), alpha=0.75) == pytest.approx(yellow)


def test_render_vector_only_layers_use_requested_bounds(xdata):
    genes = _genes_in_bounds(xdata, _bounds(), n=1)
    original = xdata.adata.obs["Cell_Type"].copy() if "Cell_Type" in xdata.adata.obs.columns else None
    xdata.adata.obs["Cell_Type"] = ["A" if i % 2 else "B" for i in range(xdata.adata.n_obs)]

    try:
        ax = xdata.render(
            cells={"color_by": "Cell_Type", "face_alpha": 0.3},
            points={"genes": genes, "s": 1, "color": "pink", "max_points": 20, "warn_on_sample": False},
            bounds=_bounds(),
        )
    finally:
        if original is None:
            del xdata.adata.obs["Cell_Type"]
        else:
            xdata.adata.obs["Cell_Type"] = original

    assert len(ax.collections) >= 1
    assert tuple(float(v) for v in ax.get_xlim()) == _bounds()[:2]
    assert tuple(float(v) for v in ax.get_ylim()) == (_bounds()[3], _bounds()[2])


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
    display_extent = (bounds[0], bounds[1], bounds[3], bounds[2])
    assert image_extent == display_extent
    assert splat_extent == display_extent
    assert tuple(float(v) for v in ax.get_xlim()) == bounds[:2]
    assert tuple(float(v) for v in ax.get_ylim()) == (bounds[3], bounds[2])


def test_pl_render_matches_xendata_method(xdata):
    import xentools

    ax = xentools.pl.render(
        xdata,
        splat={"genes": _genes(xdata), "pixel_size_um": 100, "sigma_um": 1},
        bounds=_bounds(),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
