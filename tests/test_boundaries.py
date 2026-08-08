from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import pytest


def _boundary_bounds(xdata, pad=25.0):
    centroid = xdata.cell_boundaries.geometry.centroid.iloc[0]
    return (
        float(centroid.x - pad),
        float(centroid.x + pad),
        float(centroid.y - pad),
        float(centroid.y + pad),
    )


def test_plot_boundaries_standalone_is_visible_on_dark_background(xdata):
    ax = xdata.plot_boundaries(kind="cell", bounds=_boundary_bounds(xdata))

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    assert not ax.axison


def test_plot_boundaries_bounded_query_preserves_lazy_full_boundaries(xenium_testdata):
    import xentools

    xdata = xentools.XenData(str(xenium_testdata), verbose=False)
    centroid = xdata.adata.obs[["x_centroid", "y_centroid"]].iloc[0]
    bounds = (
        float(centroid["x_centroid"] - 100),
        float(centroid["x_centroid"] + 100),
        float(centroid["y_centroid"] - 100),
        float(centroid["y_centroid"] + 100),
    )

    ax = xdata.plot_boundaries(kind="cell", bounds=bounds)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert not xdata.cell_boundaries.loaded


def test_plot_boundaries_uses_lazy_zarr_boundary_subset(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        boundary_source="zarr",
    )
    centroid = xdata.adata.obs[["x_centroid", "y_centroid"]].iloc[0]
    bounds = (
        float(centroid["x_centroid"] - 100),
        float(centroid["x_centroid"] + 100),
        float(centroid["y_centroid"] - 100),
        float(centroid["y_centroid"] + 100),
    )

    ax = xdata.plot_boundaries(kind="cell", bounds=bounds)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert not xdata.cell_boundaries.loaded


def test_plot_boundaries_standalone_respects_explicit_bounds(xdata):
    bounds = _boundary_bounds(xdata)
    ax = xdata.plot_boundaries(kind="cell", bounds=bounds)

    assert ax.get_xlim() == pytest.approx(bounds[:2])
    assert ax.get_ylim() == pytest.approx((bounds[3], bounds[2]))


def test_plot_boundaries_overlay_does_not_replace_axes_styling_or_limits(xdata):
    fig, ax = plt.subplots()
    del fig
    ax.set_facecolor("white")
    ax.set_xlim(10, 20)
    ax.set_ylim(30, 40)
    facecolor = ax.get_facecolor()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    out = xdata.plot_boundaries(kind="cell", bounds=_boundary_bounds(xdata), ax=ax)

    assert out is ax
    assert ax.get_facecolor() == facecolor
    assert ax.get_xlim() == xlim
    assert ax.get_ylim() == ylim
    assert len(ax.collections) == 1


def test_plot_boundaries_overlay_uses_axes_limits_before_active_roi(xdata):
    from xentools.core.rois import ROI

    xdata.set_active_roi(ROI.from_bounds(0, 1, 0, 1, name="empty"))
    bounds = _boundary_bounds(xdata)
    fig, ax = plt.subplots()
    del fig
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    out = xdata.plot_boundaries(kind="cell", ax=ax)

    assert out is ax
    assert len(ax.collections) == 1


def test_plot_boundaries_color_legend_defaults_outside_plot(xdata):
    ax = xdata.plot_boundaries(
        kind="cell",
        color_by="Cluster",
        bounds=_boundary_bounds(xdata, pad=100),
    )

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Cluster"
    assert legend.get_bbox_to_anchor()._bbox.x0 > 1.0


def test_plot_boundaries_can_suppress_legend(xdata):
    ax = xdata.plot_boundaries(
        kind="cell",
        color_by="Cluster",
        bounds=_boundary_bounds(xdata, pad=100),
        show_legend=False,
    )

    assert ax.get_legend() is None


def test_plot_cells_colors_by_single_gene_expression(xdata):
    gene = xdata.adata.var_names[0]
    ax = xdata.plot_cells(
        genes=gene,
        bounds=_boundary_bounds(xdata, pad=100),
        face_alpha=0.8,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert len(ax.figure.axes) == 2
    assert ax.figure.axes[-1].get_ylabel() == gene


def test_plot_cells_sums_gene_list_expression(xdata):
    genes = list(xdata.adata.var_names[:2])
    ax = xdata.plot_cells(
        genes=genes,
        bounds=_boundary_bounds(xdata, pad=100),
        show_colorbar=False,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert len(ax.figure.axes) == 1


def test_plot_cells_can_color_by_obs_annotation(xdata):
    ax = xdata.plot_cells(
        color_by="Cluster",
        bounds=_boundary_bounds(xdata, pad=100),
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert ax.get_legend() is not None


def test_plot_cells_missing_color_by_raises_clear_error(xdata):
    with pytest.raises(ValueError, match="color_by='NotAColumn'.*xdata.adata.obs"):
        xdata.plot_cells(
            color_by="NotAColumn",
            bounds=_boundary_bounds(xdata, pad=100),
        )
