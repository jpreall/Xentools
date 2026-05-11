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


def test_plot_boundaries_standalone_respects_explicit_bounds(xdata):
    bounds = _boundary_bounds(xdata)
    ax = xdata.plot_boundaries(kind="cell", bounds=bounds)

    assert ax.get_xlim() == pytest.approx(bounds[:2])
    assert ax.get_ylim() == pytest.approx(bounds[2:])


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
