from __future__ import annotations

import matplotlib.axes
import matplotlib.colors as mcolors
import pandas as pd
import pytest


def _bounds(xdata):
    xmin, xmax = map(float, xdata.frame[0])
    ymin, ymax = map(float, xdata.frame[1])
    return xmin, xmax, ymin, ymax


def _genes(xdata, n=3):
    genes = xdata.trans["feature_name"].dropna().astype(str).unique().tolist()
    assert len(genes) >= n
    return genes[:n]


def test_xendata_points_returns_axes_and_sampled_data(xdata):
    genes = _genes(xdata)

    with pytest.warns(RuntimeWarning, match="sampled"):
        ax, plotted = xdata.points(
            genes=genes,
            bounds=_bounds(xdata),
            max_points=5,
            random_state=0,
            return_data=True,
            show_legend=False,
        )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(plotted) == 5
    assert len(ax.collections) >= 1


def test_xendata_plot_points_alias_matches_points(xdata):
    assert xdata.plot_points.__func__ is xdata.points.__func__


def test_xentools_pl_points_supports_gene_set_coloring(xdata):
    import xentools

    genes = _genes(xdata)
    ax, plotted = xentools.pl.points(
        xdata,
        genes={"A": [genes[0]], "B": genes[1:3]},
        bounds=_bounds(xdata),
        max_points=100,
        return_data=True,
        warn_on_sample=False,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert set(plotted["feature_name"]).issubset(set(genes))
    assert ax.get_legend() is not None
    assert all(len(collection.get_edgecolors()) == 0 for collection in ax.collections)


def test_points_color_overrides_auto_palette_when_genes_are_supplied(xdata):
    genes = _genes(xdata, n=1)
    ax = xdata.plot_points(
        genes=genes,
        color="yellow",
        bounds=_bounds(xdata),
        max_points=20,
        show_legend=False,
        warn_on_sample=False,
    )

    yellow = mcolors.to_rgba("yellow", alpha=0.75)
    facecolors = ax.collections[-1].get_facecolors()
    assert len(facecolors) > 0
    assert tuple(facecolors[0]) == pytest.approx(yellow)


def test_points_defaults_to_assigned_transcripts_for_dataframe():
    import xentools

    df = pd.DataFrame(
        {
            "x_location": [0.0, 1.0, 2.0],
            "y_location": [0.0, 1.0, 2.0],
            "feature_name": ["A", "A", "A"],
            "cell_id": ["cell1", "UNASSIGNED", "cell2"],
        }
    )

    _ax, plotted = xentools.pl.points(
        df,
        genes="A",
        max_points=None,
        return_data=True,
        show_legend=False,
    )

    assert plotted["cell_id"].tolist() == ["cell1", "cell2"]


def test_points_can_include_unassigned_transcripts_for_dataframe():
    import xentools

    df = pd.DataFrame(
        {
            "x_location": [0.0, 1.0, 2.0],
            "y_location": [0.0, 1.0, 2.0],
            "feature_name": ["A", "A", "A"],
            "cell_id": ["cell1", "UNASSIGNED", "cell2"],
        }
    )

    _ax, plotted = xentools.pl.points(
        df,
        genes="A",
        max_points=None,
        assigned_only=False,
        return_data=True,
        show_legend=False,
    )

    assert plotted["cell_id"].tolist() == ["cell1", "UNASSIGNED", "cell2"]


def test_points_composes_on_existing_axes_and_preserves_limits(xdata):
    genes = _genes(xdata)
    bounds = _bounds(xdata)
    ax = xdata.splat(
        genes=genes,
        bounds=bounds,
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
    )
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()

    returned = xdata.points(
        genes=genes[:1],
        bounds=bounds,
        ax=ax,
        max_points=20,
        show_legend=False,
        warn_on_sample=False,
    )

    assert returned is ax
    assert ax.get_xlim() == old_xlim
    assert ax.get_ylim() == old_ylim
    assert len(ax.collections) >= 1


def test_lazy_transcripts_points_streaming_sample(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    genes = list(xdata.trans.gene_names[:3])

    ax, plotted = xdata.points(
        genes=genes,
        bounds=tuple(map(float, [xdata.xmin, xdata.xmax, xdata.ymin, xdata.ymax])),
        max_points=10,
        random_state=0,
        return_data=True,
        show_legend=False,
        warn_on_sample=False,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(plotted) <= 10
    assert set(plotted.columns) == {"x_location", "y_location", "feature_name"}
