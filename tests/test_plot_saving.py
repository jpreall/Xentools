from __future__ import annotations


def _bounds(xdata):
    xmin, xmax = map(float, xdata.frame[0])
    ymin, ymax = map(float, xdata.frame[1])
    return xmin, xmax, ymin, ymax


def _genes(xdata, n=3):
    genes = xdata.trans["feature_name"].dropna().astype(str).unique().tolist()
    assert len(genes) >= n
    return genes[:n]


def _assert_saved(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_splat_can_save_figure(xdata, tmp_path):
    path = tmp_path / "plots" / "splat.png"

    ax = xdata.plot_splat(
        _genes(xdata),
        bounds=_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
        save=path,
    )

    assert ax is not None
    _assert_saved(path)


def test_plot_points_can_save_figure_with_return_data(xdata, tmp_path):
    path = tmp_path / "points.png"

    ax, plotted = xdata.plot_points(
        _genes(xdata, 1),
        bounds=_bounds(xdata),
        max_points=10,
        show_legend=False,
        warn_on_sample=False,
        return_data=True,
        save=path,
    )

    assert ax is not None
    assert len(plotted) <= 10
    _assert_saved(path)


def test_plot_cells_can_save_figure(xdata, tmp_path):
    path = tmp_path / "cells.png"

    ax = xdata.plot_cells(
        bounds=_bounds(xdata),
        max_cells=10,
        save=path,
    )

    assert ax is not None
    _assert_saved(path)


def test_render_can_save_composite(xdata, tmp_path):
    path = tmp_path / "render.png"

    ax = xdata.render(
        splat={"genes": _genes(xdata), "pixel_size_um": 100, "sigma_um": 1},
        points={"genes": _genes(xdata, 1), "max_points": 10, "show_legend": False, "warn_on_sample": False},
        bounds=_bounds(xdata),
        save=path,
    )

    assert ax is not None
    _assert_saved(path)


def test_plot_image_can_save_figure(xdata, tmp_path):
    path = tmp_path / "image.png"

    ax = xdata.plot_image(
        "DAPI",
        bounds=(0.0, 1000.0, 0.0, 1000.0),
        level=4,
        verbose=False,
        save=path,
    )

    assert ax.get_title() == "DAPI"
    _assert_saved(path)
