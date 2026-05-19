from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest


def _test_genes(xdata, n=3):
    genes = xdata.trans["feature_name"].dropna().astype(str).unique().tolist()
    assert len(genes) >= n
    return genes[:n]


def _coarse_bounds(xdata):
    xmin, xmax = map(float, xdata.frame[0])
    ymin, ymax = map(float, xdata.frame[1])
    return xmin, xmax, ymin, ymax


def test_xendata_splat_returns_axes_by_default(xdata):
    genes = _test_genes(xdata)

    ax = xdata.splat(
        genes,
        bounds=_coarse_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1


def test_xendata_splat_can_return_display_and_raw_arrays(xdata):
    genes = _test_genes(xdata)
    kwargs = dict(
        genes=genes,
        bounds=_coarse_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
    )

    display = xdata.splat(**kwargs, return_array=True)
    raw = xdata.splat(**kwargs, return_array="raw")

    assert display.shape == raw.shape
    assert display.ndim == 3
    assert display.shape[-1] == len(genes)
    assert np.isfinite(display).all()
    assert display.min() >= 0
    assert display.max() <= 1
    assert np.issubdtype(raw.dtype, np.floating)


def test_xendata_splat_single_gene_uses_scalar_default_gain(xdata):
    gene = _test_genes(xdata, n=1)[0]

    display = xdata.splat(
        gene,
        bounds=_coarse_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
        return_array=True,
    )

    assert display.ndim == 3
    assert display.shape[-1] == 1


def test_xentools_pl_splat_matches_default_axes_contract(xdata):
    import xentools

    genes = _test_genes(xdata)
    ax = xentools.pl.splat(
        xdata.trans,
        genes=genes,
        bounds=_coarse_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
    )

    assert isinstance(ax, matplotlib.axes.Axes)


def test_xendata_splat_over_existing_image_uses_transparent_rgba_overlay(xdata):
    genes = _test_genes(xdata)
    bounds = _coarse_bounds(xdata)
    _, ax = plt.subplots()
    ax.imshow(
        np.zeros((20, 20)),
        extent=bounds,
        origin="lower",
        cmap="gray",
    )

    returned = xdata.plot_splat(
        genes,
        ax=ax,
        bounds=bounds,
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
        splat_alpha=0.5,
    )

    overlay = returned.images[-1].get_array()
    assert returned is ax
    assert len(ax.images) == 2
    assert overlay.ndim == 3
    assert overlay.shape[-1] == 4
    assert np.nanmax(overlay[..., 3]) <= 0.5


def test_splat_rejects_unknown_return_array_mode(xdata):
    gene = _test_genes(xdata, n=1)[0]

    with pytest.raises(ValueError, match="return_array"):
        xdata.splat(
            gene,
            bounds=_coarse_bounds(xdata),
            pixel_size_um=100,
            sigma_um=1,
            show_legend=False,
            return_array="both",
        )
