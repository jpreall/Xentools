from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest
import warnings


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


def test_splat_auto_pixel_size_targets_pixel_budget(xdata):
    gene = _test_genes(xdata, n=1)[0]

    with pytest.warns(UserWarning, match="auto-selected"):
        display = xdata.splat(
            gene,
            bounds=(0, 1000, 0, 1000),
            pixel_size_um="auto",
            target_pixels=10_000,
            sigma_um=0,
            smooth=False,
            show_legend=False,
            return_array=True,
        )

    assert display.shape[:2] == (100, 100)


def test_splat_explicit_pixel_size_overrides_target_pixels(xdata):
    gene = _test_genes(xdata, n=1)[0]

    display = xdata.splat(
        gene,
        bounds=(0, 1000, 0, 1000),
        pixel_size_um=100,
        target_pixels=10_000,
        sigma_um=0,
        smooth=False,
        show_legend=False,
        return_array=True,
    )

    assert display.shape[:2] == (10, 10)


def test_splat_auto_pixel_size_warning_respects_global_verbosity(xdata):
    import xentools

    gene = _test_genes(xdata, n=1)[0]
    old_verbosity = xentools.settings.verbosity
    xentools.settings.verbosity = 0
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            xdata.splat(
                gene,
                bounds=(0, 1000, 0, 1000),
                pixel_size_um="auto",
                target_pixels=10_000,
                sigma_um=0,
                smooth=False,
                show_legend=False,
                return_array=True,
            )
    finally:
        xentools.settings.verbosity = old_verbosity

    assert len(record) == 0


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


def test_xendata_splat_two_channels_displays_rgb_padded_image(xdata):
    genes = _test_genes(xdata, n=2)

    ax = xdata.splat(
        genes,
        bounds=_coarse_bounds(xdata),
        pixel_size_um=100,
        sigma_um=1,
        show_legend=False,
    )

    plotted = ax.images[-1].get_array()
    assert plotted.ndim == 3
    assert plotted.shape[-1] == 3


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


def test_splat_rejects_more_than_three_channels(xdata):
    genes = _test_genes(xdata, n=4)
    gene_sets = {f"set_{i}": [gene] for i, gene in enumerate(genes)}

    with pytest.raises(ValueError, match="at most 3"):
        xdata.splat(
            gene_sets,
            bounds=_coarse_bounds(xdata),
            pixel_size_um=100,
            sigma_um=1,
            show_legend=False,
        )


def test_binned_splat_returns_axes_and_arrays(xdata):
    genes = _test_genes(xdata)
    xdata.create_binned_adata(bin_size=100, include_features=genes)
    bounds = _coarse_bounds(xdata)

    ax = xdata.plot_binned_splat(
        genes,
        bounds=bounds,
        show_legend=False,
    )
    display = xdata.plot_binned_splat(
        genes,
        bounds=bounds,
        show_legend=False,
        return_array=True,
    )
    raw = xdata.plot_binned_splat(
        genes,
        bounds=bounds,
        show_legend=False,
        return_array="raw",
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
    assert display.shape == raw.shape
    assert display.shape[-1] == len(genes)
    assert display.min() >= 0
    assert display.max() <= 1


def test_binned_splat_over_existing_image_uses_transparent_rgba_overlay(xdata):
    genes = _test_genes(xdata)
    xdata.create_binned_adata(bin_size=100, include_features=genes)
    bounds = _coarse_bounds(xdata)
    _, ax = plt.subplots()
    ax.imshow(
        np.zeros((20, 20)),
        extent=bounds,
        origin="lower",
        cmap="gray",
    )

    returned = xdata.plot_binned_splat(
        genes,
        ax=ax,
        bounds=bounds,
        show_legend=False,
        splat_alpha=0.5,
    )

    overlay = returned.images[-1].get_array()
    assert returned is ax
    assert len(ax.images) == 2
    assert overlay.ndim == 3
    assert overlay.shape[-1] == 4
    assert np.nanmax(overlay[..., 3]) <= 0.5


def test_binned_splat_rejects_more_than_three_channels(xdata):
    genes = _test_genes(xdata, n=4)
    xdata.create_binned_adata(bin_size=100, include_features=genes)
    gene_sets = {f"set_{i}": [gene] for i, gene in enumerate(genes)}

    with pytest.raises(ValueError, match="at most 3"):
        xdata.plot_binned_splat(
            gene_sets,
            bounds=_coarse_bounds(xdata),
            show_legend=False,
        )


class _FakeLazyTranscripts:
    def __init__(self):
        self._gene_names = ["g1", "g2", "g3", "g4"]
        self._tile_meta = {"0,0": {"n": 4}}
        self.seen_channel_gene_lists = None
        self.seen_quality = None

    @property
    def frame(self):
        return np.array([[0.0, 10.0], [0.0, 10.0]])

    def query(self, *args, **kwargs):
        raise AssertionError("lazy splat fast path should avoid query()")

    def rasterize_channels(self, channel_gene_lists, bounds, pixel_size_um=1.0, quality="high"):
        self.seen_channel_gene_lists = channel_gene_lists
        self.seen_quality = quality
        return np.ones((10, 10, len(channel_gene_lists)), dtype=np.float32)


def test_splat_uses_lazy_rasterize_fast_path():
    import xentools

    lazy = _FakeLazyTranscripts()
    raw = xentools.pl.splat(
        lazy,
        genes={"R": ["g1", "g2"], "G": ["g3"], "B": ["g4"]},
        bounds=(0, 10, 0, 10),
        pixel_size_um=1,
        sigma_um=0,
        smooth=False,
        show_legend=False,
        return_array="raw",
        quality="all",
    )

    assert raw.shape == (10, 10, 3)
    assert lazy.seen_channel_gene_lists == [["g1", "g2"], ["g3"], ["g4"]]
    assert lazy.seen_quality == "all"


def test_splat_clips_large_gene_signatures_with_warning():
    import xentools

    lazy = _FakeLazyTranscripts()
    genes = {"R": ["g1", "g2", "g3"], "G": ["g4"]}

    with pytest.warns(RuntimeWarning, match="clipped large gene signatures"):
        xentools.pl.splat(
            lazy,
            genes=genes,
            bounds=(0, 10, 0, 10),
            pixel_size_um=1,
            sigma_um=0,
            smooth=False,
            show_legend=False,
            return_array="raw",
            max_signature_genes=2,
        )

    assert lazy.seen_channel_gene_lists == [["g1", "g2"], ["g4"]]


def test_splat_force_all_genes_disables_signature_clip():
    import xentools

    lazy = _FakeLazyTranscripts()
    genes = {"R": ["g1", "g2", "g3"], "G": ["g4"]}

    xentools.pl.splat(
        lazy,
        genes=genes,
        bounds=(0, 10, 0, 10),
        pixel_size_um=1,
        sigma_um=0,
        smooth=False,
        show_legend=False,
        return_array="raw",
        max_signature_genes=2,
        force_all_genes=True,
    )

    assert lazy.seen_channel_gene_lists == [["g1", "g2", "g3"], ["g4"]]
