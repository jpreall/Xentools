from __future__ import annotations


def test_top_level_module_has_explicit_public_api():
    import xentools

    expected = {
        "XenData",
        "LazyTranscripts",
        "ROI",
        "ROICollection",
        "read_xen_panel",
        "splat",
        "show_ome_tiff",
        "plot_binned_rgb",
        "build_niches",
        "evaluate_niche_k_values",
        "normalize_tp10k",
        "pl",
        "io",
        "utils",
    }

    assert hasattr(xentools, "__all__")
    assert expected.issubset(set(xentools.__all__))
    for name in expected:
        assert hasattr(xentools, name)


def test_private_helpers_are_not_star_exports():
    import xentools

    exported = set(xentools.__all__)
    assert "_invert_xen_gene_list_dict" not in exported
    assert "_read_zarr_adata" not in exported
    assert "_open_zarr_group_compat" not in exported
    assert "read_json" not in exported
    assert "read_xen_essentials" not in exported
    assert "generate_palette" not in exported
    assert "plot_palette" not in exported
    assert "create_polygon" not in exported
    assert "import_segmentation_xenium_parquet" not in exported
    assert "import_segmentation_xenium_zarr" not in exported


def test_plotting_reexports_point_to_pl_namespace():
    import xentools

    assert xentools.splat is xentools.pl.splat
    assert xentools.show_ome_tiff is xentools.pl.show_ome_tiff
    assert xentools.plot_binned_rgb is xentools.pl.plot_binned_rgb
    assert xentools.rasterize is xentools.pl.rasterize


def test_palette_helpers_live_in_plotting_namespace():
    import xentools

    assert hasattr(xentools.pl, "generate_palette")
    assert hasattr(xentools.pl, "plot_palette")
    palette = xentools.pl.generate_palette(3)
    assert len(palette) == 3
    assert all(color.startswith("#") for color in palette)


def test_low_level_helpers_live_in_namespaces():
    import xentools

    assert hasattr(xentools.io.read, "create_polygon")
    assert hasattr(xentools.io.read, "import_segmentation_xenium_parquet")
    assert hasattr(xentools.io.read, "import_segmentation_xenium_zarr")
    assert hasattr(xentools.utils, "read_json")
    assert hasattr(xentools.utils, "frame")
    assert hasattr(xentools.utils, "um_to_pixels")
    assert xentools.frame is xentools.utils.frame
    assert xentools.um_to_pixels is xentools.utils.um_to_pixels


def test_xendata_is_exported_from_core_module():
    import xentools
    from xentools.core.xendata import XenData

    assert xentools.XenData is XenData
