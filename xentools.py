#!/usr/bin/env python
import importlib.util
import os
import sys


def _load_local_module(module_name, relative_path):
    """Load a sibling module by file path for standalone `xentools.py` imports."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    if os.path.basename(module_path) == "__init__.py":
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[os.path.dirname(module_path)],
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .utils.misc import read_json
    from .utils.geometry import ROI_to_pixels, frame, um_to_pixels
except ImportError:
    read_json = _load_local_module("_xentools_utils_misc", os.path.join("utils", "misc.py")).read_json
    _geometry_utils_mod = _load_local_module("_xentools_utils_geometry", os.path.join("utils", "geometry.py"))
    ROI_to_pixels = _geometry_utils_mod.ROI_to_pixels
    frame = _geometry_utils_mod.frame
    um_to_pixels = _geometry_utils_mod.um_to_pixels


def _is_lazy_transcripts(obj) -> bool:
    """
    Detect LazyTranscripts-like objects without relying on class identity.

    Standalone ``sys.path.append(repo); import xentools`` imports can load the
    same source class under multiple module names, so ``isinstance`` is brittle.
    """
    return (
        hasattr(obj, "query")
        and callable(getattr(obj, "query"))
        and hasattr(obj, "_gene_names")
        and hasattr(obj, "_tile_meta")
    )

try:
    from .analysis.annotations import import_cell_annotations
    from .analysis.graph import build_spatial_graph
    from .analysis.neighborhoods import neighborhood_composition
    from .analysis.niches import build_niches, evaluate_niche_k_values
    from .analysis.normalization import normalize_tp10k
except ImportError:
    _annotations_mod = _load_local_module(
        "_xentools_analysis_annotations",
        os.path.join("analysis", "annotations.py"),
    )
    _graph_mod = _load_local_module(
        "_xentools_analysis_graph",
        os.path.join("analysis", "graph.py"),
    )
    _niches_mod = _load_local_module(
        "_xentools_analysis_niches",
        os.path.join("analysis", "niches.py"),
    )
    _neighborhoods_mod = _load_local_module(
        "_xentools_analysis_neighborhoods",
        os.path.join("analysis", "neighborhoods.py"),
    )
    _normalization_mod = _load_local_module(
        "_xentools_analysis_normalization",
        os.path.join("analysis", "normalization.py"),
    )
    import_cell_annotations = _annotations_mod.import_cell_annotations
    build_spatial_graph = _graph_mod.build_spatial_graph
    neighborhood_composition = _neighborhoods_mod.neighborhood_composition
    build_niches = _niches_mod.build_niches
    evaluate_niche_k_values = _niches_mod.evaluate_niche_k_values
    normalize_tp10k = _normalization_mod.normalize_tp10k

try:
    from .core.rois import (
        ROI,
        ROIClass,
        ROICollection,
        read_ROI_from_csv,
        read_ROI_from_geojson,
    )
    from .core.boundaries import LazyBoundaryGeoDataFrame
except ImportError:
    from core.rois import (
        ROI,
        ROIClass,
        ROICollection,
        read_ROI_from_csv,
        read_ROI_from_geojson,
    )
    from core.boundaries import LazyBoundaryGeoDataFrame

try:
    from .core.transcripts import LazyTranscripts
except ImportError:
    _transcripts_mod = _load_local_module("_xentools_core_transcripts", "core/transcripts.py")
    LazyTranscripts = _transcripts_mod.LazyTranscripts

try:
    from .core.xendata import XenData
except ImportError:
    from core.xendata import XenData


try:
    from . import analysis as _analysis_namespace
except ImportError:
    _analysis_namespace = _load_local_module("_xentools_analysis", os.path.join("analysis", "__init__.py"))

try:
    from . import pl as _pl_namespace
except ImportError:
    _pl_namespace = _load_local_module("_xentools_pl", os.path.join("pl", "__init__.py"))

try:
    from . import io as _io_namespace
except ImportError:
    _io_namespace = _load_local_module("_xentools_io", os.path.join("io", "__init__.py"))

try:
    from . import utils as _utils_namespace
except ImportError:
    _utils_namespace = _load_local_module("_xentools_utils", os.path.join("utils", "__init__.py"))

create_bins = _pl_namespace.create_bins
bin_expression = _pl_namespace.bin_expression
create_binned_image = _pl_namespace.create_binned_image
rasterize = _pl_namespace.rasterize
rasterize_rgb = _pl_namespace.rasterize_rgb
plot_binned_rgb = _pl_namespace.plot_binned_rgb
create_multilayer_image = _pl_namespace.create_multilayer_image
plot_binned_greyscale = _pl_namespace.plot_binned_greyscale
plot_boundaries = _pl_namespace.plot_boundaries
plot_cells = _pl_namespace.plot_cells
points = _pl_namespace.points
plot_points = _pl_namespace.plot_points
plot_splat = _pl_namespace.plot_splat
plot_image = _pl_namespace.plot_image
render = _pl_namespace.render
niche_heatmap = _pl_namespace.niche_heatmap
niche_map = _pl_namespace.niche_map
show_ome_tiff = _pl_namespace.show_ome_tiff
splat = _pl_namespace.splat
pl = _pl_namespace
analysis = _analysis_namespace
io = _io_namespace
utils = _utils_namespace
_make_gene_panel_df = _io_namespace.read._make_gene_panel_df
read_xen_panel = _io_namespace.read.read_xen_panel
read_xenium_to_anndata = _io_namespace.read.read_xenium_to_anndata
ROI_to_pixels = _utils_namespace.ROI_to_pixels
frame = _utils_namespace.frame
um_to_pixels = _utils_namespace.um_to_pixels

__all__ = [
    "io",
    "analysis",
    "XenData",
    "LazyTranscripts",
    "LazyBoundaryGeoDataFrame",
    "ROI",
    "ROIClass",
    "ROICollection",
    "read_xen_panel",
    "um_to_pixels",
    "read_xenium_to_anndata",
    "frame",
    "ROI_to_pixels",
    "import_cell_annotations",
    "read_ROI_from_csv",
    "read_ROI_from_geojson",
    "build_spatial_graph",
    "neighborhood_composition",
    "build_niches",
    "evaluate_niche_k_values",
    "normalize_tp10k",
    "create_bins",
    "bin_expression",
    "create_binned_image",
    "rasterize",
    "rasterize_rgb",
    "plot_binned_rgb",
    "create_multilayer_image",
    "plot_binned_greyscale",
    "plot_boundaries",
    "plot_cells",
    "points",
    "plot_points",
    "plot_splat",
    "plot_image",
    "render",
    "niche_heatmap",
    "niche_map",
    "show_ome_tiff",
    "splat",
    "pl",
    "utils",
]
