"""Reader modules for xentools IO."""

from .xenium import (
    create_polygon,
    import_segmentation_xenium_parquet,
    import_segmentation_xenium_zarr,
)
from .boundaries import (
    BoundaryLoadResult,
    load_xenium_boundaries,
    resolve_boundary_source,
)
from .cells import (
    CellMatrixLoadResult,
    _make_gene_panel_df,
    gene_panel_to_dataframe,
    load_xenium_cell_matrix,
    read_classic_analysis_clusters,
    read_xen_panel,
    read_xenium_to_anndata,
)
from .images import (
    _detect_linked_protein_images,
    _image_extent_um,
    _parse_ome_xml,
    _source_series_level_arrays,
    find_alignment_file,
    load_aligned_image,
)
from .metadata import (
    XeniumMetadata,
    discover_xenium_images,
    load_xenium_metadata,
    read_experiment_xenium,
)
from .loader import (
    XeniumLoadResult,
    load_xenium_folder,
)
from .transcripts import (
    TranscriptLoadResult,
    load_xenium_transcripts,
    read_transcripts_parquet,
)
from .zarr import (
    _open_zarr_group_compat,
    _read_analysis_zarr,
    _read_zarr_adata,
)

__all__ = [
    "_open_zarr_group_compat",
    "_read_analysis_zarr",
    "_read_zarr_adata",
    "_make_gene_panel_df",
    "gene_panel_to_dataframe",
    "BoundaryLoadResult",
    "CellMatrixLoadResult",
    "create_polygon",
    "import_segmentation_xenium_parquet",
    "import_segmentation_xenium_zarr",
    "load_xenium_cell_matrix",
    "load_xenium_boundaries",
    "read_classic_analysis_clusters",
    "read_xen_panel",
    "read_xenium_to_anndata",
    "resolve_boundary_source",
    "_detect_linked_protein_images",
    "_image_extent_um",
    "_parse_ome_xml",
    "_source_series_level_arrays",
    "find_alignment_file",
    "load_aligned_image",
    "XeniumMetadata",
    "XeniumLoadResult",
    "discover_xenium_images",
    "load_xenium_folder",
    "load_xenium_metadata",
    "read_experiment_xenium",
    "TranscriptLoadResult",
    "load_xenium_transcripts",
    "read_transcripts_parquet",
]
