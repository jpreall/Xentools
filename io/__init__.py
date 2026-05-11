"""Internal IO modules for xentools.

Canonical implementation now lives under `io.read` and `io.write`.
Top-level `io.*` modules remain as compatibility shims during migration.
"""

from .read import (
    _open_zarr_group_compat,
    _read_analysis_zarr,
    _read_zarr_adata,
    create_polygon,
    import_segmentation_xenium_parquet,
    import_segmentation_xenium_zarr,
)
from .write.xenium import (
    _invert_xen_gene_list_dict,
    extract_ome_channel_names,
    write_xenium_gene_groups,
)
from . import read, write

__all__ = [
    "_open_zarr_group_compat",
    "_read_analysis_zarr",
    "_read_zarr_adata",
    "create_polygon",
    "import_segmentation_xenium_parquet",
    "import_segmentation_xenium_zarr",
    "write_xenium_gene_groups",
    "_invert_xen_gene_list_dict",
    "extract_ome_channel_names",
    "read",
    "write",
]
