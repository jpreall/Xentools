"""Xenium-specific reader helpers."""

from __future__ import annotations

import os
import sys
import importlib.util
from typing import Literal

import numpy as np
import pandas as pd


def _load_local_module(module_name, relative_path):
    """Load a sibling xentools module by file path when imported outside a package."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .zarr import _open_zarr_group_compat
except ImportError:
    _open_zarr_group_compat = _load_local_module(
        "_xentools_io_read_zarr",
        "zarr.py",
    )._open_zarr_group_compat

try:
    from ...utils.metadata import _encode_xenium_cell_ids
except ImportError:
    _encode_xenium_cell_ids = _load_local_module(
        "_xentools_utils_metadata",
        os.path.join("..", "..", "utils", "metadata.py"),
    )._encode_xenium_cell_ids

__all__ = [
    "create_polygon",
    "import_segmentation_xenium_parquet",
    "import_segmentation_xenium_zarr",
]


def create_polygon(df):
    """Create a polygon from Xenium boundary rows with `vertex_x`/`vertex_y` columns."""
    from shapely.geometry import Polygon

    return Polygon(zip(df.vertex_x, df.vertex_y))


def import_segmentation_xenium_parquet(boundaries_file):
    """
    Import cell or nucleus boundaries from a Xenium `*.parquet` boundary file.
    """
    import geopandas as gpd

    boundaries_df = pd.read_parquet(boundaries_file)
    boundaries_df.set_index("cell_id", inplace=True)
    boundaries_df.index = boundaries_df.index.astype("str")

    return gpd.GeoDataFrame(
        boundaries_df.groupby("cell_id").apply(create_polygon),
        columns=["geometry"],
    )


def import_segmentation_xenium_zarr(cells_zarr_file, kind: Literal["cell", "nucleus"] = "cell"):
    """
    Import cell or nucleus boundary polygons from `cells.zarr.zip`.
    """
    import geopandas as gpd
    import zarr
    from shapely.geometry import Polygon

    set_idx = "1" if kind == "cell" else "0"
    store = zarr.storage.ZipStore(cells_zarr_file, mode="r")
    try:
        root = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
        cell_ids_raw = root["cell_id"][:]
        cell_ids = _encode_xenium_cell_ids(cell_ids_raw[:, 0], cell_ids_raw[:, 1]).astype(str)

        grp = root[f"polygon_sets/{set_idx}"]
        cell_index = grp["cell_index"][:].astype(np.int64)
        num_vertices = grp["num_vertices"][:].astype(np.int64)
        vertices = grp["vertices"][:]
    finally:
        store.close()

    geometries = {}
    for i, (idx, nverts) in enumerate(zip(cell_index, num_vertices)):
        if nverts <= 0:
            continue
        flat = vertices[i, : 2 * int(nverts)]
        coords = flat.reshape(-1, 2)
        if len(coords) < 4:
            continue
        geometries[str(cell_ids[idx])] = Polygon(coords)

    gdf = gpd.GeoDataFrame(
        {"geometry": pd.Series(geometries, dtype=object)},
        geometry="geometry",
    )
    gdf.index.name = "cell_id"
    return gdf
