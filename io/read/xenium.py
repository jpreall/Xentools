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


def _empty_boundary_geodataframe():
    import geopandas as gpd

    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry").rename_axis("cell_id")


def _boundary_rows_to_geodataframe(boundaries_df):
    import geopandas as gpd

    if boundaries_df.empty:
        return _empty_boundary_geodataframe()

    boundaries_df = boundaries_df.copy()
    boundaries_df.set_index("cell_id", inplace=True)
    boundaries_df.index = boundaries_df.index.astype("str")

    return gpd.GeoDataFrame(
        boundaries_df.groupby("cell_id").apply(create_polygon),
        columns=["geometry"],
    )


def _read_boundary_parquet_for_bounds(boundaries_file, bounds):
    """
    Read only boundary polygons likely to overlap a bounded viewport.

    The first parquet read identifies candidate cells with at least one vertex
    inside the requested bounds. The second read pulls all vertices for those
    cells, so polygons are complete rather than clipped to the viewport.
    """
    xmin, xmax, ymin, ymax = bounds
    columns = ["cell_id", "vertex_x", "vertex_y"]

    candidate_rows = pd.read_parquet(
        boundaries_file,
        columns=columns,
        filters=[
            ("vertex_x", ">=", xmin),
            ("vertex_x", "<=", xmax),
            ("vertex_y", ">=", ymin),
            ("vertex_y", "<=", ymax),
        ],
    )
    if candidate_rows.empty:
        return candidate_rows

    candidate_ids = candidate_rows["cell_id"].astype(str).unique().tolist()
    if not candidate_ids:
        return candidate_rows.iloc[0:0]

    try:
        return pd.read_parquet(
            boundaries_file,
            columns=columns,
            filters=[("cell_id", "in", candidate_ids)],
        )
    except Exception:
        all_rows = pd.read_parquet(boundaries_file, columns=columns)
        return all_rows[all_rows["cell_id"].astype(str).isin(candidate_ids)]


def _read_boundary_parquet_for_cell_ids(boundaries_file, cell_ids):
    columns = ["cell_id", "vertex_x", "vertex_y"]
    cell_ids = [str(cell_id) for cell_id in cell_ids]
    if not cell_ids:
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_parquet(
            boundaries_file,
            columns=columns,
            filters=[("cell_id", "in", cell_ids)],
        )
    except Exception:
        all_rows = pd.read_parquet(boundaries_file, columns=columns)
        return all_rows[all_rows["cell_id"].astype(str).isin(cell_ids)]


def import_segmentation_xenium_parquet(boundaries_file, bounds=None, cell_ids=None):
    """
    Import cell or nucleus boundaries from a Xenium `*.parquet` boundary file.

    When ``bounds`` is supplied as ``(xmin, xmax, ymin, ymax)``, only polygons
    with vertices in that viewport are constructed. This keeps ROI plotting
    fast without forcing full-boundary materialization at ``XenData`` init.
    """
    if cell_ids is not None:
        boundaries_df = _read_boundary_parquet_for_cell_ids(boundaries_file, cell_ids)
    elif bounds is None:
        boundaries_df = pd.read_parquet(boundaries_file)
    else:
        boundaries_df = _read_boundary_parquet_for_bounds(boundaries_file, bounds)
    return _boundary_rows_to_geodataframe(boundaries_df)


def import_segmentation_xenium_zarr(
    cells_zarr_file,
    kind: Literal["cell", "nucleus"] = "cell",
    bounds=None,
    cell_ids=None,
):
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
        encoded_cell_ids = _encode_xenium_cell_ids(cell_ids_raw[:, 0], cell_ids_raw[:, 1]).astype(str)
        selected_cell_indices = None
        if cell_ids is not None:
            requested = pd.Index([str(cell_id) for cell_id in cell_ids])
            selected_cell_indices = np.flatnonzero(pd.Index(encoded_cell_ids).isin(requested))
        elif bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            bboxes = root["bboxes"][:].reshape(-1, 4)
            selected_cell_indices = np.flatnonzero(
                (bboxes[:, 0] <= xmax)
                & (bboxes[:, 2] >= xmin)
                & (bboxes[:, 1] <= ymax)
                & (bboxes[:, 3] >= ymin)
            )

        grp = root[f"polygon_sets/{set_idx}"]
        cell_index = grp["cell_index"][:].astype(np.int64)
        num_vertices = grp["num_vertices"][:].astype(np.int64)
        if selected_cell_indices is None:
            row_indices = np.arange(len(cell_index), dtype=np.int64)
        else:
            row_indices = np.flatnonzero(np.isin(cell_index, selected_cell_indices))

        if len(row_indices) == 0:
            return _empty_boundary_geodataframe()

        cell_index = cell_index[row_indices]
        num_vertices = num_vertices[row_indices]
        vertices_array = grp["vertices"]
        try:
            vertices = vertices_array.get_orthogonal_selection((row_indices, slice(None)))
        except AttributeError:
            vertices = vertices_array[row_indices, :]
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
        geometries[str(encoded_cell_ids[idx])] = Polygon(coords)

    gdf = gpd.GeoDataFrame(
        {"geometry": pd.Series(geometries, dtype=object)},
        geometry="geometry",
    )
    gdf.index.name = "cell_id"
    return gdf
