"""Boundary source selection and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import Literal


def _load_local_module(module_name, relative_path):
    """Load a repository module by file path when imported outside a package."""
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
    from ...core.boundaries import LazyBoundaryGeoDataFrame
except ImportError:
    LazyBoundaryGeoDataFrame = _load_local_module(
        "_xentools_core_boundaries_for_reader",
        os.path.join("..", "..", "core", "boundaries.py"),
    ).LazyBoundaryGeoDataFrame

try:
    from .xenium import (
        import_segmentation_xenium_parquet,
        import_segmentation_xenium_zarr,
    )
except ImportError:
    _xenium_read = _load_local_module(
        "_xentools_io_read_xenium_for_boundaries",
        "xenium.py",
    )
    import_segmentation_xenium_parquet = _xenium_read.import_segmentation_xenium_parquet
    import_segmentation_xenium_zarr = _xenium_read.import_segmentation_xenium_zarr


__all__ = [
    "BoundaryLoadResult",
    "load_xenium_boundaries",
    "resolve_boundary_source",
]


@dataclass
class BoundaryLoadResult:
    """Loaded or lazily registered cell/nucleus boundary data."""

    cell_boundaries: object | None
    nucleus_boundaries: object | None
    boundary_source: str | None
    lazy_boundaries: bool


def resolve_boundary_source(
    xenium_folder,
    boundary_source: Literal["auto", "parquet", "zarr"] = "auto",
):
    """Resolve boundary source from user preference and available files."""
    cell_boundaries_file = os.path.join(xenium_folder, "cell_boundaries.parquet")
    nuc_boundaries_file = os.path.join(xenium_folder, "nucleus_boundaries.parquet")
    cells_zarr_file = os.path.join(xenium_folder, "cells.zarr.zip")

    has_boundary_parquet = os.path.exists(cell_boundaries_file) and os.path.exists(nuc_boundaries_file)
    has_cells_zarr = os.path.exists(cells_zarr_file)

    if boundary_source == "parquet":
        if not has_boundary_parquet:
            raise FileNotFoundError(
                "Requested boundary_source='parquet' but boundary parquet files "
                f"were not found in {xenium_folder}"
            )
        return "parquet"

    if boundary_source == "zarr":
        if not has_cells_zarr:
            raise FileNotFoundError(
                "Requested boundary_source='zarr' but cells.zarr.zip was not "
                f"found in {xenium_folder}"
            )
        return "zarr"

    if boundary_source != "auto":
        raise ValueError("boundary_source must be 'auto', 'parquet', or 'zarr'.")

    if has_boundary_parquet:
        return "parquet"
    if has_cells_zarr:
        return "zarr"
    return None


def load_xenium_boundaries(
    xenium_folder,
    *,
    boundary_source: Literal["auto", "parquet", "zarr"] = "auto",
    lazy_boundaries: bool = True,
    verbose: bool = True,
) -> BoundaryLoadResult:
    """Load or lazily register Xenium cell and nucleus boundaries."""
    resolved_boundary_source = resolve_boundary_source(
        xenium_folder,
        boundary_source=boundary_source,
    )

    cell_boundaries_file = os.path.join(xenium_folder, "cell_boundaries.parquet")
    nuc_boundaries_file = os.path.join(xenium_folder, "nucleus_boundaries.parquet")
    cells_zarr_file = os.path.join(xenium_folder, "cells.zarr.zip")

    if resolved_boundary_source == "parquet":
        if lazy_boundaries:
            if verbose:
                print("Registering lazy parquet-backed cell boundaries")
            cell_boundaries = LazyBoundaryGeoDataFrame(
                lambda: import_segmentation_xenium_parquet(cell_boundaries_file),
                label="cell boundaries from cell_boundaries.parquet",
                subset_loader=lambda bounds: import_segmentation_xenium_parquet(
                    cell_boundaries_file,
                    bounds=bounds,
                ),
                id_subset_loader=lambda cell_ids: import_segmentation_xenium_parquet(
                    cell_boundaries_file,
                    cell_ids=cell_ids,
                ),
            )
            nucleus_boundaries = LazyBoundaryGeoDataFrame(
                lambda: import_segmentation_xenium_parquet(nuc_boundaries_file),
                label="nucleus boundaries from nucleus_boundaries.parquet",
                subset_loader=lambda bounds: import_segmentation_xenium_parquet(
                    nuc_boundaries_file,
                    bounds=bounds,
                ),
                id_subset_loader=lambda cell_ids: import_segmentation_xenium_parquet(
                    nuc_boundaries_file,
                    cell_ids=cell_ids,
                ),
            )
        else:
            if verbose:
                print("Reading in cell boundaries")
            cell_boundaries = import_segmentation_xenium_parquet(cell_boundaries_file)

            if verbose:
                print("Reading in nucleus boundaries")
            nucleus_boundaries = import_segmentation_xenium_parquet(nuc_boundaries_file)

    elif resolved_boundary_source == "zarr":
        if lazy_boundaries:
            if verbose:
                print("Registering lazy Zarr-backed cell boundaries")
            cell_boundaries = LazyBoundaryGeoDataFrame(
                lambda: import_segmentation_xenium_zarr(cells_zarr_file, kind="cell"),
                label="cell boundaries from cells.zarr.zip",
                subset_loader=lambda bounds: import_segmentation_xenium_zarr(
                    cells_zarr_file,
                    kind="cell",
                    bounds=bounds,
                ),
                id_subset_loader=lambda cell_ids: import_segmentation_xenium_zarr(
                    cells_zarr_file,
                    kind="cell",
                    cell_ids=cell_ids,
                ),
            )
            nucleus_boundaries = LazyBoundaryGeoDataFrame(
                lambda: import_segmentation_xenium_zarr(cells_zarr_file, kind="nucleus"),
                label="nucleus boundaries from cells.zarr.zip",
                subset_loader=lambda bounds: import_segmentation_xenium_zarr(
                    cells_zarr_file,
                    kind="nucleus",
                    bounds=bounds,
                ),
                id_subset_loader=lambda cell_ids: import_segmentation_xenium_zarr(
                    cells_zarr_file,
                    kind="nucleus",
                    cell_ids=cell_ids,
                ),
            )
        else:
            if verbose:
                print("Reading in cell boundaries from cells.zarr.zip")
            cell_boundaries = import_segmentation_xenium_zarr(cells_zarr_file, kind="cell")

            if verbose:
                print("Reading in nucleus boundaries from cells.zarr.zip")
            nucleus_boundaries = import_segmentation_xenium_zarr(cells_zarr_file, kind="nucleus")

    else:
        cell_boundaries = None
        nucleus_boundaries = None

    return BoundaryLoadResult(
        cell_boundaries=cell_boundaries,
        nucleus_boundaries=nucleus_boundaries,
        boundary_source=resolved_boundary_source,
        lazy_boundaries=bool(lazy_boundaries and resolved_boundary_source is not None),
    )
