"""Coordinator for reading a Xenium output folder."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import Literal


def _load_local_module(module_name, relative_path):
    """Load a sibling module by file path when imported outside a package."""
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
    from .boundaries import BoundaryLoadResult, load_xenium_boundaries
except ImportError:
    _boundaries = _load_local_module("_xentools_io_read_boundaries_for_loader", "boundaries.py")
    BoundaryLoadResult = _boundaries.BoundaryLoadResult
    load_xenium_boundaries = _boundaries.load_xenium_boundaries

try:
    from .cells import CellMatrixLoadResult, load_xenium_cell_matrix
except ImportError:
    _cells = _load_local_module("_xentools_io_read_cells_for_loader", "cells.py")
    CellMatrixLoadResult = _cells.CellMatrixLoadResult
    load_xenium_cell_matrix = _cells.load_xenium_cell_matrix

try:
    from .metadata import XeniumMetadata, load_xenium_metadata
except ImportError:
    _metadata = _load_local_module("_xentools_io_read_metadata_for_loader", "metadata.py")
    XeniumMetadata = _metadata.XeniumMetadata
    load_xenium_metadata = _metadata.load_xenium_metadata

try:
    from .transcripts import TranscriptLoadResult, load_xenium_transcripts
except ImportError:
    _transcripts = _load_local_module("_xentools_io_read_transcripts_for_loader", "transcripts.py")
    TranscriptLoadResult = _transcripts.TranscriptLoadResult
    load_xenium_transcripts = _transcripts.load_xenium_transcripts


__all__ = ["XeniumLoadResult", "load_xenium_folder"]


@dataclass
class XeniumLoadResult:
    """Grouped reader outputs for a Xenium folder."""

    transcripts: TranscriptLoadResult
    cells: CellMatrixLoadResult
    boundaries: BoundaryLoadResult
    metadata: XeniumMetadata


def load_xenium_folder(
    xenium_folder,
    *,
    transcript_source: Literal["auto", "zarr", "parquet"] = "auto",
    cache_threshold: int = 5_000_000,
    eager_transcript_threshold: int = 20_000_000,
    boundary_source: Literal["auto", "parquet", "zarr"] = "auto",
    lazy_boundaries: bool = True,
    include_non_gene_features: bool = False,
    verbose: bool = True,
) -> XeniumLoadResult:
    """
    Load all file-backed data needed to initialize a XenData object.
    """
    transcripts = load_xenium_transcripts(
        xenium_folder,
        transcript_source=transcript_source,
        cache_threshold=cache_threshold,
        eager_transcript_threshold=eager_transcript_threshold,
        verbose=verbose,
    )
    cells = load_xenium_cell_matrix(
        xenium_folder,
        bundle_format=transcripts.bundle_format,
        include_non_gene_features=include_non_gene_features,
        verbose=verbose,
    )
    boundaries = load_xenium_boundaries(
        xenium_folder,
        boundary_source=boundary_source,
        lazy_boundaries=lazy_boundaries,
        verbose=verbose,
    )
    metadata = load_xenium_metadata(xenium_folder)

    return XeniumLoadResult(
        transcripts=transcripts,
        cells=cells,
        boundaries=boundaries,
        metadata=metadata,
    )
