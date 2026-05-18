"""Transcript source selection and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import Literal, Optional

import pandas as pd


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
    from ...core.transcripts import (
        LazyTranscripts,
        _count_transcripts_in_bundle,
        _detect_transcripts_format,
        _load_zarr_gene_names,
    )
except ImportError:
    _transcripts_mod = _load_local_module(
        "_xentools_core_transcripts_for_reader",
        os.path.join("..", "..", "core", "transcripts.py"),
    )
    LazyTranscripts = _transcripts_mod.LazyTranscripts
    _count_transcripts_in_bundle = _transcripts_mod._count_transcripts_in_bundle
    _detect_transcripts_format = _transcripts_mod._detect_transcripts_format
    _load_zarr_gene_names = _transcripts_mod._load_zarr_gene_names


__all__ = [
    "TranscriptLoadResult",
    "load_xenium_transcripts",
    "read_transcripts_parquet",
]


@dataclass
class TranscriptLoadResult:
    """Transcript data and source metadata selected for a Xenium bundle."""

    trans: pd.DataFrame | LazyTranscripts
    n_transcripts: int
    bundle_format: str
    transcript_format: str
    gene_names: list[str] | None = None


def read_transcripts_parquet(path):
    """Read ``transcripts.parquet`` and normalize byte-string feature names."""
    trans = pd.read_parquet(path)
    if "feature_name" in trans.columns and len(trans):
        sample = trans["feature_name"].iloc[:100]
        if sample.map(lambda x: isinstance(x, (bytes, bytearray))).any():
            trans["feature_name"] = trans["feature_name"].str.decode("utf-8")
    return trans


def load_xenium_transcripts(
    xenium_folder,
    *,
    transcript_source: Literal["auto", "zarr", "parquet"] = "auto",
    cache_threshold: int = 5_000_000,
    cache_max_bytes: Optional[int] = 512_000_000,
    eager_transcript_threshold: int = 20_000_000,
    verbose: bool = True,
) -> TranscriptLoadResult:
    """
    Resolve and load the transcript representation for a Xenium output folder.

    ``bundle_format`` reports what the data package provides/preferentially uses
    under automatic detection. ``transcript_format`` reports the actual backing
    representation selected for ``trans`` after applying user overrides and the
    eager parquet threshold.
    """
    bundle_format = _detect_transcripts_format(
        xenium_folder,
        transcript_source="auto",
    )
    n_transcripts = _count_transcripts_in_bundle(xenium_folder)

    resolved_source = transcript_source
    parquet_path = os.path.join(xenium_folder, "transcripts.parquet")
    if (
        transcript_source == "auto"
        and bundle_format == "zarr"
        and os.path.exists(parquet_path)
        and n_transcripts < int(eager_transcript_threshold)
    ):
        resolved_source = "parquet"
        if verbose:
            print(
                f"Dataset has {n_transcripts:,} transcripts "
                f"(< {int(eager_transcript_threshold):,}); using transcripts.parquet."
            )

    transcript_format = _detect_transcripts_format(
        xenium_folder,
        transcript_source=resolved_source,
    )

    gene_names = None
    if bundle_format == "zarr":
        gene_names = list(_load_zarr_gene_names(xenium_folder))
        if transcript_format == "zarr":
            zarr_path = os.path.join(xenium_folder, "transcripts.zarr.zip")
            if verbose:
                print(
                    f"Zarr format detected.  Indexing {len(gene_names):,} genes across "
                    "transcripts.zarr.zip..."
                )
            trans = LazyTranscripts(
                zarr_path,
                gene_names,
                cache_threshold=cache_threshold,
                cache_max_bytes=cache_max_bytes,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"  {trans.n_transcripts:,} transcripts in "
                    f"{len(trans._tile_meta)} spatial tiles."
                )
        else:
            if verbose:
                print("Using transcripts.parquet for rich transcript metadata...")
            trans = read_transcripts_parquet(parquet_path)
    else:
        if verbose:
            print("Reading Transcripts")
        trans = read_transcripts_parquet(parquet_path)

    return TranscriptLoadResult(
        trans=trans,
        n_transcripts=n_transcripts,
        bundle_format=bundle_format,
        transcript_format=transcript_format,
        gene_names=gene_names,
    )
