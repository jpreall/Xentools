"""
Throwaway Datashader experiments for evaluating LazyTranscripts rendering.

This is intentionally not part of the package API. It is a small sandbox for
testing whether the current LazyTranscripts abstraction is a good handoff point
for Datashader-style rendering.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_repo_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_transcripts_mod = _load_repo_module("_scratch_xentools_core_transcripts", "core/transcripts.py")
LazyTranscripts = _transcripts_mod.LazyTranscripts
_load_zarr_gene_names = _transcripts_mod._load_zarr_gene_names

RGB_CHANNEL_COLORS = {
    "R": "#ff0000",
    "G": "#00ff00",
    "B": "#0000ff",
}


def prepare_env(tmp_root: str = "/tmp") -> None:
    """
    Set writable cache directories before importing datashader/numba/matplotlib.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tmp_root, "numba"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp_root, "mpl"))


def load_bundled_lazy_transcripts(folder: str | os.PathLike | None = None, *, verbose: bool = False) -> LazyTranscripts:
    """
    Convenience loader for the repo's bundled Xenium test dataset.
    """
    repo_root = Path(__file__).resolve().parents[1]
    folder = Path(folder) if folder is not None else repo_root / "files" / "xenium_v1_testdata"
    gene_names = _load_zarr_gene_names(str(folder))
    return LazyTranscripts(str(folder / "transcripts.zarr.zip"), gene_names, verbose=verbose)


def materialize_points(
    transcripts: LazyTranscripts,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    genes: str | Iterable[str] | None = None,
    quality: str = "all",
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Query LazyTranscripts into a DataFrame suitable for Datashader Canvas.points().
    """
    if bounds is None:
        df = transcripts.query(genes=genes, quality=quality)
    else:
        xmin, xmax, ymin, ymax = map(float, bounds)
        df = transcripts.query(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            genes=genes,
            quality=quality,
        )

    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df.loc[:, keep].copy()

    required = ["x_location", "y_location"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transcript frame is missing required coordinate columns: {missing}")

    df = df.copy()
    df["x_location"] = pd.to_numeric(df["x_location"], errors="coerce")
    df["y_location"] = pd.to_numeric(df["y_location"], errors="coerce")
    df = df.dropna(subset=["x_location", "y_location"])

    return df


def datashade_points(
    df: pd.DataFrame,
    *,
    x: str = "x_location",
    y: str = "y_location",
    width: int = 600,
    height: int = 600,
    bounds: tuple[float, float, float, float] | None = None,
    by: str | None = None,
    cmap=None,
):
    """
    Aggregate a point DataFrame with Datashader and return the aggregate + image.
    """
    prepare_env()
    import datashader as ds
    import datashader.transfer_functions as tf

    if cmap is None:
        cmap = ["black", "orange", "yellow", "white"]

    if bounds is None:
        x_range = (float(df[x].min()), float(df[x].max()))
        y_range = (float(df[y].min()), float(df[y].max()))
    else:
        xmin, xmax, ymin, ymax = bounds
        x_range = (float(xmin), float(xmax))
        y_range = (float(ymin), float(ymax))

    cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)

    if by is None:
        agg = cvs.points(df, x, y)
        agg = agg.isel({agg.dims[0]: slice(None, None, -1)})
        img = tf.shade(agg, cmap=cmap)
    else:
        if not isinstance(df[by].dtype, pd.CategoricalDtype):
            df = df.copy()
            df[by] = pd.Categorical(df[by])
        agg = cvs.points(df, x, y, ds.count_cat(by))
        agg = agg.isel({agg.dims[0]: slice(None, None, -1)})
        img = tf.shade(agg, color_key=cmap)

    img = tf.set_background(img, "black")
    return agg, img


def _normalize_gene_sets(gene_sets: dict[str, Iterable[str]]):
    if not isinstance(gene_sets, dict) or not gene_sets:
        raise ValueError("gene_sets must be a non-empty dict of channel_name -> iterable of genes")
    if len(gene_sets) > 3:
        raise ValueError("At most 3 gene sets are supported for RGB-like blending.")
    normalized = {}
    for name, genes in gene_sets.items():
        if isinstance(genes, str):
            normalized[name] = [genes]
        else:
            normalized[name] = list(genes)
        if len(normalized[name]) == 0:
            raise ValueError(f"Gene set '{name}' is empty.")
    return normalized


def _assign_gene_set_channels(df: pd.DataFrame, gene_sets: dict[str, Iterable[str]], *, source_col: str = "feature_name") -> pd.DataFrame:
    normalized = _normalize_gene_sets(gene_sets)
    gene_to_channel = {}
    for channel_name, genes in normalized.items():
        for gene in genes:
            gene_to_channel.setdefault(gene, channel_name)

    out = df.copy()
    out["gene_set"] = out[source_col].map(gene_to_channel)
    out = out[out["gene_set"].notna()].copy()
    out["gene_set"] = pd.Categorical(out["gene_set"], categories=list(normalized.keys()), ordered=True)
    return out


def _default_rgb_color_key(gene_sets: dict[str, Iterable[str]]):
    names = list(_normalize_gene_sets(gene_sets).keys())
    colors = list(RGB_CHANNEL_COLORS.values())[: len(names)]
    return dict(zip(names, colors))


def datashade_gene_sets(
    transcripts: LazyTranscripts,
    *,
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    color_key: dict[str, str] | None = None,
):
    normalized = _normalize_gene_sets(gene_sets)
    flat_genes = list(dict.fromkeys(g for genes in normalized.values() for g in genes))
    df = materialize_points(
        transcripts,
        bounds=bounds,
        genes=flat_genes,
        quality=quality,
    )
    df = _assign_gene_set_channels(df, normalized)
    agg, img = datashade_points(
        df,
        width=width,
        height=height,
        bounds=bounds,
        by="gene_set",
        cmap=_default_rgb_color_key(normalized) if color_key is None else color_key,
    )
    return {
        "dataframe": df,
        "aggregate": agg,
        "image": img,
        "n_points": len(df),
        "bounds": bounds,
        "columns": list(df.columns),
        "color_key": _default_rgb_color_key(normalized) if color_key is None else color_key,
    }


def probe_lazy_transcripts(
    transcripts: LazyTranscripts,
    *,
    bounds: tuple[float, float, float, float],
    genes: str | Iterable[str] | dict[str, Iterable[str]] | None = None,
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    by: str | None = None,
):
    """
    End-to-end probe from LazyTranscripts query to Datashader aggregate.
    """
    if isinstance(genes, dict):
        return datashade_gene_sets(
            transcripts,
            bounds=bounds,
            gene_sets=genes,
            quality=quality,
            width=width,
            height=height,
        )

    df = materialize_points(
        transcripts,
        bounds=bounds,
        genes=genes,
        quality=quality,
    )
    agg, img = datashade_points(
        df,
        width=width,
        height=height,
        bounds=bounds,
        by=by,
    )
    return {
        "dataframe": df,
        "aggregate": agg,
        "image": img,
        "n_points": len(df),
        "bounds": bounds,
        "columns": list(df.columns),
    }
