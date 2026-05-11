"""
Throwaway Datashader experiments for evaluating LazyTranscripts rendering.

This is intentionally not part of the package API. It is a small sandbox for
testing whether the current LazyTranscripts abstraction is a good handoff point
for Datashader-style rendering.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Iterable
import zipfile

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
_open_zarr_group_compat = _transcripts_mod._open_zarr_group_compat

RGB_CHANNEL_COLORS = {
    "R": "#ff0000",
    "G": "#00ff00",
    "B": "#0000ff",
}

RGB_CHANNEL_VECTORS = {
    "R": (1.0, 0.0, 0.0),
    "G": (0.0, 1.0, 0.0),
    "B": (0.0, 0.0, 1.0),
}


def _is_lazy_transcripts(obj) -> bool:
    return all(hasattr(obj, attr) for attr in ("query", "frame", "n_transcripts"))


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


def summarize_transcripts_zarr_levels(zarr_path: str | os.PathLike):
    """
    Summarize the ``grids/<level>`` hierarchy in a transcripts.zarr.zip file.

    This is scratch/debug code for checking whether lower-resolution transcript
    representations exist and what they contain. It only reads zarr metadata.

    Examples
    --------
    >>> from scratch.datashader_probe import summarize_transcripts_zarr_levels
    >>> summary = summarize_transcripts_zarr_levels("files/xenium_v1_testdata/transcripts.zarr.zip")
    >>> summary["levels"]
    """
    zarr_path = Path(zarr_path)
    with zipfile.ZipFile(zarr_path) as zf:
        names = set(zf.namelist())
        grids_attrs = {}
        if "grids/.zattrs" in names:
            grids_attrs = json.loads(zf.read("grids/.zattrs").decode())

        levels = sorted(
            {
                name.split("/")[1]
                for name in names
                if name.startswith("grids/") and len(name.split("/")) > 2 and name.split("/")[1].isdigit()
            },
            key=int,
        )

        rows = []
        for level in levels:
            tile_keys = sorted(
                {
                    name.split("/")[2]
                    for name in names
                    if name.startswith(f"grids/{level}/")
                    and len(name.split("/")) > 4
                    and name.split("/")[2]
                    and name.split("/")[2] != ".zattrs"
                }
            )
            first_tile = tile_keys[0] if tile_keys else None
            arrays = {}
            if first_tile is not None:
                prefix = f"grids/{level}/{first_tile}"
                for array_name in ("location", "gene_identity", "gene_offset"):
                    zarray_name = f"{prefix}/{array_name}/.zarray"
                    if zarray_name in names:
                        meta = json.loads(zf.read(zarray_name).decode())
                        arrays[array_name] = {
                            "shape": tuple(meta.get("shape", ())),
                            "chunks": tuple(meta.get("chunks", ())),
                            "dtype": meta.get("dtype"),
                        }

            level_attrs = {}
            level_attrs_name = f"grids/{level}/.zattrs"
            if level_attrs_name in names:
                level_attrs = json.loads(zf.read(level_attrs_name).decode())

            rows.append(
                {
                    "level": int(level),
                    "n_tiles": len(tile_keys),
                    "first_tile": first_tile,
                    "arrays": arrays,
                    "level_attrs": level_attrs,
                }
            )

    return {
        "path": str(zarr_path),
        "grids_attrs": grids_attrs,
        "levels": rows,
    }


def print_transcripts_zarr_levels(zarr_path: str | os.PathLike):
    """
    Pretty-print ``summarize_transcripts_zarr_levels()`` for notebook exploration.
    """
    summary = summarize_transcripts_zarr_levels(zarr_path)
    print(f"transcripts.zarr.zip: {summary['path']}")
    attrs = summary["grids_attrs"]
    if attrs:
        print("grids attrs:")
        for key in ("grid_size", "number_grid_levels", "grid_keys", "grid_number_objects"):
            if key in attrs:
                value = attrs[key]
                if isinstance(value, list) and len(value) > 8:
                    value = value[:8] + [f"... ({len(attrs[key])} total)"]
                print(f"  {key}: {value}")
    for level in summary["levels"]:
        print(f"level {level['level']}: {level['n_tiles']} tiles")
        if level["first_tile"] is not None:
            print(f"  first tile: {level['first_tile']}")
        for name, meta in level["arrays"].items():
            print(
                f"  {name}: shape={meta['shape']} chunks={meta['chunks']} dtype={meta['dtype']}"
            )
    return summary


def read_transcript_zarr_tile(
    zarr_path: str | os.PathLike,
    *,
    level: int = 0,
    tile_key: str | None = None,
    gene_names: Iterable[str] | None = None,
    columns: tuple[str, ...] = ("x_location", "y_location", "feature_name"),
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Read one tile from a selected ``grids/<level>`` group into a DataFrame.

    This intentionally bypasses ``LazyTranscripts`` so you can directly compare
    how lower-resolution levels are encoded.
    """
    import zarr

    zarr_path = Path(zarr_path)
    store = zarr.storage.ZipStore(str(zarr_path), mode="r")
    try:
        grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
        level_grp = grp[f"grids/{int(level)}"]
        if tile_key is None:
            tile_key = sorted(level_grp.group_keys())[0]
        tile = level_grp[tile_key]

        loc = np.asarray(tile["location"][:])
        if max_rows is not None:
            loc = loc[: int(max_rows)]

        data = {}
        if "x_location" in columns:
            data["x_location"] = loc[:, 0]
        if "y_location" in columns:
            data["y_location"] = loc[:, 1]

        if "feature_name" in columns or "gene_identity" in columns:
            gene_identity = np.asarray(tile["gene_identity"][:])
            if max_rows is not None:
                gene_identity = gene_identity[: int(max_rows)]
            gene_ids = gene_identity[:, 0] if gene_identity.ndim > 1 else gene_identity
            if "gene_identity" in columns:
                data["gene_identity"] = gene_ids
            if "feature_name" in columns:
                if gene_names is None:
                    data["feature_name"] = gene_ids
                else:
                    gene_names_arr = np.asarray(list(gene_names), dtype=object)
                    valid = gene_ids < len(gene_names_arr)
                    names = np.full(len(gene_ids), "Unknown", dtype=object)
                    names[valid] = gene_names_arr[gene_ids[valid]]
                    data["feature_name"] = names

        df = pd.DataFrame(data)
        df.attrs["level"] = int(level)
        df.attrs["tile_key"] = tile_key
        df.attrs["zarr_path"] = str(zarr_path)
        return df
    finally:
        store.close()


def _level_grid_metadata(zarr_path: str | os.PathLike, level: int):
    """
    Return grid keys/counts/size for one transcript zarr level.
    """
    zarr_path = Path(zarr_path)
    with zipfile.ZipFile(zarr_path) as zf:
        attrs = json.loads(zf.read("grids/.zattrs").decode()) if "grids/.zattrs" in zf.namelist() else {}

    grid_keys_all = attrs.get("grid_keys", [])
    grid_counts_all = attrs.get("grid_number_objects", [])
    grid_size = attrs.get("grid_size", [250.0])
    grid_size = float(grid_size[0] if len(grid_size) == 1 else grid_size[int(level)])

    if grid_keys_all and isinstance(grid_keys_all[0], list):
        grid_keys = list(grid_keys_all[int(level)])
    else:
        grid_keys = list(grid_keys_all)

    if grid_counts_all and isinstance(grid_counts_all[0], list):
        grid_counts = list(grid_counts_all[int(level)])
    else:
        grid_counts = list(grid_counts_all)

    return {
        "grid_size": grid_size,
        "grid_keys": grid_keys,
        "grid_counts": grid_counts,
    }


def _candidate_level_tiles(
    zarr_path: str | os.PathLike,
    *,
    level: int,
    bounds: tuple[float, float, float, float] | None,
):
    meta = _level_grid_metadata(zarr_path, level)
    grid_size = meta["grid_size"]
    grid_keys = meta["grid_keys"]

    if bounds is None:
        return grid_keys

    xmin, xmax, ymin, ymax = map(float, bounds)
    out = []
    for key in grid_keys:
        col, row = (int(v) for v in str(key).split(",", maxsplit=1))
        x0 = col * grid_size
        x1 = (col + 1) * grid_size
        y0 = row * grid_size
        y1 = (row + 1) * grid_size
        if x1 >= xmin and x0 <= xmax and y1 >= ymin and y0 <= ymax:
            out.append(str(key))
    return out


def materialize_zarr_level_points(
    folder_or_zarr_path: str | os.PathLike,
    *,
    level: int,
    bounds: tuple[float, float, float, float] | None = None,
    genes: str | Iterable[str] | None = None,
    quality: str = "all",
) -> pd.DataFrame:
    """
    Query transcript points from a chosen ``grids/<level>`` representation.

    This mirrors the useful parts of ``LazyTranscripts.query()`` but exposes the
    zarr level for experiments. ``quality`` follows the current gene_offset
    convention: ``"high"`` uses columns 2:3, ``"low"`` uses 0:1, and ``"all"``
    uses both ranges.
    """
    import zarr

    path = Path(folder_or_zarr_path)
    if path.is_dir():
        folder = path
        zarr_path = folder / "transcripts.zarr.zip"
        gene_names = list(_load_zarr_gene_names(str(folder)))
    else:
        zarr_path = path
        gene_names = None

    if gene_names is None:
        raise ValueError("Pass a Xenium output folder, not just zarr_path, when filtering by gene names.")

    if genes is None:
        gene_ids = None
    else:
        if isinstance(genes, str):
            genes = [genes]
        gene_index = {g: i for i, g in enumerate(gene_names)}
        gene_ids = [gene_index[g] for g in genes if g in gene_index]
        if not gene_ids:
            return pd.DataFrame(columns=["x_location", "y_location", "feature_name"])

    if bounds is None:
        xmin = ymin = -np.inf
        xmax = ymax = np.inf
    else:
        xmin, xmax, ymin, ymax = map(float, bounds)

    if quality == "high":
        q_slices = [(2, 3)]
    elif quality == "low":
        q_slices = [(0, 1)]
    elif quality == "all":
        q_slices = [(0, 1), (2, 3)]
    else:
        raise ValueError("quality must be 'high', 'low', or 'all'.")

    tile_keys = _candidate_level_tiles(zarr_path, level=int(level), bounds=bounds)
    gene_names_arr = np.asarray(gene_names, dtype=object)
    frames = []

    store = zarr.storage.ZipStore(str(zarr_path), mode="r")
    try:
        grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
        for tile_key in tile_keys:
            prefix = f"grids/{int(level)}/{tile_key}"
            try:
                loc_arr = grp[f"{prefix}/location"]
                gid_arr = grp[f"{prefix}/gene_identity"]
                off_arr = grp[f"{prefix}/gene_offset"]
            except Exception:
                continue

            if gene_ids is None:
                locs = np.asarray(loc_arr[:])
                gids = np.asarray(gid_arr[:, 0])
            else:
                offsets = np.asarray(off_arr[:])
                ranges = []
                for gid in gene_ids:
                    if gid >= len(offsets):
                        continue
                    for sc, ec in q_slices:
                        start, end = int(offsets[gid, sc]), int(offsets[gid, ec])
                        if end > start:
                            ranges.append((start, end))
                if not ranges:
                    continue

                loc_chunks = []
                gid_chunks = []
                for start, end in ranges:
                    loc_chunks.append(np.asarray(loc_arr[start:end]))
                    gid_chunks.append(np.asarray(gid_arr[start:end, 0]))
                locs = np.concatenate(loc_chunks) if loc_chunks else np.empty((0, 3))
                gids = np.concatenate(gid_chunks) if gid_chunks else np.empty((0,), dtype=int)

            if len(locs) == 0:
                continue

            mask = (
                (locs[:, 0] >= xmin)
                & (locs[:, 0] <= xmax)
                & (locs[:, 1] >= ymin)
                & (locs[:, 1] <= ymax)
            )
            if not mask.any():
                continue

            gids = gids[mask]
            valid = gids < len(gene_names_arr)
            names = np.full(len(gids), "Unknown", dtype=object)
            names[valid] = gene_names_arr[gids[valid]]
            frames.append(
                pd.DataFrame(
                    {
                        "x_location": locs[mask, 0],
                        "y_location": locs[mask, 1],
                        "feature_name": names,
                        "zarr_level": int(level),
                        "tile_key": tile_key,
                    }
                )
            )
    finally:
        store.close()

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["x_location", "y_location", "feature_name", "zarr_level", "tile_key"])
    )


def datashade_gene_sets_from_zarr_level(
    folder_or_zarr_path: str | os.PathLike,
    *,
    level: int,
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    color_key: dict[str, str] | None = None,
):
    """
    Datashade gene sets from a selected transcript zarr level.
    """
    normalized = _normalize_gene_sets(gene_sets)
    flat_genes = list(dict.fromkeys(g for genes in normalized.values() for g in genes))
    df = materialize_zarr_level_points(
        folder_or_zarr_path,
        level=level,
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
        "level": int(level),
        "columns": list(df.columns),
        "color_key": _default_rgb_color_key(normalized) if color_key is None else color_key,
    }


def datashade_gene_sets_across_zarr_levels(
    folder_or_zarr_path: str | os.PathLike,
    *,
    levels: Iterable[int],
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
):
    """
    Return Datashader images for the same gene sets across multiple zarr levels.
    """
    return {
        int(level): datashade_gene_sets_from_zarr_level(
            folder_or_zarr_path,
            level=int(level),
            bounds=bounds,
            gene_sets=gene_sets,
            quality=quality,
            width=width,
            height=height,
        )
        for level in levels
    }


def compare_transcript_zarr_levels(
    folder_or_zarr_path: str | os.PathLike,
    *,
    levels: Iterable[int] | None = None,
    tile_key: str | None = None,
    max_rows: int | None = 10_000,
) -> dict[int, pd.DataFrame]:
    """
    Read the same tile key from multiple transcript zarr levels.

    ``folder_or_zarr_path`` can be either a Xenium output folder or the
    ``transcripts.zarr.zip`` path directly.
    """
    path = Path(folder_or_zarr_path)
    if path.is_dir():
        folder = path
        zarr_path = folder / "transcripts.zarr.zip"
        gene_names = _load_zarr_gene_names(str(folder))
    else:
        zarr_path = path
        gene_names = None

    summary = summarize_transcripts_zarr_levels(zarr_path)
    if levels is None:
        levels = [level["level"] for level in summary["levels"]]

    out = {}
    for level in levels:
        out[int(level)] = read_transcript_zarr_tile(
            zarr_path,
            level=int(level),
            tile_key=tile_key,
            gene_names=gene_names,
            max_rows=max_rows,
        )
    return out


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


def _normalize_density_channels(arr: np.ndarray, *, global_norm: bool = False, gains=1.0) -> np.ndarray:
    """
    Normalize smoothed channel arrays into display space.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("arr must have shape (height, width, channels)")

    if global_norm:
        m = float(arr.max())
        disp = arr / m if m > 0 else arr.copy()
    else:
        disp = np.zeros_like(arr)
        for k in range(arr.shape[-1]):
            m = float(arr[..., k].max())
            if m > 0:
                disp[..., k] = arr[..., k] / m

    if np.isscalar(gains):
        gains_arr = np.full(arr.shape[-1], float(gains), dtype=np.float32)
    else:
        gains_arr = np.asarray(tuple(gains), dtype=np.float32)
        if len(gains_arr) != arr.shape[-1]:
            raise ValueError("gains length must match number of channels")

    disp *= gains_arr.reshape(1, 1, -1)
    return np.clip(disp, 0, 1)


def _rgb_from_display_channels(display: np.ndarray, channel_names: Iterable[str]) -> np.ndarray:
    """
    Convert N display channels to RGB using R/G/B names when available.
    """
    display = np.asarray(display, dtype=np.float32)
    rgb = np.zeros((*display.shape[:2], 3), dtype=np.float32)
    fallback = list(RGB_CHANNEL_VECTORS.values())
    for i, name in enumerate(channel_names):
        color = RGB_CHANNEL_VECTORS.get(str(name), fallback[i % len(fallback)])
        rgb += display[..., i : i + 1] * np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb, 0, 1)


def _pil_from_rgb(rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")


def datashade_smoothed_gene_sets_from_dataframe(
    df: pd.DataFrame,
    *,
    gene_sets: dict[str, Iterable[str]],
    bounds: tuple[float, float, float, float],
    width: int = 600,
    height: int = 600,
    sigma_um: float = 2.0,
    sigma_px: float | None = None,
    global_norm: bool = False,
    gains=1.0,
):
    """
    Bin with Datashader, smooth the category-count rasters, and compose RGB.

    This is the Datashader analogue of ``splat()``: Datashader handles the
    point-to-pixel aggregation, then scipy applies Gaussian density smoothing.
    """
    prepare_env()
    import datashader as ds
    from scipy.ndimage import gaussian_filter

    normalized = _normalize_gene_sets(gene_sets)
    df = _assign_gene_set_channels(df, normalized)

    xmin, xmax, ymin, ymax = map(float, bounds)
    pixel_size_um = (xmax - xmin) / float(width)
    effective_sigma_px = float(sigma_um) / pixel_size_um if sigma_px is None else float(sigma_px)
    cvs = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=(xmin, xmax),
        y_range=(ymin, ymax),
    )

    if len(df) == 0:
        raw = np.zeros((int(height), int(width), len(normalized)), dtype=np.float32)
        agg = None
    else:
        agg = cvs.points(df, "x_location", "y_location", ds.count_cat("gene_set"))
        agg = agg.isel({agg.dims[0]: slice(None, None, -1)})
        raw = np.asarray(agg.transpose(agg.dims[0], agg.dims[1], "gene_set").values, dtype=np.float32)

    smoothed = (
        gaussian_filter(raw, sigma=(effective_sigma_px, effective_sigma_px, 0), mode="nearest")
        if effective_sigma_px
        else raw
    )
    display = _normalize_density_channels(smoothed, global_norm=global_norm, gains=gains)
    rgb = _rgb_from_display_channels(display, normalized.keys())
    image = _pil_from_rgb(rgb)

    return {
        "dataframe": df,
        "aggregate": agg,
        "raw": raw,
        "smoothed": smoothed,
        "display": display,
        "rgb": rgb,
        "image": image,
        "n_points": len(df),
        "bounds": bounds,
        "width": int(width),
        "height": int(height),
        "sigma_um": float(sigma_um),
        "sigma_px": effective_sigma_px,
        "pixel_size_um": pixel_size_um,
        "gene_sets": normalized,
    }


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


def datashade_smoothed_gene_sets(
    transcripts: LazyTranscripts,
    *,
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    sigma_um: float = 2.0,
    sigma_px: float | None = None,
    global_norm: bool = False,
    gains=1.0,
):
    """
    Query ``LazyTranscripts`` and render smoothed gene-set density with Datashader.
    """
    normalized = _normalize_gene_sets(gene_sets)
    flat_genes = list(dict.fromkeys(g for genes in normalized.values() for g in genes))
    df = materialize_points(
        transcripts,
        bounds=bounds,
        genes=flat_genes,
        quality=quality,
    )
    return datashade_smoothed_gene_sets_from_dataframe(
        df,
        gene_sets=normalized,
        bounds=bounds,
        width=width,
        height=height,
        sigma_um=sigma_um,
        sigma_px=sigma_px,
        global_norm=global_norm,
        gains=gains,
    )


def datashade_smoothed_gene_sets_from_zarr_level(
    folder_or_zarr_path: str | os.PathLike,
    *,
    level: int,
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    sigma_um: float = 2.0,
    sigma_px: float | None = None,
    global_norm: bool = False,
    gains=1.0,
):
    """
    Render smoothed gene-set density from a selected transcript zarr level.
    """
    normalized = _normalize_gene_sets(gene_sets)
    flat_genes = list(dict.fromkeys(g for genes in normalized.values() for g in genes))
    df = materialize_zarr_level_points(
        folder_or_zarr_path,
        level=level,
        bounds=bounds,
        genes=flat_genes,
        quality=quality,
    )
    res = datashade_smoothed_gene_sets_from_dataframe(
        df,
        gene_sets=normalized,
        bounds=bounds,
        width=width,
        height=height,
        sigma_um=sigma_um,
        sigma_px=sigma_px,
        global_norm=global_norm,
        gains=gains,
    )
    res["level"] = int(level)
    return res


def datashade_smoothed_gene_sets_across_zarr_levels(
    folder_or_zarr_path: str | os.PathLike,
    *,
    levels: Iterable[int],
    bounds: tuple[float, float, float, float],
    gene_sets: dict[str, Iterable[str]],
    quality: str = "all",
    width: int = 600,
    height: int = 600,
    sigma_um: float = 2.0,
    sigma_px: float | None = None,
    global_norm: bool = False,
    gains=1.0,
):
    """
    Compare smoothed Datashader density renderings across transcript zarr levels.
    """
    return {
        int(level): datashade_smoothed_gene_sets_from_zarr_level(
            folder_or_zarr_path,
            level=int(level),
            bounds=bounds,
            gene_sets=gene_sets,
            quality=quality,
            width=width,
            height=height,
            sigma_um=sigma_um,
            sigma_px=sigma_px,
            global_norm=global_norm,
            gains=gains,
        )
        for level in levels
    }


def benchmark_splat_vs_datashader_density(
    xdata,
    *,
    gene_sets: dict[str, Iterable[str]],
    bounds: tuple[float, float, float, float],
    width: int = 600,
    height: int | None = None,
    sigma_um: float = 2.0,
    sigma_px: float | None = None,
    quality: str = "all",
    global_norm: bool = False,
    gains=1.0,
):
    """
    Render the same gene sets with ``xdata.splat()`` and Datashader smoothing.

    Returns timing plus arrays/images. This is intended for notebook comparison,
    not production benchmarking.
    """
    height = int(width if height is None else height)
    xmin, xmax, ymin, ymax = map(float, bounds)
    pixel_size_um = (xmax - xmin) / float(width)
    effective_sigma_um = float(sigma_um) if sigma_px is None else float(sigma_px) * pixel_size_um

    t0 = time.perf_counter()
    splat_display = xdata.splat(
        gene_sets,
        bounds=bounds,
        pixel_size_um=pixel_size_um,
        sigma_um=effective_sigma_um,
        global_norm=global_norm,
        gains=gains,
        show_legend=False,
        return_array=True,
    )
    splat_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    if _is_lazy_transcripts(getattr(xdata, "trans", None)):
        ds_res = datashade_smoothed_gene_sets(
            xdata.trans,
            bounds=bounds,
            gene_sets=gene_sets,
            quality=quality,
            width=width,
            height=height,
            sigma_um=effective_sigma_um,
            sigma_px=sigma_px,
            global_norm=global_norm,
            gains=gains,
        )
    else:
        flat_genes = list(dict.fromkeys(g for genes in _normalize_gene_sets(gene_sets).values() for g in genes))
        df = xdata.trans[xdata.trans["feature_name"].isin(flat_genes)].copy()
        df = df[
            (df["x_location"] >= xmin)
            & (df["x_location"] <= xmax)
            & (df["y_location"] >= ymin)
            & (df["y_location"] <= ymax)
        ].copy()
        ds_res = datashade_smoothed_gene_sets_from_dataframe(
            df,
            gene_sets=gene_sets,
            bounds=bounds,
            width=width,
            height=height,
            sigma_um=effective_sigma_um,
            sigma_px=sigma_px,
            global_norm=global_norm,
            gains=gains,
        )
    datashader_seconds = time.perf_counter() - t0

    return {
        "splat_display": splat_display,
        "datashader": ds_res,
        "splat_seconds": splat_seconds,
        "datashader_seconds": datashader_seconds,
        "pixel_size_um": pixel_size_um,
        "sigma_um": effective_sigma_um,
        "sigma_px": float(effective_sigma_um) / pixel_size_um,
        "bounds": bounds,
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
