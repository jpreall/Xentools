"""Transcript storage helpers and lazy transcript access for xentools."""

from __future__ import annotations

import json
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
    from ..io.read.zarr import _open_zarr_group_compat
except ImportError:
    _open_zarr_group_compat = _load_local_module(
        "_xentools_io_read_zarr",
        os.path.join("..", "io", "read", "zarr.py"),
    )._open_zarr_group_compat

try:
    from ..utils.metadata import _encode_xenium_cell_ids
except ImportError:
    _encode_xenium_cell_ids = _load_local_module(
        "_xentools_utils_metadata",
        os.path.join("..", "utils", "metadata.py"),
    )._encode_xenium_cell_ids

try:
    from ..analysis.binning import _bin_transcript_dataframe
except ImportError:
    _bin_transcript_dataframe = _load_local_module(
        "_xentools_analysis_binning",
        os.path.join("..", "analysis", "binning.py"),
    )._bin_transcript_dataframe

__all__ = [
    "LazyTranscripts",
    "_detect_transcripts_format",
    "_count_transcripts_in_bundle",
    "_encode_xenium_cell_ids",
    "_load_zarr_gene_names",
    "_normalize_feature_selection",
    "_bin_transcript_dataframe",
]


def _detect_transcripts_format(folder, transcript_source: Literal["auto", "zarr", "parquet"] = "auto"):
    """
    Resolve which transcript backing store to use.

    `auto` preserves the historical preference for `transcripts.zarr.zip` when
    present, otherwise falls back to `transcripts.parquet`.
    """
    has_zarr = os.path.exists(os.path.join(folder, "transcripts.zarr.zip"))
    has_parquet = os.path.exists(os.path.join(folder, "transcripts.parquet"))

    if transcript_source == "zarr":
        if not has_zarr:
            raise FileNotFoundError(f"No transcripts.zarr.zip found in {folder}")
        return "zarr"

    if transcript_source == "parquet":
        if not has_parquet:
            raise FileNotFoundError(f"No transcripts.parquet found in {folder}")
        return "parquet"

    if has_zarr:
        return "zarr"
    if has_parquet:
        return "parquet"
    raise FileNotFoundError(
        f"Could not find transcripts.zarr.zip or transcripts.parquet in {folder}"
    )


def _count_transcripts_in_bundle(folder):
    """
    Return transcript count using file metadata without materializing the table.
    """
    zarr_path = os.path.join(folder, "transcripts.zarr.zip")
    parquet_path = os.path.join(folder, "transcripts.parquet")

    if os.path.exists(zarr_path):
        import zipfile

        with zipfile.ZipFile(zarr_path) as zf:
            if ".zattrs" in zf.namelist():
                attrs = json.loads(zf.read(".zattrs").decode())
                for key in ("number_rnas", "num_transcripts"):
                    if key in attrs:
                        return int(attrs[key])

    if os.path.exists(parquet_path):
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(parquet_path).metadata.num_rows)

    raise FileNotFoundError(
        f"Could not determine transcript count: no transcripts.zarr.zip or transcripts.parquet in {folder}"
    )


def _load_zarr_gene_names(folder):
    """
    Return the ordered gene name list whose index matches gene_identity values in
    transcripts.zarr.zip tiles.

    Xenium 5K Prime (zarr format v5.x): names are stored in transcripts.zarr.zip
    root .zattrs under the key 'gene_names'.

    Atera WTA (zarr format v6.x): no 'gene_names' in transcripts attrs; fall back
    to cell_feature_matrix.zarr.zip -> cell_features/.zattrs -> 'feature_keys'.
    """
    import zipfile

    trans_path = os.path.join(folder, "transcripts.zarr.zip")
    with zipfile.ZipFile(trans_path) as zf:
        if ".zattrs" in zf.namelist():
            attrs = json.loads(zf.read(".zattrs").decode())
            if "gene_names" in attrs:
                return attrs["gene_names"]

    cfm = os.path.join(folder, "cell_feature_matrix.zarr.zip")
    with zipfile.ZipFile(cfm) as zf:
        attrs = json.loads(zf.read("cell_features/.zattrs").decode())
    return attrs["feature_keys"]


def _normalize_feature_selection(features, available_features, arg_name="features"):
    """Return an ordered feature list after validating membership."""
    available = list(available_features)
    if features is None:
        return available
    if isinstance(features, str):
        features = [features]
    selected = list(features)
    missing = [f for f in selected if f not in set(available)]
    if missing:
        raise ValueError(
            f"Some values in {arg_name} are not present in the dataset: " + ", ".join(map(str, missing))
        )
    return selected


class LazyTranscripts:
    """
    Memory-efficient lazy accessor for Xenium/Atera transcript data in transcripts.zarr.zip.

    Transcripts are stored in a spatial tile grid (grids/0/{col},{row}/). Within each
    tile they are sorted by gene, so per-gene queries use the gene_offset index array for
    O(1) slice access rather than a full scan.
    """

    def __init__(self, zarr_path, gene_names, cache_threshold=5_000_000, verbose=True):
        self._path = str(zarr_path)
        self._gene_names = list(gene_names)
        self._gene_index = {g: i for i, g in enumerate(gene_names)}
        self.cache_threshold = cache_threshold
        self._query_cache = {}
        self._tile_meta = {}
        self._tile_size = (500.0, 500.0)
        self._tile_index_mode = "unknown"
        self._load_tile_metadata(verbose)

    def _load_tile_metadata(self, verbose=True):
        """
        Build a lightweight spatial index for level-0 transcript tiles.

        Recent Xenium/Atera transcript Zarr bundles provide an explicit map in
        ``grids/.zattrs``: level-0 ``grid_keys``, object counts, and ``grid_size``.
        Older or malformed bundles fall back to sampling one coordinate per tile.
        """
        import zipfile
        import zarr

        with zipfile.ZipFile(self._path) as zf:
            names = set(zf.namelist())
            if self._load_explicit_tile_metadata(zf, names):
                if verbose:
                    print(
                        f"  Loaded explicit spatial index for {len(self._tile_meta)} tiles "
                        f"from transcripts.zarr.zip metadata."
                    )
                return

            tile_keys = []
            for name in names:
                if name.startswith("grids/0/") and name.endswith("location/.zarray"):
                    parts = name.split("/")
                    key = parts[2]
                    if "," not in key:
                        continue
                    meta = json.loads(zf.read(name).decode())
                    n = meta["shape"][0]
                    if n > 0:
                        self._tile_meta[key] = {"n": n, "x": None, "y": None}
                        tile_keys.append(key)

        if not tile_keys:
            return

        if verbose:
            print(f"  Building spatial index for {len(tile_keys)} tiles...", end=" ", flush=True)

        store = zarr.storage.ZipStore(self._path, mode="r")
        try:
            grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
            for key in tile_keys:
                try:
                    first = grp[f"grids/0/{key}/location"][0]
                    self._tile_meta[key]["x"] = float(first[0])
                    self._tile_meta[key]["y"] = float(first[1])
                except Exception:
                    del self._tile_meta[key]
        finally:
            store.close()

        if verbose:
            print("done.")

        self._tile_size = self._estimate_tile_size()
        self._tile_index_mode = "sampled"

    def _load_explicit_tile_metadata(self, zf, names):
        """
        Load tile bounds from ``grids/.zattrs`` when the official grid map is present.
        """
        if "grids/.zattrs" not in names:
            return False

        try:
            attrs = json.loads(zf.read("grids/.zattrs").decode())
            grid_size = attrs["grid_size"]
            grid_keys = attrs["grid_keys"][0]
            grid_counts = attrs.get("grid_number_objects", [[]])[0]
        except Exception:
            return False

        if not grid_keys:
            return False

        try:
            if len(grid_size) == 1:
                dx = dy = float(grid_size[0])
            else:
                dx, dy = float(grid_size[0]), float(grid_size[1])
        except Exception:
            return False

        if dx <= 0 or dy <= 0:
            return False

        if len(grid_counts) != len(grid_keys):
            grid_counts = [None] * len(grid_keys)

        tile_meta = {}
        for key, count in zip(grid_keys, grid_counts):
            if "," not in str(key):
                return False
            try:
                col, row = (int(v) for v in str(key).split(",", maxsplit=1))
            except ValueError:
                return False

            if count is None:
                zarray_name = f"grids/0/{key}/location/.zarray"
                if zarray_name not in names:
                    continue
                try:
                    count = json.loads(zf.read(zarray_name).decode())["shape"][0]
                except Exception:
                    return False

            count = int(count)
            if count <= 0:
                continue

            x0 = col * dx
            x1 = (col + 1) * dx
            y0 = row * dy
            y1 = (row + 1) * dy
            tile_meta[str(key)] = {
                "n": count,
                "col": col,
                "row": row,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "x": (x0 + x1) / 2,
                "y": (y0 + y1) / 2,
            }

        if not tile_meta:
            return False

        self._tile_meta = tile_meta
        self._tile_size = (dx, dy)
        self._tile_index_mode = "explicit"
        return True

    def _estimate_tile_size(self):
        if not self._tile_meta:
            return (500.0, 500.0)

        xs_by_col = {}
        ys_by_row = {}
        for key, meta in self._tile_meta.items():
            if meta["x"] is None:
                continue
            col, row = map(int, key.split(","))
            xs_by_col.setdefault(col, []).append(meta["x"])
            ys_by_row.setdefault(row, []).append(meta["y"])

        def _median_step(d):
            vals = sorted(np.mean(v) for v in d.values())
            if len(vals) < 2:
                return 500.0
            return float(np.median(np.diff(vals)))

        dx = _median_step(xs_by_col)
        dy = _median_step(ys_by_row)
        return (max(dx, 50.0), max(dy, 50.0))

    def _candidate_tiles(self, xmin, xmax, ymin, ymax):
        if self._tile_index_mode == "explicit":
            if not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
                return [
                    key
                    for key, m in self._tile_meta.items()
                    if m["x1"] >= xmin and m["x0"] <= xmax and m["y1"] >= ymin and m["y0"] <= ymax
                ]

            dx, dy = self._tile_size
            col_min = int(np.floor(np.nextafter(xmin, -np.inf) / dx))
            col_max = int(np.floor(xmax / dx))
            row_min = int(np.floor(np.nextafter(ymin, -np.inf) / dy))
            row_max = int(np.floor(ymax / dy))

            keys = []
            for col in range(col_min, col_max + 1):
                for row in range(row_min, row_max + 1):
                    key = f"{col},{row}"
                    if key in self._tile_meta:
                        keys.append(key)
            return keys

        dx, dy = self._tile_size
        x0, x1 = xmin - dx, xmax + dx
        y0, y1 = ymin - dy, ymax + dy
        return [
            key
            for key, m in self._tile_meta.items()
            if m["x"] is not None and x0 <= m["x"] <= x1 and y0 <= m["y"] <= y1
        ]

    def query(self, xmin=None, xmax=None, ymin=None, ymax=None, genes=None, quality: str = "high") -> pd.DataFrame:
        """
        Return a DataFrame of transcripts within a bounding box.
        """
        import zarr

        xmin = -np.inf if xmin is None else float(xmin)
        xmax = np.inf if xmax is None else float(xmax)
        ymin = -np.inf if ymin is None else float(ymin)
        ymax = np.inf if ymax is None else float(ymax)

        if genes is None:
            gene_ids = None
        else:
            if isinstance(genes, str):
                genes = [genes]
            gene_ids = [self._gene_index[g] for g in genes if g in self._gene_index]
            if not gene_ids:
                return pd.DataFrame(columns=["x_location", "y_location", "feature_name"])

        cache_key = (
            round(xmin, 1),
            round(xmax, 1),
            round(ymin, 1),
            round(ymax, 1),
            tuple(sorted(gene_ids)) if gene_ids is not None else None,
            quality,
        )
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        tile_keys = self._candidate_tiles(xmin, xmax, ymin, ymax)

        store = zarr.storage.ZipStore(self._path, mode="r")
        frames = []
        try:
            grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
            for key in tile_keys:
                df = self._load_tile(grp, key, xmin, xmax, ymin, ymax, gene_ids, quality)
                if df is not None and len(df):
                    frames.append(df)
        finally:
            store.close()

        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["x_location", "y_location", "feature_name"])
        )

        if len(result) < self.cache_threshold:
            self._query_cache[cache_key] = result

        return result

    def _load_tile(self, grp, tile_key, xmin, xmax, ymin, ymax, gene_ids, quality):
        prefix = f"grids/0/{tile_key}"
        try:
            loc_arr = grp[f"{prefix}/location"]
            gid_arr = grp[f"{prefix}/gene_identity"]
            off_arr = grp[f"{prefix}/gene_offset"]
        except Exception:
            return None

        if quality == "high":
            q_slices = [(2, 3)]
        elif quality == "low":
            q_slices = [(0, 1)]
        else:
            q_slices = [(0, 1), (2, 3)]

        if gene_ids is not None:
            offsets = off_arr[:]
            raw_ranges = []
            for gid in gene_ids:
                if gid >= len(offsets):
                    continue
                for sc, ec in q_slices:
                    s, e = int(offsets[gid, sc]), int(offsets[gid, ec])
                    if e > s:
                        raw_ranges.append((s, e))

            if not raw_ranges:
                return None

            raw_ranges.sort()
            merged = [list(raw_ranges[0])]
            for s, e in raw_ranges[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])

            xs, ys, gs = [], [], []
            for s, e in merged:
                locs = loc_arr[s:e]
                gids = gid_arr[s:e, 0]
                mask = (
                    (locs[:, 0] >= xmin)
                    & (locs[:, 0] <= xmax)
                    & (locs[:, 1] >= ymin)
                    & (locs[:, 1] <= ymax)
                )
                if not mask.any():
                    continue
                xs.append(locs[mask, 0])
                ys.append(locs[mask, 1])
                gs.append(gids[mask])

            if not xs:
                return None
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            g_idx = np.concatenate(gs)
        else:
            locs = loc_arr[:]
            mask = (
                (locs[:, 0] >= xmin)
                & (locs[:, 0] <= xmax)
                & (locs[:, 1] >= ymin)
                & (locs[:, 1] <= ymax)
            )
            if not mask.any():
                return None
            x = locs[mask, 0]
            y = locs[mask, 1]
            g_idx = gid_arr[:, 0][mask]

        n = len(self._gene_names)
        names = np.array([self._gene_names[i] if i < n else "Unknown" for i in g_idx], dtype=object)
        return pd.DataFrame({"x_location": x, "y_location": y, "feature_name": names})

    @property
    def n_transcripts(self):
        return sum(m["n"] for m in self._tile_meta.values())

    def __len__(self):
        return self.n_transcripts

    @property
    def frame(self):
        if self._tile_index_mode == "explicit":
            x0 = [m["x0"] for m in self._tile_meta.values()]
            x1 = [m["x1"] for m in self._tile_meta.values()]
            y0 = [m["y0"] for m in self._tile_meta.values()]
            y1 = [m["y1"] for m in self._tile_meta.values()]
            if x0 and y0:
                return np.array([[min(x0), max(x1)], [min(y0), max(y1)]])

        xs = [m["x"] for m in self._tile_meta.values() if m["x"] is not None]
        ys = [m["y"] for m in self._tile_meta.values() if m["y"] is not None]
        if xs and ys:
            dx, dy = self._tile_size
            return np.array([[min(xs) - dx / 2, max(xs) + dx / 2], [min(ys) - dy / 2, max(ys) + dy / 2]])
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def to_dataframe(self, quality="high"):
        return self.query(quality=quality)

    def to_xarray_bins(self, bin_size=5, genes=None, bounds=None, quality="all", name="counts"):
        """
        Bin transcripts into a labeled xarray.DataArray with dims (feature_name, y, x).
        """
        import xarray as xr

        feature_names = _normalize_feature_selection(genes, self._gene_names, arg_name="genes")

        if bounds is None:
            df = self.query(genes=feature_names, quality=quality)
        else:
            xmin, xmax, ymin, ymax = map(float, bounds)
            df = self.query(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                genes=feature_names,
                quality=quality,
            )

        binned = _bin_transcript_dataframe(df, bin_size=bin_size, feature_names=feature_names)
        cube = np.zeros(binned["shape"], dtype=np.uint32)
        if len(binned["counts"]):
            cube[binned["feature_indices"], binned["y_indices"], binned["x_indices"]] = binned["counts"]

        attrs = {
            "bin_size": float(bin_size),
            "quality": quality,
            "units": "micron",
            "n_transcripts": int(binned["counts"].sum()),
        }
        if bounds is not None:
            attrs["bounds"] = tuple(map(float, bounds))

        return xr.DataArray(
            cube,
            dims=("feature_name", "y", "x"),
            coords={
                "feature_name": binned["feature_names"],
                "y": binned["y_centers_um"],
                "x": binned["x_centers_um"],
            },
            name=name,
            attrs=attrs,
        )

    def clear_cache(self):
        self._query_cache.clear()

    def __repr__(self):
        n = self.n_transcripts
        t = len(self._tile_meta)
        g = len(self._gene_names)
        cached = len(self._query_cache)
        return (
            f"LazyTranscripts({n:,} transcripts | {t} tiles | "
            f"{g:,} genes | tile_index={self._tile_index_mode} | "
            f"cache_threshold={self.cache_threshold:,} | "
            f"{cached} cached queries)"
        )
