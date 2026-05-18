"""Transcript storage helpers and lazy transcript access for xentools."""

from __future__ import annotations

import json
import os
import sys
import importlib.util
from collections import OrderedDict
from typing import Literal

import numpy as np
import pandas as pd


_TRANSCRIPT_COLUMNS = ["x_location", "y_location", "feature_name"]
_TRANSCRIPT_COLUMNS_WITH_Z = ["x_location", "y_location", "z_location", "feature_name"]


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


def _empty_transcript_dataframe(include_z=False):
    columns = _TRANSCRIPT_COLUMNS_WITH_Z if include_z else _TRANSCRIPT_COLUMNS
    return pd.DataFrame(columns=columns)


def _dataframe_nbytes(df):
    """Estimate DataFrame memory usage including Python object columns."""
    return int(df.memory_usage(index=True, deep=True).sum())


class LazyTranscripts:
    """
    Memory-efficient lazy accessor for Xenium/Atera transcript data in transcripts.zarr.zip.

    Transcripts are stored in a spatial tile grid (grids/0/{col},{row}/). Within each
    tile they are sorted by gene, so per-gene queries use the gene_offset index array for
    O(1) slice access rather than a full scan.
    """

    def __init__(
        self,
        zarr_path,
        gene_names,
        cache_threshold=5_000_000,
        cache_max_bytes=512_000_000,
        verbose=True,
    ):
        self._path = str(zarr_path)
        self._gene_names = list(gene_names)
        self._gene_index = {g: i for i, g in enumerate(gene_names)}
        self.cache_threshold = cache_threshold
        self.cache_max_bytes = None if cache_max_bytes is None else int(cache_max_bytes)
        self._query_cache = OrderedDict()
        self._query_cache_bytes = 0
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

    def _normalize_query_bounds(self, xmin=None, xmax=None, ymin=None, ymax=None):
        return (
            -np.inf if xmin is None else float(xmin),
            np.inf if xmax is None else float(xmax),
            -np.inf if ymin is None else float(ymin),
            np.inf if ymax is None else float(ymax),
        )

    def _normalize_gene_ids(self, genes=None):
        if genes is None:
            return None
        if isinstance(genes, str):
            genes = [genes]
        gene_ids = []
        seen = set()
        for gene in genes:
            if gene not in self._gene_index:
                continue
            gid = self._gene_index[gene]
            if gid in seen:
                continue
            seen.add(gid)
            gene_ids.append(gid)
        return gene_ids

    def _quality_slices(self, quality):
        if quality == "high":
            return [(2, 3)]
        if quality == "low":
            return [(0, 1)]
        return [(0, 1), (2, 3)]

    def _query_cache_key(self, xmin, xmax, ymin, ymax, gene_ids, quality, include_z=False):
        return (
            round(xmin, 1),
            round(xmax, 1),
            round(ymin, 1),
            round(ymax, 1),
            tuple(sorted(gene_ids)) if gene_ids is not None else None,
            quality,
            bool(include_z),
        )

    def _get_cached_query(self, cache_key):
        if cache_key not in self._query_cache:
            return None
        cached = self._query_cache.pop(cache_key)
        self._query_cache[cache_key] = cached
        return cached["data"]

    def _cache_query_result(self, cache_key, result):
        if len(result) >= self.cache_threshold:
            return

        nbytes = _dataframe_nbytes(result)
        if self.cache_max_bytes is not None and nbytes > self.cache_max_bytes:
            return

        if cache_key in self._query_cache:
            cached = self._query_cache.pop(cache_key)
            self._query_cache_bytes -= cached["nbytes"]

        while (
            self.cache_max_bytes is not None
            and self._query_cache
            and self._query_cache_bytes + nbytes > self.cache_max_bytes
        ):
            _old_key, old = self._query_cache.popitem(last=False)
            self._query_cache_bytes -= old["nbytes"]

        self._query_cache[cache_key] = {"data": result, "nbytes": nbytes}
        self._query_cache_bytes += nbytes

    def _tile_candidate_count(self, grp, tile_key, gene_ids, quality):
        """
        Return transcripts read from a tile before precise coordinate masking.

        For gene-filtered queries this uses ``gene_offset`` to count only the
        requested gene/quality slices. For all-gene queries, this reports the
        tile object count from metadata because the current all-gene query path
        reads the tile coordinate array as a whole.
        """
        if gene_ids is None:
            return int(self._tile_meta.get(tile_key, {}).get("n", 0))

        try:
            offsets = grp[f"grids/0/{tile_key}/gene_offset"][:]
        except Exception:
            return 0

        n = 0
        for gid in gene_ids:
            if gid >= len(offsets):
                continue
            for sc, ec in self._quality_slices(quality):
                s, e = int(offsets[gid, sc]), int(offsets[gid, ec])
                if e > s:
                    n += e - s
        return n

    def iter_tiles(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        genes=None,
        quality: str = "high",
        include_z: bool = False,
        include_empty: bool = False,
    ):
        """
        Yield ``(tile_key, DataFrame)`` pairs for transcripts matching a query.

        This is the streaming primitive for out-of-core reductions. Unlike
        :meth:`query`, it does not concatenate all matching tile results into a
        single DataFrame and it does not populate the query cache.

        Parameters mirror :meth:`query`. Set ``include_z=True`` to include the
        z coordinate stored as the third column of each tile's location array.
        Empty tile results are skipped by default; set ``include_empty=True`` to
        yield an empty DataFrame for each candidate tile.
        """
        import zarr

        xmin, xmax, ymin, ymax = self._normalize_query_bounds(xmin, xmax, ymin, ymax)
        gene_ids = self._normalize_gene_ids(genes)
        if genes is not None and not gene_ids:
            return

        tile_keys = self._candidate_tiles(xmin, xmax, ymin, ymax)

        store = zarr.storage.ZipStore(self._path, mode="r")
        try:
            grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
            for key in tile_keys:
                df = self._load_tile(grp, key, xmin, xmax, ymin, ymax, gene_ids, quality, include_z=include_z)
                if df is not None and len(df):
                    yield key, df
                elif include_empty:
                    yield key, _empty_transcript_dataframe(include_z=include_z)
        finally:
            store.close()

    def diagnose_query(self, xmin=None, xmax=None, ymin=None, ymax=None, genes=None, quality: str = "high"):
        """
        Return a dictionary describing how much work a lazy transcript query does.

        The diagnostic reports candidate tile count, candidate transcript count
        before precise coordinate masking, returned transcript count, and the
        survival fraction after coordinate masking. It is intended for measuring
        tile-pruning efficiency on real datasets and ROI sizes.
        """
        import zarr

        xmin, xmax, ymin, ymax = self._normalize_query_bounds(xmin, xmax, ymin, ymax)
        gene_ids = self._normalize_gene_ids(genes)
        if genes is not None and not gene_ids:
            return {
                "bounds": (xmin, xmax, ymin, ymax),
                "quality": quality,
                "requested_genes": list(genes) if not isinstance(genes, str) else [genes],
                "matched_genes": 0,
                "tile_index_mode": self._tile_index_mode,
                "tile_size": self._tile_size,
                "candidate_tiles": 0,
                "candidate_transcripts": 0,
                "returned_transcripts": 0,
                "survival_fraction": 0.0,
            }

        tile_keys = self._candidate_tiles(xmin, xmax, ymin, ymax)
        candidate_transcripts = 0
        returned_transcripts = 0

        store = zarr.storage.ZipStore(self._path, mode="r")
        try:
            grp = _open_zarr_group_compat(zarr, store, mode="r", force_v2=True)
            for key in tile_keys:
                candidate_transcripts += self._tile_candidate_count(grp, key, gene_ids, quality)
                df = self._load_tile(grp, key, xmin, xmax, ymin, ymax, gene_ids, quality)
                if df is not None:
                    returned_transcripts += len(df)
        finally:
            store.close()

        if genes is None:
            requested_genes = None
        elif isinstance(genes, str):
            requested_genes = [genes]
        else:
            requested_genes = list(genes)

        survival_fraction = (
            returned_transcripts / candidate_transcripts
            if candidate_transcripts
            else 0.0
        )
        return {
            "bounds": (xmin, xmax, ymin, ymax),
            "quality": quality,
            "requested_genes": requested_genes,
            "matched_genes": len(gene_ids) if gene_ids is not None else None,
            "tile_index_mode": self._tile_index_mode,
            "tile_size": self._tile_size,
            "candidate_tiles": len(tile_keys),
            "candidate_transcripts": int(candidate_transcripts),
            "returned_transcripts": int(returned_transcripts),
            "survival_fraction": float(survival_fraction),
        }

    def query(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        genes=None,
        quality: str = "high",
        include_z: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of transcripts within a bounding box.

        Parameters
        ----------
        include_z
            If ``True``, include ``z_location`` from the transcript zarr
            ``location`` array. The default is ``False`` to keep common
            plotting queries and cached DataFrames smaller.
        """
        xmin, xmax, ymin, ymax = self._normalize_query_bounds(xmin, xmax, ymin, ymax)
        gene_ids = self._normalize_gene_ids(genes)
        if genes is not None and not gene_ids:
            return _empty_transcript_dataframe(include_z=include_z)

        cache_key = self._query_cache_key(xmin, xmax, ymin, ymax, gene_ids, quality, include_z=include_z)
        cached = self._get_cached_query(cache_key)
        if cached is not None:
            return cached

        frames = [
            df
            for _tile_key, df in self.iter_tiles(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                genes=genes,
                quality=quality,
                include_z=include_z,
            )
        ]

        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else _empty_transcript_dataframe(include_z=include_z)
        )

        self._cache_query_result(cache_key, result)

        return result

    def _load_tile(self, grp, tile_key, xmin, xmax, ymin, ymax, gene_ids, quality, include_z=False):
        prefix = f"grids/0/{tile_key}"
        try:
            loc_arr = grp[f"{prefix}/location"]
            gid_arr = grp[f"{prefix}/gene_identity"]
            off_arr = grp[f"{prefix}/gene_offset"]
        except Exception:
            return None

        q_slices = self._quality_slices(quality)

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

            xs, ys, zs, gs = [], [], [], []
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
                if include_z:
                    zs.append(locs[mask, 2])
                gs.append(gids[mask])

            if not xs:
                return None
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            z = np.concatenate(zs) if include_z else None
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
            z = locs[mask, 2] if include_z else None
            g_idx = gid_arr[:, 0][mask]

        n = len(self._gene_names)
        names = np.array([self._gene_names[i] if i < n else "Unknown" for i in g_idx], dtype=object)
        if include_z:
            return pd.DataFrame({"x_location": x, "y_location": y, "z_location": z, "feature_name": names})
        return pd.DataFrame({"x_location": x, "y_location": y, "feature_name": names})

    @property
    def n_transcripts(self):
        return sum(m["n"] for m in self._tile_meta.values())

    @property
    def columns(self):
        """Columns returned by lazy transcript queries."""
        return pd.Index(_TRANSCRIPT_COLUMNS)

    @property
    def gene_names(self):
        """Ordered gene/feature names available for transcript queries."""
        return pd.Index(self._gene_names)

    @property
    def n_genes(self):
        """Number of gene/feature names available in the transcript zarr."""
        return len(self._gene_names)

    @property
    def shape(self):
        """
        DataFrame-like shape of the lazy transcript table.

        Only the lightweight transcript columns exposed by lazy queries are
        represented here; richer per-transcript metadata may require reading
        ``transcripts.parquet`` when available.
        """
        return (self.n_transcripts, len(_TRANSCRIPT_COLUMNS))

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
        self._query_cache_bytes = 0

    def cache_info(self):
        """
        Return current query-cache accounting.

        ``bytes`` is estimated from cached DataFrame memory usage, including
        object columns. ``max_bytes=None`` means no byte-level cap is enforced.
        """
        return {
            "entries": len(self._query_cache),
            "bytes": int(self._query_cache_bytes),
            "max_bytes": self.cache_max_bytes,
            "cache_threshold": self.cache_threshold,
        }

    def __repr__(self):
        n = self.n_transcripts
        t = len(self._tile_meta)
        g = self.n_genes
        cache = self.cache_info()
        frame = self.frame
        x0, x1 = frame[0]
        y0, y1 = frame[1]
        preview = ", ".join(map(str, self._gene_names[:6]))
        if g > 6:
            preview += ", ..."
        cache_limit = "unlimited" if cache["max_bytes"] is None else f"{cache['max_bytes']:,} bytes"
        return (
            "LazyTranscripts\n"
            f"  source: {os.path.basename(self._path)}\n"
            f"  shape: ({n:,}, {len(_TRANSCRIPT_COLUMNS)})  "
            f"[{', '.join(_TRANSCRIPT_COLUMNS)}]\n"
            f"  genes/features: {g:,}"
            + (f"  ({preview})" if preview else "")
            + "\n"
            f"  spatial extent: x={x0:,.1f}-{x1:,.1f} um, "
            f"y={y0:,.1f}-{y1:,.1f} um\n"
            f"  tile index: {self._tile_index_mode} ({t:,} tiles; "
            f"tile_size={self._tile_size[0]:g} x {self._tile_size[1]:g} um)\n"
            f"  cache: {cache['entries']} entries, {cache['bytes']:,} bytes "
            f"(limit={cache_limit}; row_threshold={self.cache_threshold:,})\n"
            "  optional columns: z_location via include_z=True\n"
            "  query with: .query(xmin=..., xmax=..., ymin=..., ymax=..., "
            "genes=[...], quality='high'|'low'|'all', include_z=False)"
        )
