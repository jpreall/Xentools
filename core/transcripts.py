"""Transcript storage helpers and lazy transcript access for xentools."""

from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "LazyTranscripts",
    "_detect_transcripts_format",
    "_count_transcripts_in_bundle",
    "_encode_xenium_cell_ids",
    "_load_zarr_gene_names",
    "_normalize_feature_selection",
    "_bin_transcript_dataframe",
]


def _open_zarr_group_compat(zarr_module, store, mode="r", *, force_v2=False):
    """
    Open a Zarr group across Zarr 2 and 3.

    Zarr 3 accepts ``zarr_format=2`` so we can emit/read v2 metadata explicitly.
    Zarr 2 rejects that keyword and already uses v2 metadata.
    """
    kwargs = {"store": store, "mode": mode}
    if force_v2:
        try:
            return zarr_module.open_group(**kwargs, zarr_format=2)
        except TypeError as exc:
            if "zarr_format" not in str(exc):
                raise
    return zarr_module.open_group(**kwargs)


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


def _encode_xenium_cell_ids(prefix_arr, suffix_arr):
    """Convert raw cell_id columns to Xenium hash strings (e.g. 'aaaabbbb-12345')."""
    prefix_arr = prefix_arr.astype(np.uint32)
    suffix_arr = suffix_arr.astype(np.uint32)
    hex_strs = np.char.zfill(np.vectorize(lambda x: format(x, "x"))(prefix_arr), 8)
    trans = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    shifted = np.char.translate(hex_strs, trans)
    return np.char.add(np.char.add(shifted, "-"), suffix_arr.astype(str))


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


def _bin_transcript_dataframe(df, bin_size, feature_names):
    """
    Aggregate a transcript table into a shared binning spec used by xarray and AnnData views.
    """
    feature_names = list(feature_names)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

    if df.empty:
        return {
            "feature_names": np.asarray(feature_names, dtype=object),
            "x_bin_values": np.array([], dtype=int),
            "y_bin_values": np.array([], dtype=int),
            "x_centers_um": np.array([], dtype=float),
            "y_centers_um": np.array([], dtype=float),
            "feature_indices": np.array([], dtype=int),
            "x_indices": np.array([], dtype=int),
            "y_indices": np.array([], dtype=int),
            "counts": np.array([], dtype=np.uint32),
            "shape": (len(feature_names), 0, 0),
            "occupied_x_bins": np.array([], dtype=int),
            "occupied_y_bins": np.array([], dtype=int),
            "bin_rows": np.array([], dtype=int),
        }

    work = df.loc[:, ["x_location", "y_location", "feature_name"]].copy()
    work["x_bin"] = np.floor(work["x_location"] / bin_size).astype(int)
    work["y_bin"] = np.floor(work["y_location"] / bin_size).astype(int)

    grouped = work.groupby(["feature_name", "y_bin", "x_bin"]).size().reset_index(name="count")

    feature_index = {gene: i for i, gene in enumerate(feature_names)}
    grouped = grouped[grouped["feature_name"].isin(feature_index)].copy()

    if grouped.empty:
        return {
            "feature_names": np.asarray(feature_names, dtype=object),
            "x_bin_values": np.array([], dtype=int),
            "y_bin_values": np.array([], dtype=int),
            "x_centers_um": np.array([], dtype=float),
            "y_centers_um": np.array([], dtype=float),
            "feature_indices": np.array([], dtype=int),
            "x_indices": np.array([], dtype=int),
            "y_indices": np.array([], dtype=int),
            "counts": np.array([], dtype=np.uint32),
            "shape": (len(feature_names), 0, 0),
            "occupied_x_bins": np.array([], dtype=int),
            "occupied_y_bins": np.array([], dtype=int),
            "bin_rows": np.array([], dtype=int),
        }

    x_min_bin = int(grouped["x_bin"].min())
    x_max_bin = int(grouped["x_bin"].max())
    y_min_bin = int(grouped["y_bin"].min())
    y_max_bin = int(grouped["y_bin"].max())

    x_bin_values = np.arange(x_min_bin, x_max_bin + 1, dtype=int)
    y_bin_values = np.arange(y_min_bin, y_max_bin + 1, dtype=int)
    x_centers_um = (x_bin_values + 0.5) * float(bin_size)
    y_centers_um = (y_bin_values + 0.5) * float(bin_size)

    grouped["feature_index"] = grouped["feature_name"].map(feature_index).astype(int)
    grouped["x_index"] = grouped["x_bin"] - x_min_bin
    grouped["y_index"] = grouped["y_bin"] - y_min_bin

    occupied_bins = (
        grouped.loc[:, ["x_bin", "y_bin"]].drop_duplicates().sort_values(["y_bin", "x_bin"]).reset_index(drop=True)
    )
    occupied_keys = list(zip(occupied_bins["x_bin"], occupied_bins["y_bin"]))
    occupied_lookup = {key: i for i, key in enumerate(occupied_keys)}
    grouped["bin_row"] = [
        occupied_lookup[(x_bin, y_bin)]
        for x_bin, y_bin in zip(grouped["x_bin"], grouped["y_bin"])
    ]

    return {
        "feature_names": np.asarray(feature_names, dtype=object),
        "x_bin_values": x_bin_values,
        "y_bin_values": y_bin_values,
        "x_centers_um": x_centers_um,
        "y_centers_um": y_centers_um,
        "feature_indices": grouped["feature_index"].to_numpy(dtype=int),
        "x_indices": grouped["x_index"].to_numpy(dtype=int),
        "y_indices": grouped["y_index"].to_numpy(dtype=int),
        "counts": grouped["count"].to_numpy(dtype=np.uint32),
        "shape": (len(feature_names), len(y_bin_values), len(x_bin_values)),
        "occupied_x_bins": occupied_bins["x_bin"].to_numpy(dtype=int),
        "occupied_y_bins": occupied_bins["y_bin"].to_numpy(dtype=int),
        "bin_rows": grouped["bin_row"].to_numpy(dtype=int),
    }


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
        self._load_tile_metadata(verbose)

    def _load_tile_metadata(self, verbose=True):
        """
        Build a lightweight spatial index from tile metadata and one sampled point per tile.
        """
        import zipfile
        import zarr

        tile_keys = []
        with zipfile.ZipFile(self._path) as zf:
            names = set(zf.namelist())
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
            f"{g:,} genes | cache_threshold={self.cache_threshold:,} | "
            f"{cached} cached queries)"
        )
