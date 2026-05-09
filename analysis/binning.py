"""Spatial binning helpers for transcript-level data."""

from __future__ import annotations

import numpy as np

__all__ = [
    "_bin_transcript_dataframe",
    "_normalize_feature_selection",
    "create_binned_adata",
]


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
            f"Some values in {arg_name} are not present in the dataset: "
            + ", ".join(map(str, missing))
        )
    return selected


def _is_lazy_transcripts(transcripts):
    return hasattr(transcripts, "_tile_meta") and hasattr(transcripts, "query")


def _empty_binning(feature_names):
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


def _bin_transcript_dataframe(df, bin_size, feature_names):
    """
    Aggregate transcript coordinates into a shared sparse binning spec.

    Parameters
    ----------
    df
        DataFrame with ``x_location``, ``y_location``, and ``feature_name``.
    bin_size
        Spatial bin size in microns.
    feature_names
        Ordered feature list defining output columns/channels.
    """
    feature_names = list(feature_names)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

    if df.empty:
        return _empty_binning(feature_names)

    work = df.loc[:, ["x_location", "y_location", "feature_name"]].copy()
    work["x_bin"] = np.floor(work["x_location"] / bin_size).astype(int)
    work["y_bin"] = np.floor(work["y_location"] / bin_size).astype(int)

    grouped = work.groupby(["feature_name", "y_bin", "x_bin"]).size().reset_index(name="count")

    feature_index = {gene: i for i, gene in enumerate(feature_names)}
    grouped = grouped[grouped["feature_name"].isin(feature_index)].copy()

    if grouped.empty:
        return _empty_binning(feature_names)

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
        grouped.loc[:, ["x_bin", "y_bin"]]
        .drop_duplicates()
        .sort_values(["y_bin", "x_bin"])
        .reset_index(drop=True)
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


def create_binned_adata(
    xdata,
    bin_size=5,
    exclude_unassigned=True,
    distance_to_nucleus: float = None,
    include_features: list = None,
):
    """
    Create a binned AnnData object from a XenData-like object's transcripts.

    The returned AnnData stores bins as observations and features/genes as
    variables. The count matrix is sparse CSR. Bin coordinates are stored both
    in plotting-bin units (``obs[['x_bin', 'y_bin']]`` and ``obsm['spatial']``)
    and original micron coordinates (``obsm['spatial_um']``).
    """
    import os

    import anndata as ad
    import pandas as pd
    from scipy.sparse import coo_matrix

    print(f"Creating binned AnnData object with bin size of {bin_size}um...")

    include_features = _normalize_feature_selection(
        include_features,
        xdata.features,
        arg_name="include_features",
    )

    if _is_lazy_transcripts(xdata.trans):
        bounds = None if xdata.subset_roi is None else xdata.subset_roi.bounds
        parquet_path = os.path.join(xdata.xenium_folder, "transcripts.parquet")
        need_metadata = exclude_unassigned or (distance_to_nucleus is not None)

        use_parquet = False
        parquet_columns = ["x_location", "y_location", "feature_name"]
        if exclude_unassigned:
            parquet_columns.append("cell_id")
        if distance_to_nucleus is not None:
            parquet_columns.append("nucleus_distance")

        if need_metadata and os.path.exists(parquet_path):
            try:
                import pyarrow.parquet as pq

                schema_names = set(pq.ParquetFile(parquet_path).schema.names)
                use_parquet = set(parquet_columns).issubset(schema_names)
            except Exception:
                use_parquet = False

        if use_parquet:
            print("Using transcripts.parquet to preserve transcript metadata filters...")
            df = pd.read_parquet(parquet_path, columns=parquet_columns)
            if bounds is not None:
                df = df[
                    (df["x_location"] >= bounds[0])
                    & (df["x_location"] <= bounds[1])
                    & (df["y_location"] >= bounds[2])
                    & (df["y_location"] <= bounds[3])
                ].copy()
        else:
            if need_metadata:
                print(
                    "Transcript metadata are unavailable in the lazy Zarr view; "
                    "proceeding without them."
                )
            query_kwargs = {}
            if bounds is not None:
                query_kwargs.update(
                    xmin=bounds[0],
                    xmax=bounds[1],
                    ymin=bounds[2],
                    ymax=bounds[3],
                )
            df = xdata.trans.query(
                genes=include_features,
                quality="all",
                **query_kwargs,
            )
    else:
        df = xdata.trans.copy()

    df = df[df["feature_name"].isin(include_features)].copy()

    if exclude_unassigned:
        if "cell_id" in df.columns:
            print("Using only transcripts assigned to cells/nuclei...")
            df = df[df["cell_id"] != "UNASSIGNED"].copy()
        else:
            print("Transcript cell assignments are unavailable; skipping exclude_unassigned filter.")

    if distance_to_nucleus is not None:
        if "nucleus_distance" in df.columns:
            print(f"Excluding transcripts further than {distance_to_nucleus}um from the nucleus...")
            df = df[df["nucleus_distance"] <= distance_to_nucleus].copy()
        else:
            print("Transcript nucleus distances are unavailable; skipping distance_to_nucleus filter.")

    binned = _bin_transcript_dataframe(
        df,
        bin_size=bin_size,
        feature_names=include_features,
    )

    expression_matrix = coo_matrix(
        (binned["counts"], (binned["bin_rows"], binned["feature_indices"])),
        shape=(len(binned["occupied_x_bins"]), len(binned["feature_names"])),
    )

    binned_adata = ad.AnnData(X=expression_matrix.tocsr())
    binned_adata.obs_names = [f"bin_{i}" for i in range(binned_adata.n_obs)]
    binned_adata.var_names = list(binned["feature_names"])

    if binned_adata.n_obs:
        x_bins = binned["occupied_x_bins"].astype(int)
        y_bins = binned["occupied_y_bins"].astype(int)
        ymid = (y_bins.max() - y_bins.min()) / 2.0
        y_bins_plot = -(y_bins - ymid).astype(int)
        bin_coords = np.column_stack([x_bins, y_bins_plot])
        spatial_um = np.column_stack(
            [
                (x_bins + 0.5) * float(bin_size),
                (y_bins + 0.5) * float(bin_size),
            ]
        )
    else:
        bin_coords = np.zeros((0, 2), dtype=int)
        spatial_um = np.zeros((0, 2), dtype=float)

    binned_adata.obs[["x_bin", "y_bin"]] = bin_coords
    binned_adata.obsm["spatial"] = bin_coords
    binned_adata.obsm["spatial_um"] = spatial_um
    binned_adata.uns["bin_size"] = float(bin_size)
    binned_adata.uns["pixel_size"] = xdata.pixel_size
    binned_adata.uns["x_bin_edges"] = (
        np.array([], dtype=float)
        if not len(binned["x_bin_values"])
        else np.arange(
            binned["x_bin_values"][0] * bin_size,
            (binned["x_bin_values"][-1] + 1) * bin_size + bin_size,
            bin_size,
            dtype=float,
        )
    )
    binned_adata.uns["y_bin_edges"] = (
        np.array([], dtype=float)
        if not len(binned["y_bin_values"])
        else np.arange(
            binned["y_bin_values"][0] * bin_size,
            (binned["y_bin_values"][-1] + 1) * bin_size + bin_size,
            bin_size,
            dtype=float,
        )
    )

    return binned_adata
