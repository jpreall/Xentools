"""Zarr-format readers for xentools."""

from __future__ import annotations

import json
import os
import sys
import importlib.util

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
    from ...utils.metadata import _encode_xenium_cell_ids
except ImportError:
    _encode_xenium_cell_ids = _load_local_module(
        "_xentools_utils_metadata",
        os.path.join("..", "..", "utils", "metadata.py"),
    )._encode_xenium_cell_ids

__all__ = [
    "_open_zarr_group_compat",
    "_read_zarr_adata",
    "_read_analysis_zarr",
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


def _zarr_keys(group, key_method):
    """Return group keys across zarr versions, where key methods may be generators."""
    return set(getattr(group, key_method)())


def _read_cell_feature_matrix_arrays(cf, attrs):
    """
    Read Xenium cell-feature matrix arrays across observed zarr layouts.

    Newer bundles store a cell-major CSC representation at ``cell_features/csc``.
    Older bundles store a feature-major CSC representation directly under
    ``cell_features`` as ``data``, ``indices``, and ``indptr``.
    """
    from scipy import sparse

    cfm_cell_ids_raw = cf["cell_id"][:]
    array_keys = _zarr_keys(cf, "array_keys")
    group_keys = _zarr_keys(cf, "group_keys")

    if "csc" in group_keys:
        csc_data = cf["csc/data"][:]
        csc_indices = cf["csc/indices"][:].astype(np.int32)
        csc_indptr = cf["csc/indptr"][:]

        if "indptr" in array_keys:
            n_features = int(cf["indptr"].shape[0]) - 1
        else:
            n_features = int(attrs.get("number_features", len(attrs.get("feature_keys", []))))
        n_cells = int(csc_indptr.shape[0]) - 1

        X = sparse.csc_matrix(
            (csc_data, csc_indices, csc_indptr),
            shape=(n_features, n_cells),
        ).T.tocsr()
        return X, cfm_cell_ids_raw, n_cells, n_features

    if {"data", "indices", "indptr"}.issubset(array_keys):
        data = cf["data"][:]
        indices = cf["indices"][:].astype(np.int32)
        indptr = cf["indptr"][:]

        n_features = int(attrs.get("number_features", len(indptr) - 1))
        n_cells = int(attrs.get("number_cells", cfm_cell_ids_raw.shape[0]))

        X = sparse.csc_matrix(
            (data, indices, indptr),
            shape=(n_cells, n_features),
        ).tocsr()
        return X, cfm_cell_ids_raw, n_cells, n_features

    raise KeyError(
        "Unsupported cell_feature_matrix.zarr.zip layout: expected either "
        "cell_features/csc/{data,indices,indptr} or flat "
        "cell_features/{data,indices,indptr}."
    )


def _read_zarr_adata(folder, verbose=True, include_non_gene_features=False):
    """
    Build an AnnData object from the zarr-format cell feature matrix and cell summaries.

    By default, only true gene features are retained in ``adata.X``/``adata.var``.
    Xenium bundles may also include control/deprecated/unassigned codewords and
    aggregate features such as "Total transcripts"; those are excluded from the
    expression matrix unless ``include_non_gene_features=True``. "Total
    transcripts" is always removed from ``adata.var`` and stored as
    ``adata.obs["total_transcripts"]`` when present.
    """
    import anndata as ad
    import zarr

    cfm_path = os.path.join(folder, "cell_feature_matrix.zarr.zip")
    cells_path = os.path.join(folder, "cells.zarr.zip")

    if verbose:
        print("  Loading expression matrix from cell_feature_matrix.zarr.zip...", end=" ", flush=True)
    store_cfm = zarr.storage.ZipStore(cfm_path, mode="r")
    try:
        grp_cfm = _open_zarr_group_compat(zarr, store_cfm, mode="r", force_v2=True)
        cf = grp_cfm["cell_features"]
        attrs = dict(cf.attrs)
        X, cfm_cell_ids_raw, n_cells, n_features = _read_cell_feature_matrix_arrays(cf, attrs)
        cfm_cell_ids = cfm_cell_ids_raw[:, 0]
    finally:
        store_cfm.close()

    if verbose:
        print(f"done. ({n_cells:,} cells × {n_features:,} features)")

    if not attrs:
        import zipfile

        with zipfile.ZipFile(cfm_path) as zf:
            attrs = json.loads(zf.read("cell_features/.zattrs").decode())
    var = pd.DataFrame(
        {
            "gene_ids": attrs["feature_ids"],
            "feature_types": attrs["feature_types"],
        },
        index=pd.Index(attrs["feature_keys"], name=""),
    )
    feature_types = var["feature_types"].astype(str).str.lower()
    feature_names = var.index.astype(str)
    total_mask = feature_names == "Total transcripts"
    if include_non_gene_features:
        keep_features = ~total_mask
    else:
        keep_features = (feature_types == "gene") & ~total_mask

    if verbose:
        print("  Loading cell summaries from cells.zarr.zip...", end=" ", flush=True)
    store_cells = zarr.storage.ZipStore(cells_path, mode="r")
    try:
        grp_cells = _open_zarr_group_compat(zarr, store_cells, mode="r", force_v2=True)
        cell_summary = grp_cells["cell_summary"][:]
        cells_ids = grp_cells["cell_id"][:, 0]
    finally:
        store_cells.close()

    if verbose:
        print("done.")

    id_to_row = {int(cid): i for i, cid in enumerate(cells_ids)}
    summary_order = np.array([id_to_row.get(int(cid), -1) for cid in cfm_cell_ids])
    missing = (summary_order == -1).sum()
    if missing:
        summary_aligned = np.where(
            (summary_order[:, None] >= 0),
            cell_summary[np.clip(summary_order, 0, len(cell_summary) - 1)],
            np.nan,
        )
    else:
        summary_aligned = cell_summary[summary_order]

    obs_index = pd.Index(
        _encode_xenium_cell_ids(cfm_cell_ids_raw[:, 0], cfm_cell_ids_raw[:, 1]),
        name="cell_id",
    )
    obs = pd.DataFrame(
        {
            "x_centroid": summary_aligned[:, 0],
            "y_centroid": summary_aligned[:, 1],
            "cell_area": summary_aligned[:, 2],
            "nucleus_centroid_x": summary_aligned[:, 3],
            "nucleus_centroid_y": summary_aligned[:, 4],
            "nucleus_area": summary_aligned[:, 5],
        },
        index=obs_index,
    )

    if total_mask.any():
        total_idx = np.flatnonzero(total_mask)[0]
        obs["total_transcripts"] = np.asarray(X[:, total_idx].toarray()).ravel().astype(np.int64)

    if not keep_features.all():
        X = X[:, np.asarray(keep_features)].tocsr()
        var = var.loc[keep_features].copy()
        if verbose:
            removed = int((~keep_features).sum())
            kept = int(keep_features.sum())
            print(f"  Filtered cell feature matrix to {kept:,} gene feature(s); removed {removed:,} non-gene/aggregate feature(s).")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = summary_aligned[:, :2].astype(np.float32)

    return adata


def _read_analysis_zarr(folder, verbose=True):
    """
    Build a clusters DataFrame from analysis.zarr.zip.
    """
    import zarr

    analysis_path = os.path.join(folder, "analysis.zarr.zip")
    cells_path = os.path.join(folder, "cells.zarr.zip")

    if not os.path.exists(analysis_path):
        return pd.DataFrame()

    if verbose:
        print("  Loading cluster assignments from analysis.zarr.zip...", end=" ", flush=True)

    store_cells = zarr.storage.ZipStore(cells_path, mode="r")
    try:
        grp_cells = _open_zarr_group_compat(zarr, store_cells, mode="r", force_v2=True)
        cell_ids_raw = grp_cells["cell_id"][:]
        cell_ids = cell_ids_raw[:, 0].astype(np.int64)
    finally:
        store_cells.close()

    store_an = zarr.storage.ZipStore(analysis_path, mode="r")
    try:
        grp_an = _open_zarr_group_compat(zarr, store_an, mode="r", force_v2=True)
        attrs = dict(grp_an["cell_groups"].attrs)
        grouping_names = attrs["grouping_names"]
        group_names = attrs["group_names"]

        columns = {}
        for k, (grp_name, clust_labels) in enumerate(zip(grouping_names, group_names)):
            indptr = grp_an[f"cell_groups/{k}/indptr"][:]
            indices = grp_an[f"cell_groups/{k}/indices"][:]
            if len(indptr) == len(clust_labels):
                # Some Xenium Prime analysis bundles store group start offsets
                # without the terminal pointer; the final group runs to the end.
                indptr = np.append(indptr, len(indices))
            elif len(indptr) != len(clust_labels) + 1:
                raise ValueError(
                    f"Unexpected analysis.zarr.zip cell_groups/{k} layout: "
                    f"{len(clust_labels)} labels but indptr has length {len(indptr)}"
                )

            assignment = np.empty(len(cell_ids), dtype=object)
            assignment[:] = np.nan
            for ci, label in enumerate(clust_labels):
                pos = indices[indptr[ci] : indptr[ci + 1]]
                assignment[pos] = label

            if "graphclust" in grp_name:
                col_name = "Cluster"
            else:
                col_name = grp_name.replace("gene_expression_", "")

            columns[col_name] = assignment
    finally:
        store_an.close()

    df = pd.DataFrame(
        columns,
        index=pd.Index(_encode_xenium_cell_ids(cell_ids_raw[:, 0], cell_ids_raw[:, 1]), name="cell_id"),
    )
    if "Cluster" in df.columns:
        df["Cluster"] = df["Cluster"].astype("str").astype("category")

    if verbose:
        n_clust = df["Cluster"].nunique() if "Cluster" in df.columns else 0
        print(f"done. ({n_clust} graph clusters, {len(df):,} cells, {len(df.columns)} groupings)")

    return df
