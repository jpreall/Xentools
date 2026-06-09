"""Shared metadata helpers for xentools."""

from __future__ import annotations

import numpy as np

__all__ = ["_annotate_adata_count_metrics", "_encode_xenium_cell_ids"]


def _encode_xenium_cell_ids(prefix_arr, suffix_arr):
    """Convert raw cell_id columns to Xenium hash strings (e.g. 'aaaabbbb-12345')."""
    prefix_arr = prefix_arr.astype(np.uint32)
    suffix_arr = suffix_arr.astype(np.uint32)
    hex_strs = np.char.zfill(np.vectorize(lambda x: format(x, "x"))(prefix_arr), 8)
    trans = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    shifted = np.char.translate(hex_strs, trans)
    return np.char.add(np.char.add(shifted, "-"), suffix_arr.astype(str))


def _axis_sum_1d(matrix, axis):
    values = matrix.sum(axis=axis)
    if hasattr(values, "A1"):
        return values.A1
    return np.asarray(values).ravel()


def _gene_feature_mask(var):
    names = var.index.astype(str)
    if "feature_types" in var.columns:
        feature_types = var["feature_types"].astype(str).str.lower()
        mask = feature_types.isin(["gene", "gene expression"]).to_numpy()
        if mask.any():
            return mask

    non_gene = names.str.contains(
        "codeword|control|deprecated|unassigned|total transcripts",
        case=False,
        regex=True,
    )
    return ~non_gene.to_numpy()


def _annotate_adata_count_metrics(adata, counts=None):
    """
    Add raw count metrics to AnnData without densifying the expression matrix.

    ``n_transcripts`` is the per-cell sum over true gene features only.
    ``total_counts`` is the per-feature sum across cells.
    ``n_cells`` is the number of cells with at least one count for each feature.
    """
    if counts is None:
        counts = adata.X

    gene_mask = _gene_feature_mask(adata.var)
    if gene_mask.any():
        gene_counts = counts[:, gene_mask]
        n_transcripts = _axis_sum_1d(gene_counts, axis=1)
    else:
        n_transcripts = np.zeros(adata.n_obs, dtype=np.int64)

    adata.obs["n_transcripts"] = np.asarray(n_transcripts).ravel().astype(np.int64)
    adata.var["total_counts"] = _axis_sum_1d(counts, axis=0).astype(np.int64)
    adata.var["n_cells"] = _axis_sum_1d(counts > 0, axis=0).astype(np.int64)
    return adata
