"""Normalization helpers for xentools."""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_tp10k"]


def normalize_tp10k(counts, log1p: bool = False):
    """
    Convert a count matrix to transcripts per 10,000 and return CSR.

    Parameters
    ----------
    counts
        Dense or sparse count matrix with observations/cells in rows.
    log1p
        If True, apply ``log1p`` after TP10K scaling. This is useful for tools
        such as CellTypist that expect log-normalized expression.
    """
    from scipy import sparse

    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)

    libsize = np.asarray(counts.sum(axis=1)).flatten()
    libsize[libsize == 0] = 1
    scaled = counts.multiply(1e4 / libsize[:, None])
    if log1p:
        scaled = scaled.log1p()
    return sparse.csr_matrix(scaled)
