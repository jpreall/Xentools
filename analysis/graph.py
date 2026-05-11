"""Spatial graph construction helpers."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import sparse

__all__ = ["_apply_weights", "build_spatial_graph"]


def _apply_weights(
    d,
    scheme: Optional[Literal["binary", "inverse", "gaussian"]],
    sigma: Optional[float],
):
    if scheme in (None, "binary"):
        return np.ones_like(d, dtype=float)
    if scheme == "inverse":
        return 1.0 / np.maximum(d, 1e-12)
    if scheme == "gaussian":
        if not sigma:
            raise ValueError("sigma must be provided for gaussian weighting")
        return np.exp(-(d**2) / (2.0 * sigma**2))
    raise ValueError(f"Unknown weight scheme: {scheme}")


def build_spatial_graph(
    coords: np.ndarray,
    *,
    use_radius: Optional[float] = None,
    n_neighbors: int = 30,
    include_self: bool = False,
    symmetrize: Literal["none", "max", "mean"] = "max",
    weight_scheme: Optional[Literal["binary", "inverse", "gaussian"]] = "binary",
    sigma: Optional[float] = None,
) -> sparse.csr_matrix:
    """
    Return a CSR spatial adjacency matrix using ``scipy.spatial.cKDTree``.
    """
    from scipy.spatial import cKDTree

    n_obs = coords.shape[0]
    tree = cKDTree(coords)

    if use_radius is not None:
        distances = tree.sparse_distance_matrix(
            tree,
            max_distance=use_radius,
            output_type="coo_matrix",
        )
        if not include_self:
            mask = distances.row != distances.col
            distances = sparse.coo_matrix(
                (distances.data[mask], (distances.row[mask], distances.col[mask])),
                shape=(n_obs, n_obs),
            )
        weights = _apply_weights(distances.data, weight_scheme, sigma)
        graph = sparse.coo_matrix(
            (weights, (distances.row, distances.col)),
            shape=(n_obs, n_obs),
        ).tocsr()
    else:
        k = n_neighbors + (1 if include_self else 0)
        d, idx = tree.query(coords, k=k)
        rows = np.repeat(np.arange(n_obs), k)
        cols = idx.ravel()
        if not include_self:
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]
            d = d.ravel()[mask]
        else:
            d = d.ravel()
        weights = _apply_weights(d, weight_scheme, sigma)
        graph = sparse.coo_matrix((weights, (rows, cols)), shape=(n_obs, n_obs)).tocsr()

    if symmetrize == "max":
        graph = graph.maximum(graph.T)
    elif symmetrize == "mean":
        graph = 0.5 * (graph + graph.T)
        graph.eliminate_zeros()

    return graph
