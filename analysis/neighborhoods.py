"""ROI-aware cell neighborhood composition helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse


def _load_local_module(module_name, relative_path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .graph import build_spatial_graph
except ImportError:
    build_spatial_graph = _load_local_module(
        "_xentools_analysis_graph_for_neighborhoods",
        "analysis/graph.py",
    ).build_spatial_graph

__all__ = ["neighborhood_composition"]


def _as_adata(data):
    return data.adata if hasattr(data, "adata") else data


def _resolve_roi(data, roi):
    if roi == "active":
        return getattr(data, "active_roi", None)
    if roi is None:
        return None
    if isinstance(roi, str) and hasattr(data, "rois"):
        return data.rois.resolve(roi)
    if hasattr(roi, "contains_points"):
        return roi
    raise ValueError("roi must be 'active', None, an ROI name, or an ROI-like object with contains_points().")


def _coordinates(adata, xy_key):
    if xy_key is not None and xy_key in adata.obsm:
        coords = np.asarray(adata.obsm[xy_key], dtype=float)
        if coords.shape[1] > 2:
            coords = coords[:, :2]
        return coords
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        return adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].to_numpy(dtype=float)
    raise KeyError(f"Could not find spatial coordinates in adata.obsm['{xy_key}'] or obs centroid columns.")


def _label_categories(labels):
    if isinstance(labels.dtype, CategoricalDtype):
        return list(labels.cat.categories)
    return sorted(pd.Series(labels).dropna().astype(str).unique().tolist())


def _one_hot(labels, categories):
    cat = pd.Categorical(labels.astype(str), categories=[str(c) for c in categories])
    codes = cat.codes
    valid = codes >= 0
    rows = np.flatnonzero(valid)
    cols = codes[valid]
    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(labels), len(categories)))


def neighborhood_composition(
    data,
    *,
    label_key: str = "Cluster",
    roi="active",
    xy_key: Optional[str] = "spatial",
    use_radius: Optional[float] = None,
    n_neighbors: int = 30,
    include_self: bool = False,
    symmetrize: Literal["none", "max", "mean"] = "max",
    weight_scheme: Optional[Literal["binary", "inverse", "gaussian"]] = "binary",
    sigma: Optional[float] = None,
    normalize: Literal["count", "prop"] = "prop",
    include_neighbor_total: bool = True,
) -> pd.DataFrame:
    """
    Compute cell-neighborhood label composition, defaulting to the active ROI.

    Parameters
    ----------
    data
        A ``XenData`` object or an AnnData object. Passing ``XenData`` enables
        ROI-aware behavior.
    label_key
        Column in ``adata.obs`` containing cell labels such as clusters or cell
        types.
    roi
        ``"active"`` uses ``xdata.active_roi`` when present. ``None`` uses all
        cells. A string resolves a named ROI from ``xdata.rois``. ROI-like
        objects with ``contains_points`` are also accepted.
    normalize
        ``"count"`` returns raw neighbor counts or weighted sums. ``"prop"``
        row-normalizes each cell's neighborhood vector to sum to 1.

    Returns
    -------
    pandas.DataFrame
        Per-cell neighborhood composition. Rows are cells inside the selected
        ROI, columns are categories from ``label_key``. If
        ``include_neighbor_total=True``, ``"_neighbor_total"`` stores the row
        sum of adjacency weights before composition normalization.
    """
    adata = _as_adata(data)
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs['{label_key}'] not found.")
    if normalize not in {"count", "prop"}:
        raise ValueError("normalize must be 'count' or 'prop'.")

    coords = _coordinates(adata, xy_key)
    selected_roi = _resolve_roi(data, roi)
    if selected_roi is None:
        mask = np.ones(adata.n_obs, dtype=bool)
    else:
        mask = np.asarray(selected_roi.contains_points(coords[:, 0], coords[:, 1]), dtype=bool)

    categories = _label_categories(adata.obs[label_key])
    columns = [str(c) for c in categories]
    if include_neighbor_total:
        columns = columns + ["_neighbor_total"]

    if not mask.any():
        out = pd.DataFrame(columns=columns, index=adata.obs_names[mask])
        out.attrs["roi"] = getattr(selected_roi, "name", None) if selected_roi is not None else None
        out.attrs["label_key"] = label_key
        return out

    coords_sub = coords[mask]
    labels_sub = adata.obs.loc[mask, label_key]
    index = adata.obs_names[mask]
    n_obs = coords_sub.shape[0]

    if n_obs <= 1:
        composition = np.zeros((n_obs, len(categories)), dtype=float)
        neighbor_total = np.zeros(n_obs, dtype=float)
    else:
        effective_neighbors = min(int(n_neighbors), n_obs - 1)
        graph = build_spatial_graph(
            coords_sub,
            use_radius=use_radius,
            n_neighbors=effective_neighbors,
            include_self=include_self,
            symmetrize=symmetrize,
            weight_scheme=weight_scheme,
            sigma=sigma,
        )
        onehot = _one_hot(labels_sub, categories)
        raw = (graph @ onehot).astype(float)
        neighbor_total = np.asarray(graph.sum(axis=1)).ravel()
        if normalize == "prop":
            row_sums = np.asarray(raw.sum(axis=1)).ravel()
            safe = row_sums.copy()
            safe[safe == 0] = 1.0
            raw = sparse.diags(1.0 / safe) @ raw
        composition = raw.toarray()

    out = pd.DataFrame(composition, index=index, columns=[str(c) for c in categories])
    if include_neighbor_total:
        out["_neighbor_total"] = neighbor_total
    out.attrs["roi"] = getattr(selected_roi, "name", None) if selected_roi is not None else None
    out.attrs["label_key"] = label_key
    out.attrs["normalize"] = normalize
    return out
