"""Lightweight discovery of stable, well-expressed genes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = ["find_housekeeping_genes"]


def _as_adata(data):
    return data.adata if hasattr(data, "adata") else data


def _coordinates(adata, xy_key):
    if xy_key in adata.obsm:
        return np.asarray(adata.obsm[xy_key], dtype=float)[:, :2]
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        return adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].to_numpy(dtype=float)
    raise KeyError(
        f"Could not find coordinates in adata.obsm['{xy_key}'] or obs centroid columns."
    )


def _resolve_rois(data, rois):
    if rois is None:
        if not hasattr(data, "rois"):
            raise TypeError("Named ROIs require XenData; use ROI objects or groupby for AnnData.")
        resolved = data.rois.all_rois()
    else:
        if isinstance(rois, (str, int)) or hasattr(rois, "contains_points"):
            rois = [rois]
        resolved = []
        for roi in rois:
            if isinstance(roi, (str, int)):
                if not hasattr(data, "rois"):
                    raise TypeError("Named ROIs require a XenData object.")
                roi = data.rois.resolve(roi)
            resolved.append(roi)
    if not resolved:
        raise ValueError("No ROIs were supplied and XenData.rois is empty.")
    if any(not hasattr(roi, "contains_points") for roi in resolved):
        raise TypeError("Each ROI must provide contains_points(x, y).")
    return resolved


def _group_masks(data, adata, *, rois, groupby, xy_key, min_cells):
    if rois is not None and groupby is not None:
        raise ValueError("Pass either rois or groupby, not both.")

    groups = []
    if groupby is not None:
        if groupby not in adata.obs:
            raise KeyError(f"adata.obs['{groupby}'] not found.")
        labels = adata.obs[groupby]
        for label in pd.unique(labels.dropna()):
            groups.append((str(label), np.asarray(labels == label)))
    elif rois is not None or hasattr(data, "rois"):
        coords = _coordinates(adata, xy_key)
        for i, roi in enumerate(_resolve_rois(data, rois)):
            name = str(getattr(roi, "name", None) or f"roi_{i}")
            mask = np.asarray(roi.contains_points(coords[:, 0], coords[:, 1]), dtype=bool)
            groups.append((name, mask))
    else:
        groups.append(("all", np.ones(adata.n_obs, dtype=bool)))

    small = [(name, int(mask.sum())) for name, mask in groups if mask.sum() < min_cells]
    if small:
        details = ", ".join(f"{name} ({n})" for name, n in small)
        raise ValueError(f"Groups with fewer than min_cells={min_cells}: {details}.")
    return groups


def _rank01(values, *, ascending=True):
    return pd.Series(values).rank(method="average", pct=True, ascending=ascending).to_numpy()


def _raw_count_issue(matrix):
    values = sparse.csr_matrix(matrix, dtype=float).data
    if values.size == 0:
        return None
    if not np.isfinite(values).all():
        return "contains non-finite values"
    if np.min(values) < 0:
        return "contains negative values"
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        return "contains non-integer values and appears normalized or transformed"
    return None


def _resolve_raw_counts(adata, layer):
    """Resolve raw counts without silently treating transformed data as counts."""
    if layer not in (None, "auto"):
        if layer not in adata.layers:
            available = list(adata.layers.keys())
            raise KeyError(f"adata.layers[{layer!r}] not found. Available layers: {available}.")
        matrix = adata.layers[layer]
        issue = _raw_count_issue(matrix)
        if issue:
            raise ValueError(f"adata.layers[{layer!r}] {issue}; a raw-count layer is required.")
        return sparse.csr_matrix(matrix, dtype=float), f"adata.layers[{layer!r}]"

    x_issue = _raw_count_issue(adata.X)
    if x_issue is None:
        return sparse.csr_matrix(adata.X, dtype=float), "adata.X"
    if layer is None:
        raise ValueError(_raw_count_guidance(adata, f"adata.X {x_issue}"))

    preferred_names = ["counts", "raw_counts", "raw"]
    layer_names = list(adata.layers.keys())
    ordered_names = [name for name in preferred_names if name in adata.layers]
    ordered_names.extend(name for name in layer_names if name not in ordered_names)
    valid_layers = [name for name in ordered_names if _raw_count_issue(adata.layers[name]) is None]
    if valid_layers:
        selected = valid_layers[0]
        return sparse.csr_matrix(adata.layers[selected], dtype=float), f"adata.layers[{selected!r}]"

    if adata.raw is not None:
        raw_names = pd.Index(adata.raw.var_names.astype(str))
        current_names = pd.Index(adata.var_names.astype(str))
        if current_names.isin(raw_names).all():
            raw_matrix = adata.raw[:, current_names].X
            if _raw_count_issue(raw_matrix) is None:
                return sparse.csr_matrix(raw_matrix, dtype=float), "adata.raw.X"

    raise ValueError(_raw_count_guidance(adata, f"adata.X {x_issue}"))


def _raw_count_guidance(adata, reason):
    layer_details = []
    for name in adata.layers.keys():
        issue = _raw_count_issue(adata.layers[name])
        layer_details.append(f"{name!r} ({'raw-count compatible' if issue is None else issue})")
    layer_text = ", ".join(layer_details) if layer_details else "none"
    raw_text = "present" if adata.raw is not None else "not present"
    return (
        f"Housekeeping analysis requires raw counts, but {reason}. "
        f"Available layers: {layer_text}; adata.raw is {raw_text}. "
        "If raw counts are stored in a layer, rerun with layer='<layer_name>' or "
        "layer='auto'. Otherwise reload XenData to restore the original count matrix, "
        "and before normalization preserve it with "
        "xdata.adata.layers['counts'] = xdata.adata.X.copy()."
    )


def find_housekeeping_genes(
    data,
    *,
    rois=None,
    groupby: str | None = None,
    n_genes: int = 20,
    layer: str | None = "auto",
    xy_key: str = "spatial",
    min_cells: int = 20,
    min_detection: float = 0.25,
    min_expression_quantile: float = 0.5,
    exclude_genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Find well-expressed genes with stable expression across tissues or ROIs.

    The method uses raw counts without library-size normalization. It filters
    genes by raw mean expression and worst-group detection, measures
    within-group stability with Poisson-adjusted Pearson dispersion, and
    measures between-group consistency from log-transformed group means. A
    percentile score combines expression, detection, and both stability terms.

    Parameters
    ----------
    data
        :class:`~xentools.XenData` or AnnData with cells in rows and genes in
        columns.
    rois
        ROI names, indices, or ROI objects. With XenData, ``None`` uses every
        registered ROI unless ``groupby`` is supplied.
    groupby
        Alternative to ``rois``: an ``adata.obs`` column defining tissues,
        samples, cores, or other comparison groups.
    n_genes
        Number of genes marked in the ``selected`` column.
    layer
        Raw-count source. ``"auto"`` uses ``adata.X`` when it contains counts,
        then searches raw-compatible layers and ``adata.raw``. Pass a layer
        name explicitly, or ``None`` to require raw counts in ``adata.X``.
    min_detection
        Minimum fraction of cells with nonzero counts required in every group.
    min_expression_quantile
        Exclude genes below this quantile of global mean raw expression.

    Returns
    -------
    pandas.DataFrame
        Ranked genes with selection status, score components, and per-group
        expression and detection statistics.
    """
    if n_genes < 1:
        raise ValueError("n_genes must be at least 1.")
    if not 0 <= min_detection <= 1:
        raise ValueError("min_detection must be between 0 and 1.")
    if not 0 <= min_expression_quantile <= 1:
        raise ValueError("min_expression_quantile must be between 0 and 1.")
    adata = _as_adata(data)
    counts, count_source = _resolve_raw_counts(adata, layer)
    if counts.shape != (adata.n_obs, adata.n_vars):
        raise ValueError("Expression matrix shape does not match adata dimensions.")

    groups = _group_masks(
        data, adata, rois=rois, groupby=groupby, xy_key=xy_key, min_cells=min_cells
    )
    group_mean = []
    group_detection = []
    group_dispersion = []
    for _, mask in groups:
        values = counts[mask]
        n_group = int(mask.sum())
        mean = np.asarray(values.mean(axis=0)).ravel()
        mean_sq = np.asarray(values.power(2).mean(axis=0)).ravel()
        group_mean.append(mean)
        # Pearson dispersion around the tissue-specific mean. This is the
        # residual variance after accounting for tissue identity, adjusted for
        # the Poisson mean-variance relationship of count data.
        residual_ss = n_group * np.maximum(mean_sq - mean**2, 0)
        dispersion = np.divide(
            residual_ss,
            (n_group - 1) * mean,
            out=np.zeros_like(mean),
            where=mean > 0,
        )
        group_dispersion.append(dispersion)
        group_detection.append(np.asarray((counts[mask] > 0).mean(axis=0)).ravel())

    means = np.vstack(group_mean)
    detections = np.vstack(group_detection)
    dispersions = np.vstack(group_dispersion)
    global_expression = np.asarray(counts.mean(axis=0)).ravel()
    within_dispersion = dispersions.mean(axis=0)
    between_log_variance = np.log1p(means).var(axis=0)
    min_detected = detections.min(axis=0)

    expression_cutoff = np.quantile(global_expression, min_expression_quantile)
    eligible = (global_expression >= expression_cutoff) & (min_detected >= min_detection)
    if exclude_genes is not None:
        eligible &= ~np.isin(np.asarray(adata.var_names, dtype=str), list(exclude_genes))

    # Percentile components avoid letting a few extreme genes dominate.
    expression_rank = _rank01(global_expression, ascending=True)
    detection_rank = _rank01(min_detected, ascending=True)
    within_stability = _rank01(within_dispersion, ascending=False)
    between_stability = _rank01(between_log_variance, ascending=False)
    stability_score = 0.60 * within_stability + 0.40 * between_stability
    score = (
        0.30 * expression_rank
        + 0.25 * detection_rank
        + 0.27 * within_stability
        + 0.18 * between_stability
    )
    score[~eligible] = np.nan

    result = pd.DataFrame(
        {
            "mean_expression": global_expression,
            "min_detection": min_detected,
            "within_group_dispersion": within_dispersion,
            "between_group_log_variance": between_log_variance,
            "stability_score": stability_score,
            "score": score,
            "eligible": eligible,
        },
        index=pd.Index(adata.var_names.astype(str), name="gene"),
    )
    for i, (name, _) in enumerate(groups):
        result[f"mean_count__{name}"] = means[i]
        result[f"dispersion__{name}"] = dispersions[i]
        result[f"detection__{name}"] = detections[i]

    result = result.sort_values(
        ["eligible", "score", "stability_score", "mean_expression"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    result["rank"] = pd.Series(
        np.arange(1, int(eligible.sum()) + 1),
        index=result.index[0 : int(eligible.sum())],
        dtype="Int64",
    )
    result["selected"] = False
    result.iloc[: min(n_genes, int(eligible.sum())), result.columns.get_loc("selected")] = True
    result.attrs.update(
        {
            "groups": [name for name, _ in groups],
            "n_genes_requested": n_genes,
            "normalization": "none (raw counts)",
            "count_source": count_source,
            "stability_method": "within-group Pearson dispersion and between-group log-mean variance",
            "expression_cutoff": float(expression_cutoff),
            "min_detection": float(min_detection),
            "min_expression_quantile": float(min_expression_quantile),
        }
    )
    return result
