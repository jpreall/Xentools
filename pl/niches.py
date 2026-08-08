"""Plotting helpers for xentools niche and neighborhood analyses."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .boundaries import plot_cells
from ._save import save_figure


__all__ = ["niche_heatmap", "niche_map"]


def _as_adata(data):
    return data.adata if hasattr(data, "adata") else data


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


def _auto_niche_label_key(adata, key_added):
    prefix = f"{key_added}_k"
    candidates = [col for col in adata.obs.columns if str(col).startswith(prefix)]
    if not candidates:
        raise KeyError(
            f"No niche label column found in adata.obs starting with '{prefix}'. "
            "Run build_niches(..., k_niches=<int>) or pass groupby/color explicitly."
        )
    return candidates[-1]


def _roi_bounds(roi):
    return None if roi is None else roi.bounds


def _full_bounds(coords):
    return (
        float(coords[:, 0].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].min()),
        float(coords[:, 1].max()),
    )


def _composition_frame(data, key_added, matrix_key=None, celltypes_key=None):
    if isinstance(data, pd.DataFrame):
        cols = [c for c in data.columns if c != "_neighbor_total"]
        return data.loc[:, cols].astype(float)

    adata = _as_adata(data)
    matrix_key = f"{key_added}_X" if matrix_key is None else matrix_key
    celltypes_key = f"{key_added}_celltypes" if celltypes_key is None else celltypes_key
    if matrix_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{matrix_key}'] not found.")
    if celltypes_key not in adata.uns:
        raise KeyError(f"adata.uns['{celltypes_key}'] not found.")

    matrix = np.asarray(adata.obsm[matrix_key], dtype=float)
    columns = [str(c) for c in adata.uns[celltypes_key]]
    return pd.DataFrame(matrix, index=adata.obs_names, columns=columns)


def niche_heatmap(
    data,
    *,
    key_added: str = "niche",
    matrix_key: Optional[str] = None,
    celltypes_key: Optional[str] = None,
    groupby: Optional[str] = None,
    cmap: str = "viridis",
    ax=None,
    figsize=None,
    show_values: bool = False,
    value_fmt: str = ".2f",
    cbar: bool = True,
    title: Optional[str] = None,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Plot a heatmap of mean neighborhood composition by niche label.

    ``data`` can be a ``XenData`` object, an AnnData object containing
    ``adata.obsm[f"{key_added}_X"]`` and ``adata.uns[f"{key_added}_celltypes"]``,
    or a DataFrame returned by ``neighborhood_composition()``. For AnnData-like
    inputs, rows are averaged by ``groupby``; when ``groupby`` is omitted,
    xentools uses the latest ``adata.obs[f"{key_added}_k..."]`` column.
    """
    import matplotlib.pyplot as plt

    comp = _composition_frame(
        data,
        key_added=key_added,
        matrix_key=matrix_key,
        celltypes_key=celltypes_key,
    )

    if isinstance(data, pd.DataFrame):
        heat = comp
        ylabel = "Cell"
    else:
        adata = _as_adata(data)
        if groupby is None:
            groupby = _auto_niche_label_key(adata, key_added)
        if groupby not in adata.obs:
            raise KeyError(f"adata.obs['{groupby}'] not found.")
        labels = adata.obs[groupby].reindex(comp.index)
        heat = comp.groupby(labels, observed=False).mean()
        ylabel = groupby

    if ax is None:
        fig_w = max(5, 0.45 * max(1, heat.shape[1]))
        fig_h = max(3, 0.35 * max(1, heat.shape[0]))
        _, ax = plt.subplots(figsize=figsize or (fig_w, fig_h))

    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index.astype(str))
    ax.set_xlabel("Neighborhood label")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Niche composition")

    if show_values:
        values = heat.to_numpy(dtype=float)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, format(values[i, j], value_fmt), ha="center", va="center", fontsize=8)

    if cbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean composition")

    ax.grid(False)
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    return ax


def niche_map(
    data,
    *,
    color: Optional[str] = None,
    key_added: str = "niche",
    k: Optional[int] = None,
    roi="active",
    xy_key: Optional[str] = "spatial",
    ax=None,
    figsize=(8, 8),
    s: float = 3,
    alpha: float = 0.8,
    cmap: str = "viridis",
    palette: Optional[dict] = None,
    use_boundaries: bool = True,
    face_alpha: float = 0.85,
    edge_alpha: float = 0.0,
    linewidth: float = 0.0,
    background: str = "black",
    show_axis: bool = False,
    show_legend: bool = True,
    legend_loc: str = "outside right",
    title: Optional[str] = None,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Plot cells spatially, colored by niche label or niche-composition value.

    For ``XenData`` inputs with cell boundaries, this delegates to
    :func:`plot_cells` and defaults to edge-free filled polygons to reduce
    boundary noise. For AnnData-only inputs or ``use_boundaries=False``, it
    falls back to a centroid scatter plot.

    If ``color`` is omitted, xentools uses ``adata.obs[f"{key_added}_k{k}"]``
    when ``k`` is supplied, otherwise the latest niche label column starting
    with ``f"{key_added}_k"``. If ``color`` names one of the entries in
    ``adata.uns[f"{key_added}_celltypes"]``, the corresponding continuous column
    from ``adata.obsm[f"{key_added}_X"]`` is plotted.
    """
    adata = _as_adata(data)
    coords = _coordinates(adata, xy_key)
    selected_roi = _resolve_roi(data, roi)

    if color is None:
        color = f"{key_added}_k{int(k)}" if k is not None else _auto_niche_label_key(adata, key_added)

    celltypes = [str(c) for c in adata.uns.get(f"{key_added}_celltypes", [])]
    temporary_obs_key = None
    if color in celltypes and color not in adata.obs:
        matrix = np.asarray(adata.obsm[f"{key_added}_X"], dtype=float)
        temporary_obs_key = f"_{key_added}_{color}_composition"
        while temporary_obs_key in adata.obs:
            temporary_obs_key = f"_{temporary_obs_key}"
        adata.obs[temporary_obs_key] = matrix[:, celltypes.index(color)]
        color_by = temporary_obs_key
        color_label = color
    elif color in adata.obs:
        color_by = color
        color_label = color
    else:
        raise KeyError(
            f"'{color}' was not found in adata.obs or adata.uns['{key_added}_celltypes']."
        )

    if use_boundaries and hasattr(data, "cell_boundaries") and getattr(data, "cell_boundaries", None) is not None:
        is_numeric = pd.api.types.is_numeric_dtype(adata.obs[color_by])
        bounds = _roi_bounds(selected_roi)
        if selected_roi is None and roi is None and coords.size:
            bounds = _full_bounds(coords)
        try:
            ax = plot_cells(
                data,
                color_by=color_by,
                cmap=cmap,
                palette=palette,
                face_alpha=face_alpha,
                edge_alpha=edge_alpha,
                linewidth=linewidth,
                bounds=bounds,
                ax=ax,
                figsize=figsize,
                show_legend=show_legend,
                legend_loc=legend_loc,
                legend_title=color_label,
                show_colorbar=is_numeric,
                colorbar_label=color_label,
                background=background,
                show_axis=show_axis,
                save=save,
                save_kwargs=save_kwargs,
            )
            if title is not None:
                ax.set_title(title)
            return ax
        finally:
            if temporary_obs_key is not None:
                del adata.obs[temporary_obs_key]

    if selected_roi is None:
        mask = np.ones(adata.n_obs, dtype=bool)
    else:
        mask = np.asarray(selected_roi.contains_points(coords[:, 0], coords[:, 1]), dtype=bool)

    values = adata.obs[color_by]
    plot_coords = coords[mask]
    plot_values = values.loc[adata.obs_names[mask]]

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    if mask.any():
        x_lo = float(plot_coords[:, 0].min())
        x_hi = float(plot_coords[:, 0].max())
        y_lo = float(plot_coords[:, 1].min())
        y_hi = float(plot_coords[:, 1].max())
        display_coords = plot_coords
    else:
        x_lo = x_hi = y_lo = y_hi = None
        display_coords = plot_coords

    if pd.api.types.is_numeric_dtype(plot_values):
        sc = ax.scatter(
            display_coords[:, 0],
            display_coords[:, 1],
            c=plot_values.to_numpy(dtype=float),
            s=s,
            alpha=alpha,
            cmap=cmap,
            linewidths=0,
        )
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)
    else:
        categories = (
            list(plot_values.cat.categories)
            if hasattr(plot_values, "cat")
            else sorted(plot_values.dropna().astype(str).unique())
        )
        if palette is None:
            colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(categories))))
            palette = {cat: mcolors.to_hex(colors[i]) for i, cat in enumerate(categories)}
        mapped = plot_values.astype(str).map(palette).fillna("gray")
        ax.scatter(
            display_coords[:, 0],
            display_coords[:, 1],
            c=mapped,
            s=s,
            alpha=alpha,
            linewidths=0,
        )
        if show_legend:
            handles = [
                Line2D([0], [0], marker="o", linestyle="", color=palette[cat], label=str(cat), markersize=5)
                for cat in categories
            ]
            if legend_loc == "outside right":
                ax.legend(handles=handles, title=color_label, loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(handles=handles, title=color_label, loc=legend_loc)

    if mask.any():
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)
    ax.set_aspect("equal")
    ax.set_title(title or color_label)
    if not show_axis:
        ax.set_axis_off()
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    return ax
