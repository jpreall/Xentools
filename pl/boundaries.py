from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._shared import _load_local_module
from ._save import save_figure

try:
    from ..core.rois import _roi_bounds_um
except ImportError:
    _roi_bounds_um = _load_local_module("_xentools_core_rois_for_pl_boundaries", "core/rois.py")._roi_bounds_um

__all__ = ["plot_boundaries", "plot_cells"]


def _generate_palette(n):
    import colorsys
    import random

    if n <= 0:
        return []
    hues = [i / n for i in range(n)]
    return [
        "#{:02X}{:02X}{:02X}".format(
            int(r * 255),
            int(g * 255),
            int(b * 255),
        )
        for r, g, b in (
            colorsys.hls_to_rgb(h, 0.5, random.uniform(0.5, 0.9))
            for h in hues
        )
    ]


def _normalize_gene_list(genes):
    if genes is None:
        return None
    if isinstance(genes, str):
        return [genes]
    return list(genes)


def _matrix_to_1d(arr):
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    arr = np.asarray(arr)
    return arr.ravel()


def _expression_series(adata, genes, layer=None):
    genes = _normalize_gene_list(genes)
    if not genes:
        return None, None

    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Gene(s) not found in adata.var_names: {missing}")

    matrix = adata[:, genes].layers[layer] if layer is not None else adata[:, genes].X
    if len(genes) == 1:
        values = _matrix_to_1d(matrix)
        label = genes[0]
    else:
        values = _matrix_to_1d(matrix.sum(axis=1))
        label = " + ".join(genes)
    return pd.Series(values, index=adata.obs_names, dtype=float), label


def _nearest_obs_values(gdf, adata, values):
    from scipy.spatial import KDTree

    obs = adata.obs[["x_centroid", "y_centroid"]].copy()
    obs["_value"] = values.reindex(adata.obs_names)
    obs = obs.dropna(subset=["x_centroid", "y_centroid", "_value"])
    if obs.empty:
        return pd.Series(np.nan, index=gdf.index, dtype=float)

    tree = KDTree(obs[["x_centroid", "y_centroid"]].values)
    bnd_cx = gdf.geometry.centroid.x.values
    bnd_cy = gdf.geometry.centroid.y.values
    _, nn_idx = tree.query(np.column_stack([bnd_cx, bnd_cy]))
    return pd.Series(obs["_value"].iloc[nn_idx].values, index=gdf.index, dtype=float)


def plot_boundaries(
    xdata,
    kind: str = "cell",
    color_by: Optional[str] = None,
    genes=None,
    layer: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    palette: Optional[dict] = None,
    facecolor="none",
    edgecolor="white",
    face_alpha: float = 0.5,
    edge_alpha: float = 0.2,
    linewidth: float = 0.5,
    bounds=None,
    ax=None,
    figsize: tuple = (8, 8),
    max_cells: Optional[int] = None,
    show_legend: bool = True,
    legend_loc: str = "outside right",
    legend_title: Optional[str] = None,
    background: str = "black",
    show_axis: bool = False,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Overlay cell or nucleus boundary polygons on an axes.

    Parameters match ``XenData.plot_boundaries()``. The first argument is any
    object with ``cell_boundaries``/``nucleus_boundaries`` and optional
    ``adata``/``active_roi`` attributes.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.cm import ScalarMappable

    owns_ax = ax is None
    if genes is not None and color_by is not None:
        raise ValueError("Use either genes=... for expression coloring or color_by=..., not both.")

    gdf = xdata.cell_boundaries if kind == "cell" else xdata.nucleus_boundaries
    if gdf is None:
        raise ValueError(
            f"No {kind} boundaries loaded. "
            "Check that boundary parquet files or cells.zarr.zip were available on init."
        )

    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    elif ax is not None and hasattr(gdf, "query_bounds"):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmin, xmax = min(xlim), max(xlim)
        ymin, ymax = min(ylim), max(ylim)
    elif getattr(xdata, "active_roi", None) is not None:
        xmin, xmax, ymin, ymax = _roi_bounds_um(xdata.active_roi)
    else:
        xmin = ymin = xmax = ymax = None

    if xmin is not None and hasattr(gdf, "query_ids"):
        adata = getattr(xdata, "adata", None)
        if (
            adata is not None
            and "x_centroid" in adata.obs.columns
            and "y_centroid" in adata.obs.columns
        ):
            obs = adata.obs
            candidate_ids = obs.index[
                (obs["x_centroid"] >= xmin)
                & (obs["x_centroid"] <= xmax)
                & (obs["y_centroid"] >= ymin)
                & (obs["y_centroid"] <= ymax)
            ]
            gdf = gdf.query_ids(candidate_ids)
        else:
            gdf = gdf.query_bounds((xmin, xmax, ymin, ymax))
    elif xmin is not None and hasattr(gdf, "query_bounds"):
        gdf = gdf.query_bounds((xmin, xmax, ymin, ymax))
    elif xmin is not None:
        cx = gdf.geometry.centroid.x
        cy = gdf.geometry.centroid.y
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        gdf = gdf[mask]

    if len(gdf) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor(background)
            ax.set_facecolor(background)
            if not show_axis:
                ax.set_axis_off()
        save_figure(ax, save=save, save_kwargs=save_kwargs)
        return ax

    if max_cells is not None and len(gdf) > max_cells:
        gdf = gdf.sample(max_cells, random_state=0)

    adata = getattr(xdata, "adata", None)
    if color_by is not None and (adata is None or color_by not in adata.obs.columns):
        available = [] if adata is None else list(adata.obs.columns)
        preview = ", ".join(map(str, available[:10]))
        suffix = "" if len(available) <= 10 else ", ..."
        raise ValueError(
            f"color_by={color_by!r} was not found in xdata.adata.obs. "
            f"Available obs columns: [{preview}{suffix}]"
        )

    expression_label = None
    expression_norm = None
    expression_cmap = None
    if genes is not None:
        if adata is None:
            raise ValueError("Expression coloring requires xdata.adata.")
        expression_values, expression_label = _expression_series(adata, genes, layer=layer)
        expr_col = expression_values.reindex(gdf.index)
        if expr_col.isna().all():
            expr_col = _nearest_obs_values(gdf, adata, expression_values)

        finite = expr_col[np.isfinite(expr_col)]
        if finite.empty:
            face_rgba = [(0, 0, 0, 0)] * len(gdf)
        else:
            expression_cmap = plt.get_cmap(cmap)
            if vmin is None:
                vmin = float(finite.min())
            if vmax is None:
                vmax = float(finite.max())
            if vmax == vmin:
                vmax = vmin + 1e-12
            expression_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            raw_colors = expression_cmap(expression_norm(expr_col.fillna(vmin).values))
            raw_colors[:, 3] = np.where(expr_col.isna().values, 0.0, face_alpha)
            face_rgba = [tuple(color) for color in raw_colors]
    elif color_by is not None and adata is not None and color_by in adata.obs.columns:
        obs_col = adata.obs[color_by].reindex(gdf.index)

        if obs_col.isna().all():
            from scipy.spatial import KDTree

            obs_sub = adata.obs[["x_centroid", "y_centroid", color_by]].dropna(
                subset=["x_centroid", "y_centroid"]
            )
            tree = KDTree(obs_sub[["x_centroid", "y_centroid"]].values)
            bnd_cx = gdf.geometry.centroid.x.values
            bnd_cy = gdf.geometry.centroid.y.values
            _, nn_idx = tree.query(np.column_stack([bnd_cx, bnd_cy]))
            obs_col = pd.Series(
                obs_sub[color_by].iloc[nn_idx].values,
                index=gdf.index,
            )

        categories = (
            None
            if pd.api.types.is_numeric_dtype(obs_col)
            else (
                list(obs_col.cat.categories)
                if hasattr(obs_col, "cat")
                else sorted(obs_col.dropna().unique())
            )
        )
        if categories is None:
            expression_label = color_by
            finite = obs_col[np.isfinite(obs_col)]
            if finite.empty:
                face_rgba = [(0, 0, 0, 0)] * len(gdf)
            else:
                expression_cmap = plt.get_cmap(cmap)
                if vmin is None:
                    vmin = float(finite.min())
                if vmax is None:
                    vmax = float(finite.max())
                if vmax == vmin:
                    vmax = vmin + 1e-12
                expression_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                raw_colors = expression_cmap(expression_norm(obs_col.fillna(vmin).values))
                raw_colors[:, 3] = np.where(obs_col.isna().values, 0.0, face_alpha)
                face_rgba = [tuple(color) for color in raw_colors]
        else:
            if palette is None:
                palette = dict(zip(categories, _generate_palette(len(categories))))
            raw_colors = obs_col.astype(object).map(palette).fillna("gray").tolist()
            face_rgba = [(*mcolors.to_rgb(c), face_alpha) for c in raw_colors]
    else:
        if facecolor == "none":
            face_rgba = [(0, 0, 0, 0)] * len(gdf)
        else:
            rgb = mcolors.to_rgb(facecolor)
            face_rgba = [(*rgb, face_alpha)] * len(gdf)

    edge_rgb = mcolors.to_rgb(edgecolor)
    edge_rgba = (*edge_rgb, edge_alpha)

    if xmin is not None:
        y_lo, y_hi = ymin, ymax
    else:
        y_lo = gdf.geometry.bounds["miny"].min()
        y_hi = gdf.geometry.bounds["maxy"].max()

    patches = []
    patch_facecolors = []
    for geom, rgba in zip(gdf.geometry, face_rgba):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            patches.append(MplPolygon(np.asarray(poly.exterior.coords), closed=True))
            patch_facecolors.append(rgba)

    pc = PatchCollection(
        patches,
        facecolors=patch_facecolors,
        edgecolors=[edge_rgba] * len(patches),
        linewidths=linewidth,
    )

    if owns_ax:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.set_aspect("equal")
        if xmin is not None:
            ax.set_xlim(xmin, xmax)
        else:
            ax.set_xlim(gdf.geometry.bounds["minx"].min(), gdf.geometry.bounds["maxx"].max())
        ax.set_ylim(y_hi, y_lo)

    ax.add_collection(pc)
    ax.set_aspect("equal")
    if owns_ax and not show_axis:
        ax.set_axis_off()

    def _place_legend(ax, handles):
        title = color_by if legend_title is None else legend_title
        legend_kwargs = dict(
            handles=handles,
            title=title,
            fontsize="small",
            title_fontsize="small",
            labelcolor="white",
            facecolor="black",
            edgecolor="black",
            framealpha=0.85,
        )

        if legend_loc == "outside right":
            return ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                **legend_kwargs,
            )
        if legend_loc == "outside left":
            return ax.legend(
                loc="center right",
                bbox_to_anchor=(-0.02, 0.5),
                borderaxespad=0,
                **legend_kwargs,
            )
        if legend_loc == "outside bottom":
            ncol = min(max(len(handles), 1), 6)
            return ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                borderaxespad=0,
                ncol=ncol,
                **legend_kwargs,
            )
        if legend_loc == "outside top":
            ncol = min(max(len(handles), 1), 6)
            return ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                borderaxespad=0,
                ncol=ncol,
                **legend_kwargs,
            )
        return ax.legend(loc=legend_loc, **legend_kwargs)

    if show_legend and color_by is not None and palette is not None:
        from matplotlib.patches import Patch

        observed = set(obs_col.dropna().astype(object)) if "obs_col" in locals() else set()
        handles = [
            Patch(facecolor=(*mcolors.to_rgb(c), face_alpha), edgecolor=edge_rgba, label=str(k))
            for k, c in palette.items()
            if k in observed
        ]
        if handles:
            _place_legend(ax, handles)
    elif show_colorbar and expression_norm is not None and expression_cmap is not None:
        label = colorbar_label if colorbar_label is not None else expression_label
        sm = ScalarMappable(norm=expression_norm, cmap=expression_cmap)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cbar.set_label(label)

    save_figure(ax, save=save, save_kwargs=save_kwargs)
    return ax


def plot_cells(
    xdata,
    genes=None,
    color_by: Optional[str] = None,
    layer: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    palette: Optional[dict] = None,
    facecolor="none",
    edgecolor="white",
    face_alpha: float = 0.8,
    edge_alpha: float = 0.2,
    linewidth: float = 0.5,
    bounds=None,
    ax=None,
    figsize: tuple = (8, 8),
    max_cells: Optional[int] = None,
    show_legend: bool = True,
    legend_loc: str = "outside right",
    legend_title: Optional[str] = None,
    background: str = "black",
    show_axis: bool = False,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Plot cell boundary polygons, optionally filled by gene expression.

    ``genes`` may be a single gene or a list of genes. Lists are summed per
    cell before coloring. When ``genes`` is omitted, ``color_by`` can be used
    to color cells by an ``adata.obs`` annotation.
    """
    return plot_boundaries(
        xdata,
        kind="cell",
        genes=genes,
        color_by=color_by,
        layer=layer,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        palette=palette,
        facecolor=facecolor,
        edgecolor=edgecolor,
        face_alpha=face_alpha,
        edge_alpha=edge_alpha,
        linewidth=linewidth,
        bounds=bounds,
        ax=ax,
        figsize=figsize,
        max_cells=max_cells,
        show_legend=show_legend,
        legend_loc=legend_loc,
        legend_title=legend_title,
        background=background,
        show_axis=show_axis,
        save=save,
        save_kwargs=save_kwargs,
    )
