from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._shared import _load_local_module

try:
    from ..core.rois import _roi_bounds_um
except ImportError:
    _roi_bounds_um = _load_local_module("_xentools_core_rois_for_pl_boundaries", "core/rois.py")._roi_bounds_um

__all__ = ["plot_boundaries"]


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


def plot_boundaries(
    xdata,
    kind: str = "cell",
    color_by: Optional[str] = None,
    palette: Optional[dict] = None,
    facecolor="none",
    edgecolor="white",
    face_alpha: float = 0.3,
    edge_alpha: float = 0.8,
    linewidth: float = 0.5,
    bounds=None,
    ax=None,
    figsize: tuple = (8, 8),
    max_cells: Optional[int] = None,
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

    gdf = xdata.cell_boundaries if kind == "cell" else xdata.nucleus_boundaries
    if gdf is None or len(gdf) == 0:
        raise ValueError(
            f"No {kind} boundaries loaded. "
            "Check that boundary parquet files or cells.zarr.zip were available on init."
        )

    if bounds is None and getattr(xdata, "active_roi", None) is not None:
        xmin, xmax, ymin, ymax = _roi_bounds_um(xdata.active_roi)
    elif bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    else:
        xmin = ymin = xmax = ymax = None

    if xmin is not None:
        cx = gdf.geometry.centroid.x
        cy = gdf.geometry.centroid.y
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        gdf = gdf[mask]

    if max_cells is not None and len(gdf) > max_cells:
        gdf = gdf.sample(max_cells, random_state=0)

    if len(gdf) == 0:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    adata = getattr(xdata, "adata", None)
    if color_by is not None and adata is not None and color_by in adata.obs.columns:
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
            list(obs_col.cat.categories)
            if hasattr(obs_col, "cat")
            else sorted(obs_col.dropna().unique())
        )
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

    if ax is not None:
        y_lo, y_hi = ax.get_ylim()
    elif xmin is not None:
        y_lo, y_hi = ymin, ymax
    else:
        y_lo = gdf.geometry.bounds["miny"].min()
        y_hi = gdf.geometry.bounds["maxy"].max()

    def _flip_coords(coords):
        arr = np.array(coords)
        arr[:, 1] = y_lo + y_hi - arr[:, 1]
        return arr

    patches = []
    patch_facecolors = []
    for geom, rgba in zip(gdf.geometry, face_rgba):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            patches.append(MplPolygon(_flip_coords(poly.exterior.coords), closed=True))
            patch_facecolors.append(rgba)

    pc = PatchCollection(
        patches,
        facecolors=patch_facecolors,
        edgecolors=[edge_rgba] * len(patches),
        linewidths=linewidth,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_xlim(gdf.geometry.bounds["minx"].min(), gdf.geometry.bounds["maxx"].max())
        ax.set_ylim(y_lo, y_hi)

    ax.add_collection(pc)
    ax.set_aspect("equal")

    if color_by is not None and palette is not None:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=(*mcolors.to_rgb(c), face_alpha), edgecolor=edge_rgba, label=str(k))
            for k, c in palette.items()
            if k in (obs_col.values if "obs_col" in locals() else [])
        ]
        if handles:
            ax.legend(
                handles=handles,
                fontsize="small",
                labelcolor="white",
                facecolor="black",
                edgecolor="black",
                loc="upper right",
            )

    return ax
