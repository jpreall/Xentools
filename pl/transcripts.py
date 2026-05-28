from __future__ import annotations

from typing import Optional, Union
import warnings

import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from ._save import save_figure
from ._shared import _load_local_module

try:
    from ..core.transcripts import LazyTranscripts
except ImportError:
    _transcripts_mod = _load_local_module("_xentools_core_transcripts_for_pl", "core/transcripts.py")
    LazyTranscripts = _transcripts_mod.LazyTranscripts


__all__ = [
    "create_bins",
    "bin_expression",
    "create_binned_image",
    "rasterize",
    "rasterize_rgb",
    "plot_binned_rgb",
    "create_multilayer_image",
    "plot_binned_greyscale",
    "points",
    "splat",
]


def _is_lazy_transcripts(obj) -> bool:
    """
    Detect LazyTranscripts-like objects without relying on exact class identity.

    During standalone/module fallback imports, the same source class can be loaded
    under multiple module names, making ``isinstance(obj, LazyTranscripts)`` too
    brittle for public plotting entry points.
    """
    return (
        hasattr(obj, "query")
        and callable(getattr(obj, "query"))
        and hasattr(obj, "_gene_names")
        and hasattr(obj, "_tile_meta")
    )


def _normalize_genes_for_query(genes):
    if genes is None:
        return None
    if isinstance(genes, str):
        return [genes]
    if isinstance(genes, dict):
        return list(dict.fromkeys(gene for gene_list in genes.values() for gene in gene_list))
    return list(genes)


def _resolve_active_roi(data, bounds):
    if bounds is not None or not hasattr(data, "active_roi"):
        return None
    roi = getattr(data, "active_roi", None)
    if roi is None:
        roi = getattr(data, "subset_roi", None)
    return roi


def _filter_transcript_dataframe(
    df,
    genes=None,
    bounds=None,
    roi=None,
    assigned_only=None,
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
):
    if df is None or len(df) == 0:
        return df

    mask = np.ones(len(df), dtype=bool)
    query_genes = _normalize_genes_for_query(genes)
    if query_genes is not None:
        mask &= df[gene_col].isin(query_genes).to_numpy()

    if assigned_only is None:
        assigned_only = "cell_id" in df.columns
    if assigned_only and "cell_id" in df.columns:
        cell_ids = df["cell_id"].astype(str).to_numpy()
        mask &= (cell_ids != "UNASSIGNED") & (cell_ids != "") & (cell_ids != "nan")

    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        mask &= (
            (df[x_col].to_numpy() >= xmin)
            & (df[x_col].to_numpy() <= xmax)
            & (df[y_col].to_numpy() >= ymin)
            & (df[y_col].to_numpy() <= ymax)
        )

    out = df.loc[mask]
    if roi is not None and len(out):
        out = roi.crop_dataframe(out, x_col=x_col, y_col=y_col)
    return out


def _sample_dataframe(df, max_points=None, random_state=0):
    if max_points is None or len(df) <= max_points:
        return df.copy(), False
    return df.sample(n=int(max_points), random_state=random_state).copy(), True


def _priority_sample_lazy_transcripts(
    lazy_transcripts,
    *,
    genes=None,
    bounds=None,
    roi=None,
    quality="all",
    max_points=100_000,
    random_state=0,
):
    query_genes = _normalize_genes_for_query(genes)
    query_kwargs = {}
    if bounds is not None:
        query_kwargs.update(xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3])

    rng = np.random.default_rng(random_state)
    total = 0
    reservoir = []
    for _tile_key, tile_df in lazy_transcripts.iter_tiles(genes=query_genes, quality=quality, **query_kwargs):
        if roi is not None and len(tile_df):
            tile_df = roi.crop_dataframe(tile_df)
        if len(tile_df) == 0:
            continue

        total += len(tile_df)
        if max_points is None:
            reservoir.append(tile_df)
            continue

        tile_df = tile_df.copy()
        tile_df["_xentools_sample_priority"] = rng.random(len(tile_df))
        reservoir.append(tile_df)
        combined = pd.concat(reservoir, ignore_index=True)
        if len(combined) > max_points:
            combined = combined.nsmallest(int(max_points), "_xentools_sample_priority")
        reservoir = [combined]

    if not reservoir:
        return pd.DataFrame(columns=["x_location", "y_location", "feature_name"]), 0, False

    sampled = pd.concat(reservoir, ignore_index=True)
    if "_xentools_sample_priority" in sampled.columns:
        sampled = sampled.drop(columns="_xentools_sample_priority")
    return sampled, total, max_points is not None and total > max_points


def _prepare_point_dataframe(
    data,
    *,
    genes=None,
    bounds=None,
    quality="all",
    max_points=100_000,
    random_state=0,
    assigned_only=None,
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
):
    roi = _resolve_active_roi(data, bounds)
    if bounds is None and roi is not None:
        bounds = roi.bounds

    if hasattr(data, "trans") and not isinstance(data, pd.DataFrame):
        source = data.trans
    else:
        source = data

    if _is_lazy_transcripts(source):
        df, total, sampled = _priority_sample_lazy_transcripts(
            source,
            genes=genes,
            bounds=bounds,
            roi=roi,
            quality=quality,
            max_points=max_points,
            random_state=random_state,
        )
        return df, bounds, roi, total, sampled

    if not isinstance(source, pd.DataFrame):
        raise ValueError("data must be XenData, LazyTranscripts, or pd.DataFrame")

    df = _filter_transcript_dataframe(
        source,
        genes=genes,
        bounds=bounds,
        roi=roi,
        assigned_only=assigned_only,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
    )
    total = len(df)
    df, sampled = _sample_dataframe(df, max_points=max_points, random_state=random_state)
    return df, bounds, roi, total, sampled


def _point_groups(df, genes=None, gene_col="feature_name"):
    if genes is None:
        return pd.Series("transcripts", index=df.index, dtype=object), ["transcripts"]
    if isinstance(genes, dict):
        mapping = {}
        for label, gene_list in genes.items():
            for gene in gene_list:
                mapping[gene] = label
        groups = df[gene_col].map(mapping).fillna("other").astype(object)
        labels = [label for label in genes if label in set(groups)]
        if "other" in set(groups):
            labels.append("other")
        return groups, labels
    groups = df[gene_col].astype(object)
    labels = list(dict.fromkeys(_normalize_genes_for_query(genes) or groups.dropna().tolist()))
    labels = [label for label in labels if label in set(groups)]
    return groups, labels


def create_bins(df, bin_size=5):
    x_min, x_max = df["x_location"].min(), df["x_location"].max()
    y_min, y_max = df["y_location"].min(), df["y_location"].max()
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    return x_edges, y_edges


def bin_expression(df, bin_size=5, normalize=False):
    x_edges, y_edges = create_bins(df, bin_size=bin_size)
    counts, _, _ = np.histogram2d(df["y_location"], df["x_location"], bins=[y_edges, x_edges])
    if normalize:
        counts = counts / counts.max() if counts.max() > 0 else counts
    return counts


def create_binned_image(df, bin_size=5, colormap="Grays", return_array=False, vmax=None, vmin=None):
    from PIL import Image

    x_edges, y_edges = create_bins(df, bin_size=bin_size)
    counts, _, _ = np.histogram2d(df["y_location"], df["x_location"], bins=[y_edges, x_edges])

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = counts.max()

    if (vmax is not None) or (vmin is not None):
        counts = np.clip(counts, vmin, vmax)
        norm_counts = (counts - vmin) / (vmax - vmin)
    else:
        norm_counts = counts / counts.max() if counts.max() > 0 else counts

    colored_img = pl.get_cmap(colormap)(norm_counts)
    img_array = (colored_img[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    if return_array:
        return img, img_array
    return img


def rasterize(
    xdata,
    features=None,
    bin_size=10,
    colormap="Greys_r",
    vmax=None,
    vmin=None,
    title: str = "",
    return_img=False,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Rasterize selected transcript features into a simple binned image.

    This is a lightweight, legacy-style transcript visualization. It returns a
    PIL image when ``return_img=True``; otherwise it displays the image on a new
    Matplotlib figure. For richer spatial display, prefer ``splat()``.
    """
    if features is None:
        features = list(xdata.features)
        if title == "":
            title = "All Features"
    elif isinstance(features, str):
        features = [features]
    else:
        features = list(features)

    valid_features = []
    for feature in features:
        if feature in xdata.features:
            valid_features.append(feature)
        else:
            print(f'Feature "{feature}" not found in the dataset. Skipping...')

    if len(valid_features) == 0:
        print("No valid features found in the dataset. Exiting...")
        return None

    trans = xdata.trans
    if _is_lazy_transcripts(trans):
        bounds = None
        roi = getattr(xdata, "active_roi", None)
        if roi is None:
            roi = getattr(xdata, "subset_roi", None)
        if roi is not None:
            bounds = roi.bounds

        query_kwargs = {}
        if bounds is not None:
            query_kwargs.update(
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
            )
        toplot = trans.query(genes=valid_features, quality="all", **query_kwargs)
    else:
        toplot = trans[trans["feature_name"].isin(valid_features)]

    if toplot.empty:
        print("No transcripts found for the selected features. Exiting...")
        return None

    img = create_binned_image(
        toplot,
        bin_size=bin_size,
        colormap=colormap,
        vmax=vmax,
        vmin=vmin,
    )

    w, h = img.size
    dpi = plt.rcParams["figure.dpi"]

    if return_img:
        return img

    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img, cmap=colormap)
    ax.axis("off")

    if title == "":
        if len(valid_features) < 3:
            title = ", ".join(valid_features)
        else:
            title = ", ".join(valid_features[:3]) + "..."

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    plt.show()


def rasterize_rgb(xdata, genes_or_gene_sets, bin_size=8, fig_scale=10, gammas=[1, 1, 1], log=False, include_unassigned=False, save=None, save_kwargs: Optional[dict] = None):
    from PIL import Image

    df = xdata.trans
    if include_unassigned is False:
        df = df[df["cell_id"] != "UNASSIGNED"].copy()

    if isinstance(genes_or_gene_sets, list):
        genes = genes_or_gene_sets
        if len(genes) > 3:
            raise ValueError("Only 3 genes can be rasterized at a time.")
        if len(genes) < 1:
            raise ValueError("At least 1 gene must be rasterized.")
        gene_sets = {gene: [gene] for gene in genes}
    elif isinstance(genes_or_gene_sets, dict):
        gene_sets = genes_or_gene_sets
        if len(gene_sets) > 3:
            raise ValueError("Only 3 gene sets can be rasterized at a time.")
        if len(gene_sets) < 1:
            raise ValueError("At least 1 gene set must be rasterized.")
    else:
        raise ValueError("Input must be a list of genes or a dictionary of gene sets.")

    x_min, x_max = df["x_location"].min(), df["x_location"].max()
    y_min, y_max = df["y_location"].min(), df["y_location"].max()
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    aspect_ratio = (x_edges[-1] - x_edges[0]) / (y_edges[-1] - y_edges[0])

    x_idx = np.digitize(df["x_location"].values, x_edges) - 1
    y_idx = np.digitize(df["y_location"].values, y_edges) - 1
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    counts = np.zeros((3, shape[0], shape[1]), dtype=np.float32)

    for ch, (_, gene_list) in enumerate(gene_sets.items()):
        if ch >= 3:
            break
        mask = df["feature_name"].isin(gene_list).values
        if np.any(mask):
            np.add.at(counts[ch], (y_idx[mask], x_idx[mask]), 1)

    if log:
        counts = np.log1p(counts)

    for ch in range(3):
        max_val = counts[ch].max()
        counts[ch] = np.round(255 * counts[ch] / max_val) if max_val > 0 else 0

    merged = np.stack([counts[0], counts[1], counts[2]], axis=-1).clip(0, 255).astype(np.uint8)
    for i in range(3):
        channel_norm = merged[:, :, i] / 255.0
        channel_corr = 255 * np.power(channel_norm, 1.0 / gammas[i])
        merged[:, :, i] = np.clip(channel_corr, 0, 255).astype(np.uint8)

    im = Image.fromarray(merged)

    legend_handles = [Patch(color=color, label=label) for label, color in zip(list(gene_sets.keys()), ["red", "green", "blue"][: len(gene_sets)])]
    fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
    ax.imshow(im)
    ax.axis("off")
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="medium",
        labelcolor="white",
        facecolor="black",
        edgecolor="black",
    )
    pl.tight_layout()
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    pl.show()


def plot_binned_rgb(xdata, genes_or_gene_sets, norm="per_gene", fig_scale=10, gammas=[1, 1, 1], log=False, flip=True, bounds: tuple | None = None, save=None, save_kwargs: Optional[dict] = None):
    from PIL import Image
    import anndata as ad

    xmin, xmax, ymin, ymax = bounds if bounds is not None else (None, None, None, None)

    if hasattr(xdata, "binned_adata"):
        data = xdata.binned_adata
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if "spatial" not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")

    if isinstance(genes_or_gene_sets, list):
        genes = genes_or_gene_sets
        if len(genes) > 3:
            raise ValueError("Only 3 genes can be rasterized at a time.")
        if len(genes) < 1:
            raise ValueError("At least 1 gene must be rasterized.")
        gene_sets = {gene: [gene] for gene in genes}
    elif isinstance(genes_or_gene_sets, dict):
        gene_sets = genes_or_gene_sets
        if len(gene_sets) > 3:
            raise ValueError("Only 3 gene sets can be rasterized at a time.")
        if len(gene_sets) < 1:
            raise ValueError("At least 1 gene set must be rasterized.")
    else:
        raise ValueError("Input must be a list of genes or a dictionary of gene sets.")

    xy_coords = data.obsm["spatial"]
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w = int(np.abs(x_max - x_min)) + 1
    h = int(np.abs(y_max - y_min)) + 1
    aspect_ratio = w / h
    imdata = np.zeros((h, w, 3))
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)

    for n, (_, genes) in enumerate(gene_sets.items()):
        expr = data[:, genes].X.todense().sum(1).A1
        for xi, yi, intensity in zip(x_idx, y_idx, expr):
            imdata[yi, xi, n] = intensity

    if log:
        imdata = np.log1p(imdata)

    assert norm in ["per_gene", "global"], "Invalid normalization method. Choose 'per_gene' or 'global'."
    global_max = imdata.max()
    for ch in range(3):
        max_val = imdata[:, :, ch].max() if norm == "per_gene" else global_max
        imdata[:, :, ch] = np.round(255 * imdata[:, :, ch] / max_val) if max_val > 0 else 0

    if flip is True:
        imdata = imdata[::-1, :, :]

    for i in range(3):
        channel_norm = imdata[:, :, i] / 255.0
        channel_corr = 255 * np.power(channel_norm, 1.0 / gammas[i])
        imdata[:, :, i] = np.clip(channel_corr, 0, 255).astype(np.uint8)

    im = Image.fromarray(np.uint8(imdata)) if bounds is None else imdata[..., :3].astype(np.uint8)
    fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
    if bounds is None:
        ax.imshow(im)
    else:
        ax.imshow(imdata.astype(np.uint8), extent=[xmin, xmax, ymin, ymax])
    ax.axis("off")
    ax.legend(
        handles=[Patch(color=color, label=label) for label, color in zip(list(gene_sets.keys()), ["red", "green", "blue"][: len(gene_sets)])],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="medium",
        labelcolor="white",
        facecolor="black",
        edgecolor="black",
    )
    pl.tight_layout()
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    pl.show()


def create_multilayer_image(xdata, genes, log=False):
    import anndata as ad

    if hasattr(xdata, "binned_adata"):
        data = xdata.binned_adata
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if "spatial" not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")

    if genes is None:
        genes = data.var_names.tolist()
    elif isinstance(genes, str):
        genes = [genes]

    missing_genes = [gene for gene in genes if gene not in data.var_names]
    if missing_genes:
        raise ValueError(f"The following genes are not present in the binned AnnData object: {', '.join(missing_genes)}")

    coords = data.obsm["spatial"]
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x_min, y_min = x_coords.min(), y_coords.min()
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)
    w = int(x_coords.max() - x_min) + 1
    h = int(y_coords.max() - y_min) + 1
    imdata = np.zeros((h, w, len(genes)), dtype=np.uint16)

    data = data[:, genes].X
    if hasattr(data, "toarray"):
        data = data.toarray()

    if log:
        data = np.log2(data + 1)
        data = np.clip(data, 0, 255).astype(np.uint8)
    elif np.max(data) > 255:
        data = np.clip(data, 0, 2**16 - 1).astype(np.uint16)
    else:
        data = np.clip(data, 0, 255).astype(np.uint8)

    imdata[y_idx, x_idx, :] = data
    return imdata[::-1, :, :]


def plot_binned_greyscale(xdata, genes, fig_scale=10, gamma=1, log=False, flip=True, return_img=False, cmap="inferno", save=None, save_kwargs: Optional[dict] = None):
    import anndata as ad
    from scipy.sparse import issparse

    if hasattr(xdata, "binned_adata"):
        data = xdata.binned_adata
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if "spatial" not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")

    if isinstance(genes, str):
        genes = [genes]

    xy_coords = data.obsm["spatial"]
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w = int(np.abs(x_max - x_min)) + 1
    h = int(np.abs(y_max - y_min)) + 1
    aspect_ratio = w / h
    imdata = np.zeros((h, w, 1))
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)

    expr = data[:, genes].X.todense().sum(1).A1 if issparse(data[:, genes].X) else data[:, genes].X.sum(1)
    imdata[y_idx, x_idx, 0] = expr
    if log:
        imdata = np.log1p(imdata)
    imdata = imdata[:, :, 0]
    norm_imdata = imdata / imdata.max()
    imdata = np.round(255 * norm_imdata)
    channel_norm = imdata / 255.0
    imdata = np.clip(255 * np.power(channel_norm, 1.0 / gamma), 0, 255)
    if flip is True:
        imdata = imdata[::-1, :]
    if return_img:
        return imdata
    fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
    ax.imshow(imdata, cmap=cmap)
    ax.axis("off")
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    pl.show()


def points(
    data=None,
    genes: Union[None, str, list[str], dict] = None,
    *,
    bounds=None,
    quality: str = "all",
    max_points: Optional[int] = 100_000,
    random_state: Optional[int] = 0,
    x_col: str = "x_location",
    y_col: str = "y_location",
    gene_col: str = "feature_name",
    color: str = "white",
    palette: Optional[dict] = None,
    cmap: str = "tab20",
    marker: str = "o",
    markers: Optional[dict] = None,
    s: float = 8,
    alpha: float = 0.75,
    linewidths: float = 0,
    edgecolors="none",
    ax=None,
    figsize=(8, 8),
    background: str = "black",
    show_axis: bool = False,
    show_legend: bool = True,
    legend_loc: str = "outside right",
    legend_title: Optional[str] = None,
    max_legend_items: int = 20,
    preserve_limits: bool = True,
    rasterized: bool = True,
    warn_on_sample: bool = True,
    assigned_only: Optional[bool] = None,
    save=None,
    save_kwargs: Optional[dict] = None,
    return_data: bool = False,
):
    """
    Plot individual transcripts as point markers.

    This is intended for focused ROIs and compositing with existing image,
    splat, or boundary axes. For lazy zarr-backed transcripts, capped plotting
    uses streaming tile reads plus random priority sampling, so large queries
    do not need to materialize every matching transcript before downsampling.

    Set ``max_points=None`` to draw every matching transcript.
    """
    import matplotlib.colors as mcolors

    owns_ax = ax is None
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be a positive integer or None.")

    df, resolved_bounds, _roi, total_points, sampled = _prepare_point_dataframe(
        data,
        genes=genes,
        bounds=bounds,
        quality=quality,
        max_points=max_points,
        random_state=random_state,
        assigned_only=assigned_only,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
    )

    if sampled and warn_on_sample:
        warnings.warn(
            f"points() sampled {len(df):,} of {total_points:,} matching transcripts. "
            "Pass max_points=None to draw all points, or increase max_points.",
            RuntimeWarning,
            stacklevel=2,
        )

    if owns_ax:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
    else:
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()

    if len(df) == 0:
        if owns_ax and not show_axis:
            ax.set_axis_off()
        save_figure(ax, save=save, save_kwargs=save_kwargs)
        return (ax, df) if return_data else ax

    if resolved_bounds is None:
        xmin = float(df[x_col].min())
        xmax = float(df[x_col].max())
        ymin = float(df[y_col].min())
        ymax = float(df[y_col].max())
    else:
        xmin, xmax, ymin, ymax = map(float, resolved_bounds)

    y_lo, y_hi = (ymin, ymax) if owns_ax else ax.get_ylim()
    plot_y = y_lo + y_hi - df[y_col].to_numpy()
    groups, labels = _point_groups(df, genes=genes, gene_col=gene_col)

    if genes is None:
        ax.scatter(
            df[x_col].to_numpy(),
            plot_y,
            c=color,
            marker=marker,
            s=s,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
            rasterized=rasterized,
        )
    else:
        if palette is None:
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(1, len(labels))))
            palette = {label: mcolors.to_hex(colors[i]) for i, label in enumerate(labels)}

        handles = []
        group_values = groups.to_numpy()
        for label in labels:
            mask = group_values == label
            if not mask.any():
                continue
            this_marker = markers.get(label, marker) if markers is not None else marker
            this_color = palette.get(label, color)
            ax.scatter(
                df.loc[mask, x_col].to_numpy(),
                plot_y[mask],
                c=this_color,
                marker=this_marker,
                s=s,
                alpha=alpha,
                linewidths=linewidths,
                edgecolors=edgecolors,
                rasterized=rasterized,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=this_marker,
                    linestyle="",
                    color=this_color,
                    label=str(label),
                    markersize=max(4, np.sqrt(s)),
                    alpha=alpha,
                )
            )

        if show_legend and handles and len(handles) <= max_legend_items:
            legend_kwargs = dict(
                handles=handles,
                title=legend_title or ("gene set" if isinstance(genes, dict) else gene_col),
                fontsize="small",
                title_fontsize="small",
                labelcolor="white",
                facecolor="black",
                edgecolor="black",
                framealpha=0.85,
            )
            if legend_loc == "outside right":
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, **legend_kwargs)
            elif legend_loc == "outside left":
                ax.legend(loc="center right", bbox_to_anchor=(-0.02, 0.5), borderaxespad=0, **legend_kwargs)
            elif legend_loc == "outside bottom":
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.02),
                    borderaxespad=0,
                    ncol=min(len(handles), 6),
                    **legend_kwargs,
                )
            elif legend_loc == "outside top":
                ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    borderaxespad=0,
                    ncol=min(len(handles), 6),
                    **legend_kwargs,
                )
            else:
                ax.legend(loc=legend_loc, **legend_kwargs)

    if owns_ax:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    elif preserve_limits:
        ax.set_xlim(old_xlim)
        ax.set_ylim(old_ylim)

    ax.set_aspect("equal")
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    ax.grid(False)

    save_figure(ax, save=save, save_kwargs=save_kwargs)
    return (ax, df) if return_data else ax


def splat(
    data=None,
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
    genes: Union[None, str, list[str], dict] = None,
    gains=1.0,
    bounds=None,
    pixel_size_um=1.0,
    sigma_um=2.0,
    ax=None,
    global_norm=False,
    smooth=True,
    show_ticks=False,
    show_legend: bool = True,
    legend_loc: str = "outside right",
    return_array=False,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Rasterize transcript positions into one or more display channels.

    By default this function behaves like a plotting function and returns the
    matplotlib axes containing the rendered image. Set ``return_array=True`` or
    ``return_array="display"`` to return the normalized display array instead,
    or ``return_array="raw"`` to return the raw binned/smoothed raster.
    """
    if hasattr(data, "trans") and not isinstance(data, pd.DataFrame):
        df = data.trans
    elif isinstance(data, pd.DataFrame):
        df = data
    elif _is_lazy_transcripts(data):
        df = data
    else:
        raise ValueError("data must be XenData, LazyTranscripts, or pd.DataFrame")

    if _is_lazy_transcripts(df):
        if genes is None:
            query_genes = None
        elif isinstance(genes, str):
            query_genes = [genes]
        elif isinstance(genes, list):
            query_genes = genes
        elif isinstance(genes, dict):
            query_genes = list({g for gs in genes.values() for g in gs})
        else:
            query_genes = None

        if bounds is not None:
            qxmin, qxmax, qymin, qymax = bounds
        else:
            qxmin = qxmax = qymin = qymax = None

        df = df.query(xmin=qxmin, xmax=qxmax, ymin=qymin, ymax=qymax, genes=query_genes)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    g = df[gene_col].to_numpy()

    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    nx = int(np.ceil((xmax - xmin) / pixel_size_um))
    ny = int(np.ceil((ymax - ymin) / pixel_size_um))
    sigma_px = sigma_um / pixel_size_um
    gene_signatures = None

    if genes is None:
        chan_names = ["Random transcripts"]
    elif isinstance(genes, dict):
        chan_names = list(genes.keys())
        gene_signatures = genes
    elif isinstance(genes, str):
        chan_names = [genes]
    else:
        chan_names = list(genes)

    n_channels = len(chan_names)
    if n_channels > 3:
        raise ValueError(
            "plot_splat can display at most 3 gene or gene-set channels as RGB. "
            f"You passed {n_channels}. Select up to 3 entries for splat plotting, "
            "or pass the full dictionary to a summary plot such as scanpy.pl.dotplot."
        )
    if np.isscalar(gains):
        gains = (float(gains),) * n_channels
    else:
        gains = tuple(gains)
        if len(gains) != n_channels:
            raise ValueError(f"gains length ({len(gains)}) must match number of channels ({n_channels})")

    rgb = np.zeros((ny, nx, n_channels), dtype=np.float32)
    codes, uniq = pd.factorize(g)
    gene_to_code = {gene: i for i, gene in enumerate(uniq)}
    chan_assignment = np.full(len(uniq), -1, dtype=np.int32)
    for k, chan_name in enumerate(chan_names):
        if gene_signatures is not None:
            for gene in gene_signatures[chan_name]:
                if gene in gene_to_code:
                    chan_assignment[gene_to_code[gene]] = k
        elif chan_name in gene_to_code:
            chan_assignment[gene_to_code[chan_name]] = k

    transcript_channel = chan_assignment[codes]
    x_idx_all = ((x - xmin) / pixel_size_um).astype(np.int32)
    y_idx_all = ((ymax - y) / pixel_size_um).astype(np.int32)
    valid_all = (x_idx_all >= 0) & (x_idx_all < nx) & (y_idx_all >= 0) & (y_idx_all < ny)

    for k in range(n_channels):
        mask = (transcript_channel == k) & valid_all
        if not mask.any():
            continue
        flat_idx = y_idx_all[mask].astype(np.int64) * nx + x_idx_all[mask]
        rgb[..., k] = np.bincount(flat_idx, minlength=ny * nx).reshape(ny, nx).astype(np.float32)

    if smooth:
        rgb = gaussian_filter(rgb, sigma=[sigma_px, sigma_px, 0], mode="nearest")

    if global_norm:
        base = rgb.copy()
        m = base.max()
        disp = base / m if m > 0 else base
    else:
        base = rgb.copy()
        disp = np.zeros_like(base)
        for k in range(n_channels):
            m = base[..., k].max()
            if m > 0:
                disp[..., k] = base[..., k] / m

    disp = np.clip(disp * np.array(gains, dtype=float).reshape(1, 1, -1), 0, 1)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 12) if n_channels == 3 else (6, 6))

    extent = [xmin, xmax, ymin, ymax]
    if n_channels == 1:
        ax.imshow(disp[..., 0], extent=extent, origin="lower", interpolation="nearest", cmap="gray")
    elif n_channels >= 3:
        ax.imshow(disp[..., :3], extent=extent, origin="lower", interpolation="nearest")

    if show_ticks:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(False)

    channel_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if show_legend and chan_names != ["Random transcripts"]:
        if n_channels == 1:
            legend_colors = [plt.get_cmap("gray")(0.9)[:3]]
        else:
            legend_colors = [
                channel_colors[i] if i < len(channel_colors) else plt.get_cmap("hsv")(i / n_channels)[:3]
                for i in range(n_channels)
            ]

        handles = [Patch(color=c, label=n) for c, n in zip(legend_colors, chan_names)]
        legend_kwargs = dict(handles=handles, fontsize="medium", labelcolor="white", facecolor="black", edgecolor="black")
        if legend_loc == "outside right":
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **legend_kwargs)
        elif legend_loc == "outside left":
            ax.legend(loc="center right", bbox_to_anchor=(0, 0.5), **legend_kwargs)
        elif legend_loc == "outside bottom":
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=min(n_channels, 4), **legend_kwargs)
        elif legend_loc == "outside top":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=min(n_channels, 4), **legend_kwargs)
        else:
            ax.legend(loc=legend_loc, **legend_kwargs)
    else:
        ax.set_title(", ".join(chan_names[:3]))

    save_figure(ax, save=save, save_kwargs=save_kwargs)
    if return_array in (False, None):
        return ax
    if return_array is True or return_array == "display":
        return disp
    if return_array == "raw":
        return rgb
    if return_array == "_all":
        return rgb, disp, ax
    raise ValueError("return_array must be False, True, 'display', or 'raw'.")
