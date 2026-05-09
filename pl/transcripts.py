from __future__ import annotations

from typing import Union

import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

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

    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(img, cmap=colormap)
    plt.axis("off")

    if title == "":
        if len(valid_features) < 3:
            title = ", ".join(valid_features)
        else:
            title = ", ".join(valid_features[:3]) + "..."

    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def rasterize_rgb(xdata, genes_or_gene_sets, bin_size=8, fig_scale=10, gammas=[1, 1, 1], log=False, include_unassigned=False):
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
    pl.show()


def plot_binned_rgb(xdata, genes_or_gene_sets, norm="per_gene", fig_scale=10, gammas=[1, 1, 1], log=False, flip=True, bounds: tuple | None = None):
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


def plot_binned_greyscale(xdata, genes, fig_scale=10, gamma=1, log=False, flip=True, return_img=False, cmap="inferno"):
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
    pl.show()


def splat(
    data=None,
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
    genes: Union[None, str, list[str], dict] = None,
    gains=(1.0, 1.0, 1.0),
    bounds=None,
    pixel_size_um=1.0,
    sigma_um=2.0,
    ax=None,
    global_norm=False,
    smooth=True,
    show_ticks=False,
    show_legend: bool = True,
    legend_loc: str = "outside right",
):
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

    return rgb, disp, ax
