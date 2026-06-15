from __future__ import annotations

from typing import Optional

import numpy as np

from ._save import save_figure


__all__ = ["render"]


_SPLAT_OPTION_KEYS = {
    "genes",
    "gains",
    "pixel_size_um",
    "sigma_um",
    "global_norm",
    "smooth",
    "show_ticks",
    "show_legend",
    "legend_loc",
    "splat_alpha",
    "splat_cmap",
}

_BINNED_SPLAT_OPTION_KEYS = {
    "genes",
    "gains",
    "sigma_um",
    "global_norm",
    "smooth",
    "show_ticks",
    "show_legend",
    "legend_loc",
    "splat_alpha",
    "splat_cmap",
}

_POINT_OPTION_KEYS = {
    "genes",
    "max_points",
    "random_state",
    "color",
    "palette",
    "cmap",
    "marker",
    "markers",
    "s",
    "alpha",
    "linewidths",
    "edgecolors",
    "show_legend",
    "legend_loc",
    "legend_title",
    "assigned_only",
}

_IMAGE_OPTION_KEYS = {
    "channel",
    "level",
    "cmap",
    "vmin",
    "vmax",
    "alpha",
    "color",
    "blend",
    "clip_percentile",
    "z_index",
}

_RGB_COLORS = ("red", "green", "blue")


def _normalize_legend_setting(legend):
    if isinstance(legend, str):
        return legend.lower() not in {"false", "none", "off", "no"}
    return bool(legend)


def _legend_gene_channel_names(genes):
    if genes is None:
        return []
    if isinstance(genes, str):
        return [genes]
    if isinstance(genes, dict):
        return list(genes.keys())
    return list(genes)


def _legend_point_colors(names, point_opts):
    if not names:
        return []

    palette = point_opts.get("palette")
    color = point_opts.get("color")
    if isinstance(palette, dict):
        return [palette.get(name, color or "white") for name in names]
    if color is not None:
        return [color] * len(names)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = point_opts.get("cmap", "tab20")
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(1, len(names))))
    return [mcolors.to_hex(colors[i]) for i in range(len(names))]


def _legend_image_color(channel, spec, index):
    color = spec.get("color")
    if color is not None:
        return color
    cmap = spec.get("cmap")
    if cmap is not None:
        cmap_name = getattr(cmap, "name", str(cmap)).lower()
        if "gray" in cmap_name or "grey" in cmap_name:
            return "0.7"
        if "blue" in cmap_name:
            return "royalblue"
        if "green" in cmap_name:
            return "lime"
        if "red" in cmap_name:
            return "red"
    if index == 0:
        name = str(channel).lower()
        if name == "dapi" or "dapi" in name:
            return "0.7"
        return "0.7"
    return _default_image_color(channel, index - 1)


def _categorical_values(values):
    import pandas as pd

    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return None
    if hasattr(series, "cat"):
        observed = set(series.dropna().astype(object))
        categories = [cat for cat in series.cat.categories if cat in observed]
    else:
        categories = sorted(series.dropna().astype(object).unique())
    return categories


def _cell_boundaries_for_legend(xdata, bounds):
    gdf = getattr(xdata, "cell_boundaries", None)
    if gdf is None or bounds is None:
        return gdf

    xmin, xmax, ymin, ymax = bounds
    adata = getattr(xdata, "adata", None)
    if (
        hasattr(gdf, "query_ids")
        and adata is not None
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
        return gdf.query_ids(candidate_ids)
    if hasattr(gdf, "query_bounds"):
        return gdf.query_bounds((xmin, xmax, ymin, ymax))
    if hasattr(gdf, "geometry"):
        cx = gdf.geometry.centroid.x
        cy = gdf.geometry.centroid.y
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        return gdf[mask]
    return gdf


def _cell_label_values_for_legend(xdata, cell_opts, bounds):
    color_by = cell_opts.get("color_by")
    if color_by is None:
        return None

    gdf = _cell_boundaries_for_legend(xdata, bounds)
    adata = getattr(xdata, "adata", None)
    if adata is not None and color_by in adata.obs.columns:
        if gdf is not None and hasattr(gdf, "index"):
            obs_col = adata.obs[color_by].reindex(gdf.index)
            if obs_col.isna().all() and {"x_centroid", "y_centroid", color_by}.issubset(adata.obs.columns):
                from scipy.spatial import KDTree

                obs_sub = adata.obs[["x_centroid", "y_centroid", color_by]].dropna(
                    subset=["x_centroid", "y_centroid"]
                )
                if not obs_sub.empty and hasattr(gdf, "geometry") and len(gdf) > 0:
                    tree = KDTree(obs_sub[["x_centroid", "y_centroid"]].values)
                    bnd_cx = gdf.geometry.centroid.x.values
                    bnd_cy = gdf.geometry.centroid.y.values
                    _, nn_idx = tree.query(np.column_stack([bnd_cx, bnd_cy]))
                    return obs_sub[color_by].iloc[nn_idx].values
            return obs_col
        return adata.obs[color_by]

    return None


def _categories_for_cells(xdata, cell_opts, bounds):
    values = _cell_label_values_for_legend(xdata, cell_opts, bounds)
    if values is None:
        return None
    return _categorical_values(values)


def _generate_palette(n):
    import colorsys

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
            colorsys.hls_to_rgb(h, 0.52, 0.75)
            for h in hues
        )
    ]


def _collect_render_legend(
    xdata,
    *,
    image_specs,
    splat_opts,
    binned_splat_opts,
    point_opts,
    cell_opts,
    bounds,
    legend_max_items,
):
    sections = []

    if image_specs:
        entries = []
        for i, spec in enumerate(image_specs):
            channel = spec.get("channel")
            entries.append(
                {
                    "kind": "patch",
                    "label": f"{channel}",
                    "color": _legend_image_color(channel, spec, i),
                    "alpha": 1.0,
                }
            )
        sections.append(("Images", entries))

    if splat_opts is not None:
        names = _legend_gene_channel_names(splat_opts.get("genes"))
        entries = [
            {"kind": "patch", "label": name, "color": _RGB_COLORS[i], "alpha": 1.0}
            for i, name in enumerate(names[:3])
        ]
        if entries:
            sections.append(("Transcript density", entries))

    if binned_splat_opts is not None:
        names = _legend_gene_channel_names(binned_splat_opts.get("genes"))
        entries = [
            {"kind": "patch", "label": name, "color": _RGB_COLORS[i], "alpha": 1.0}
            for i, name in enumerate(names[:3])
        ]
        if entries:
            sections.append(("Binned transcript density", entries))

    if point_opts is not None:
        names = _legend_gene_channel_names(point_opts.get("genes"))
        marker = point_opts.get("marker", "o")
        alpha = float(point_opts.get("alpha", 0.75))
        if names:
            entries = []
            point_colors = _legend_point_colors(names, point_opts)
            for name, this_color in zip(names[:legend_max_items], point_colors[:legend_max_items]):
                entries.append({"kind": "point", "label": name, "color": this_color, "marker": marker, "alpha": alpha})
            if len(names) > legend_max_items:
                entries.append({"kind": "text", "label": f"... {len(names) - legend_max_items} more", "color": "none"})
        else:
            color = point_opts.get("color", "white")
            entries = [{"kind": "point", "label": "transcripts", "color": color, "marker": marker, "alpha": alpha}]
        sections.append(("Transcript points", entries))

    if cell_opts is not None:
        entries = []
        face_alpha = float(cell_opts.get("face_alpha", 0.3))
        edge_alpha = float(cell_opts.get("edge_alpha", 0.2))
        edgecolor = cell_opts.get("edgecolor", "white")
        color_by = cell_opts.get("color_by")
        genes = cell_opts.get("genes")

        categories = _categories_for_cells(xdata, cell_opts, bounds)
        if categories is not None:
            palette = cell_opts.get("palette")
            if palette is None:
                palette = dict(zip(categories, _generate_palette(len(categories))))
                cell_opts["palette"] = palette
            for category in categories[:legend_max_items]:
                entries.append(
                    {
                        "kind": "patch",
                        "label": str(category),
                        "color": palette.get(category, "gray"),
                        "edgecolor": edgecolor,
                        "alpha": face_alpha,
                    }
                )
            if len(categories) > legend_max_items:
                entries.append({"kind": "text", "label": f"... {len(categories) - legend_max_items} more", "color": "none"})
            title = f"Cells: {color_by}"
        elif genes is not None:
            label = genes if isinstance(genes, str) else " + ".join(map(str, genes))
            entries.append({"kind": "gradient", "label": label, "color": cell_opts.get("cmap", "viridis")})
            title = "Cells: expression"
        elif cell_opts.get("facecolor", "none") != "none":
            entries.append(
                {
                    "kind": "patch",
                    "label": "cell fill",
                    "color": cell_opts.get("facecolor", "white"),
                    "edgecolor": edgecolor,
                    "alpha": face_alpha,
                }
            )
            title = "Cells"
        else:
            entries.append({"kind": "line", "label": "cell outlines", "color": edgecolor, "alpha": edge_alpha})
            title = "Cells"
        if entries:
            sections.append((title, entries))

    return sections


def _draw_combined_legend(
    ax,
    sections,
    *,
    legend_loc="outside right",
    legend_title=None,
    fontsize="small",
):
    if not sections:
        return None

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    labels = []
    for section_title, entries in sections:
        handles.append(Line2D([], [], linestyle="", marker="", color="none"))
        labels.append(str(section_title))
        for entry in entries:
            kind = entry.get("kind", "patch")
            label = "  " + str(entry.get("label", ""))
            color = entry.get("color", "white")
            alpha = float(entry.get("alpha", 1.0))
            if kind == "point":
                handle = Line2D(
                    [0],
                    [0],
                    marker=entry.get("marker", "o"),
                    linestyle="",
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="none",
                    alpha=alpha,
                    markersize=6,
                )
            elif kind == "line":
                handle = Line2D([0], [0], color=color, alpha=alpha, linewidth=1.5)
            elif kind == "text":
                handle = Line2D([], [], linestyle="", marker="", color="none")
            elif kind == "gradient":
                handle = Patch(facecolor="0.7", edgecolor="0.7", alpha=1.0)
                label = "  " + str(entry.get("label", "")) + f" ({color})"
            else:
                handle = Patch(
                    facecolor=color,
                    edgecolor=entry.get("edgecolor", color),
                    alpha=alpha,
                )
            handles.append(handle)
            labels.append(label)

    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        title=legend_title,
        fontsize=fontsize,
        title_fontsize=fontsize,
        labelcolor="white",
        facecolor="black",
        edgecolor="black",
        framealpha=0.9,
        handlelength=1.2,
        handletextpad=0.6,
        borderpad=0.8,
    )
    if legend_loc == "outside right":
        legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, **legend_kwargs)
    elif legend_loc == "outside left":
        legend = ax.legend(loc="center right", bbox_to_anchor=(-0.02, 0.5), borderaxespad=0, **legend_kwargs)
    elif legend_loc == "outside bottom":
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), borderaxespad=0, ncol=1, **legend_kwargs)
    elif legend_loc == "outside top":
        legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0, ncol=1, **legend_kwargs)
    else:
        legend = ax.legend(loc=legend_loc, **legend_kwargs)

    section_label_indices = []
    idx = 0
    for _section_title, entries in sections:
        section_label_indices.append(idx)
        idx += 1 + len(entries)
    for i, text in enumerate(legend.get_texts()):
        if i in section_label_indices:
            text.set_weight("bold")
    return legend


def _is_option_dict(value, option_keys):
    return isinstance(value, dict) and bool(set(value) & option_keys)


def _normalize_single_image(spec):
    if spec is None or spec is False:
        return []
    if isinstance(spec, str):
        return [{"channel": spec}]
    if isinstance(spec, dict):
        if "channel" in spec:
            return [dict(spec)]
        return [{"channel": channel, **(opts if isinstance(opts, dict) else {})} for channel, opts in spec.items()]
    if isinstance(spec, (list, tuple)):
        out = []
        for item in spec:
            out.extend(_normalize_single_image(item))
        return out
    raise ValueError("image/images must be a channel name, list, or dictionary.")


def _normalize_images(image=None, images=None, level=None):
    specs = []
    specs.extend(_normalize_single_image(image))
    specs.extend(_normalize_single_image(images))
    if level is not None:
        for spec in specs:
            spec.setdefault("level", level)
    return specs


def _normalize_gene_layer(value, option_keys, default_key="genes"):
    if value is None or value is False:
        return None
    if _is_option_dict(value, option_keys):
        return dict(value)
    return {default_key: value}


def _resolve_bounds(xdata, bounds):
    if bounds is not None:
        return bounds
    roi = getattr(xdata, "active_roi", None)
    if roi is not None:
        return roi.bounds
    return (xdata.xmin, xdata.xmax, xdata.ymin, xdata.ymax)


def _color_cmap(color):
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        f"xentools_{str(color).replace('#', '')}",
        [(0, 0, 0), mcolors.to_rgb(color)],
    )


def _default_image_color(channel, index):
    name = str(channel).lower()
    if name == "dapi" or "dapi" in name:
        return "royalblue"
    cycle = ["lime", "magenta", "cyan", "orange", "red", "yellow"]
    return cycle[index % len(cycle)]


def _make_latest_image_signal_overlay(ax, *, alpha=0.7, color=None):
    """
    Convert the most recently added scalar image artist into RGBA with alpha
    proportional to signal. This approximates screen/additive behavior for
    dark-background fluorescence layers in Matplotlib.
    """
    if not ax.images:
        return

    artist = ax.images[-1]
    arr = np.ma.filled(np.asarray(artist.get_array()), 0)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgba = arr.astype(float, copy=True)
        if rgba.max() > 1:
            rgba = rgba / 255.0
        signal = rgba[..., :3].max(axis=-1)
        rgba[..., 3] = np.clip(signal * float(alpha), 0, 1)
        artist.set_data(rgba)
        return
    if arr.ndim == 3:
        rgb = arr.astype(float, copy=True)
        if rgb.max() > 1:
            rgb = rgb / 255.0
        signal = rgb.max(axis=-1)
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=float)
        rgba[..., :3] = rgb[..., :3]
        rgba[..., 3] = np.clip(signal * float(alpha), 0, 1)
        artist.set_data(rgba)
        return

    normed = artist.norm(arr)
    normed = np.nan_to_num(np.asarray(normed, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    normed = np.clip(normed, 0, 1)
    cmap = _color_cmap(color) if color is not None else artist.cmap
    rgba = cmap(normed)
    rgba[..., 3] = np.clip(normed * float(alpha), 0, 1)
    artist.set_data(rgba)


def _dim_latest_scalar_image(ax, alpha=1.0):
    """
    Dim a normal scalar background image without making black pixels gray.

    This preserves the image as an opaque background but raises the display
    vmax, which makes the rendered intensity darker in the same spirit as the
    legacy ``image_alpha`` behavior in ``plot_splat(image_channel=...)``.
    """
    if not ax.images or alpha is None or float(alpha) >= 1.0:
        return
    if float(alpha) < 0:
        raise ValueError("Image alpha must be >= 0.")

    artist = ax.images[-1]
    arr = np.asarray(artist.get_array())
    if arr.ndim != 2:
        return

    vmin, vmax = artist.get_clim()
    if float(alpha) == 0:
        vmax = np.inf
    else:
        vmax = vmax / float(alpha)
    artist.set_clim(vmin, vmax)


def render(
    xdata,
    *,
    image=None,
    images=None,
    splat=None,
    binned_splat=None,
    points=None,
    cells=None,
    bounds=None,
    ax=None,
    figsize=(8, 8),
    dpi: Optional[int] = None,
    level: Optional[int] = None,
    background: str = "black",
    show_axis: bool = False,
    title: Optional[str] = None,
    legend=True,
    legend_loc: str = "outside right",
    legend_title: Optional[str] = None,
    legend_max_items: int = 30,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Render a composite spatial view from common XenData layer types.

    This is the high-level plotting compositor used by
    :meth:`xentools.XenData.render`. It is designed for exploratory Xenium
    views where the user wants a DAPI/protein image, transcript-density splat,
    transcript points, and/or cell boundaries in one aligned plot without
    manually managing Matplotlib artist order.

    Layer order is fixed intentionally:

    1. Image channels are drawn first.
    2. Transcript splats are drawn over images with transparent zero-signal
       pixels.
    3. Transcript points are drawn over splats.
    4. Cell boundaries are drawn last.

    This avoids most call-order mistakes while still accepting an existing
    Matplotlib axes for advanced manual composition.

    Parameters
    ----------
    xdata : xentools.XenData
        Loaded Xenium data object. ``render`` calls the object's
        ``plot_image``, ``plot_splat``, ``plot_points``, and ``plot_cells``
        methods internally.
    image, images
        Image channel specification(s). Accepted forms:

        - ``image="DAPI"``
        - ``image={"channel": "DAPI", "level": 2, "alpha": 0.5}``
        - ``images=["DAPI", "18S"]``
        - ``images={"DAPI": {"level": 2}, "18S": {"color": "green", "alpha": 0.6}}``

        Image option keys include ``level``, ``cmap``, ``vmin``, ``vmax``,
        ``clip_percentile``, ``z_index``, ``alpha``, ``color``, and ``blend``.
        The first image is treated as the opaque background by default. Later
        images default to transparent signal overlays. For a background image,
        ``alpha < 1`` dims the display intensity without making black pixels
        gray.
    splat : str, sequence, dict, or None
        Transcript-density layer. Simple forms are passed as genes:

        - ``splat="EPCAM"``
        - ``splat=["C7", "Epcam", "Tagln"]``
        - ``splat={"R": ["C7"], "G": ["Epcam"], "B": ["Tagln"]}``

        To pass plotting options, provide a dictionary containing ``genes``.
        Common option keys include ``gains``, ``pixel_size_um``, ``sigma_um``,
        ``smooth``, ``global_norm``, ``splat_alpha``, and ``show_legend``.
    binned_splat : str, sequence, dict, or None
        Binned transcript-density layer from ``xdata.binned_adata``. This has
        the same simple gene and gene-set forms as ``splat`` but uses
        ``plot_binned_splat`` internally. Run ``xdata.create_binned_adata()``
        before using it. Common option keys include ``gains``, ``sigma_um``,
        ``smooth``, ``global_norm``, ``splat_alpha``, and ``show_legend``.
    points : str, sequence, dict, or None
        Transcript point layer. Simple gene forms are accepted, or provide a
        dictionary containing ``genes`` plus options for ``plot_points``.
        Common option keys include ``s``, ``alpha``, ``color``, ``palette``,
        ``marker``, ``markers``, ``max_points``, ``assigned_only``, and
        ``show_legend``.
    cells
        ``True`` to draw default cell boundaries, ``False``/``None`` to omit
        them, or a dictionary of ``plot_cells`` options. Common option keys
        include ``color_by``, ``genes``, ``face_alpha``, ``edge_alpha``,
        ``linewidth``, ``cmap``, and ``show_legend``.
    bounds
        Spatial window as ``(xmin, xmax, ymin, ymax)`` in microns. Defaults to
        the active ROI when present, otherwise the full transcript frame.
    ax
        Existing Matplotlib axes. When supplied, ``render`` draws into it.
    figsize, dpi
        Figure size and DPI used when ``ax`` is omitted.
    level
        Default image pyramid level applied to image layers that do not specify
        their own ``level``.
    background
        Figure and axes background color for new axes.
    show_axis
        Whether to show axis ticks and labels.
    title
        Optional axes title.
    legend : bool or str
        Whether to draw a single combined legend for the rendered layers.
        Default ``True``. String values ``"off"``, ``"false"``, ``"none"``,
        and ``"no"`` disable it. When enabled, lower-level categorical legends
        from splats, transcript points, and cells are suppressed so the figure
        has one coordinated legend.
    legend_loc : str
        Legend placement. Supported outside placements are ``"outside right"``,
        ``"outside left"``, ``"outside bottom"``, and ``"outside top"``.
        Other values are passed to Matplotlib as regular legend locations.
    legend_title : str or None
        Optional title for the combined legend.
    legend_max_items : int
        Maximum number of entries shown per categorical section before adding
        a compact ``"... N more"`` line.
    save
        Optional path to save the rendered figure.
    save_kwargs
        Optional dictionary forwarded to ``Figure.savefig``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the composite.

    Examples
    --------
    Basic DAPI + splat + points + cells:

    >>> ax = xentools.pl.render(
    ...     xdata,
    ...     image={"channel": "DAPI", "level": 2, "alpha": 0.5},
    ...     splat={"genes": ["C7", "Epcam", "Tagln"], "gains": [3, 3, 3]},
    ...     points={"genes": ["Prss3"], "s": 0.4},
    ...     cells=True,
    ...     bounds=(500, 1500, 900, 1900),
    ... )

    Multiple image channels:

    >>> ax = xdata.render(
    ...     images={
    ...         "DAPI": {"level": 2, "alpha": 0.5},
    ...         "18S": {"level": 2, "color": "green", "alpha": 0.6},
    ...     },
    ...     splat={"genes": ["C7", "Epcam", "Tagln"], "gains": [3, 3, 3]},
    ... )
    """
    import matplotlib.pyplot as plt

    bounds = _resolve_bounds(xdata, bounds)
    image_specs = _normalize_images(image=image, images=images, level=level)
    splat_opts = _normalize_gene_layer(splat, _SPLAT_OPTION_KEYS)
    binned_splat_opts = _normalize_gene_layer(binned_splat, _BINNED_SPLAT_OPTION_KEYS)
    point_opts = _normalize_gene_layer(points, _POINT_OPTION_KEYS)
    cell_opts = {} if cells is True else (dict(cells) if isinstance(cells, dict) else None)
    show_combined_legend = _normalize_legend_setting(legend)
    legend_sections = []

    if show_combined_legend:
        legend_sections = _collect_render_legend(
            xdata,
            image_specs=image_specs,
            splat_opts=splat_opts,
            binned_splat_opts=binned_splat_opts,
            point_opts=point_opts,
            cell_opts=cell_opts,
            bounds=bounds,
            legend_max_items=legend_max_items,
        )
        if splat_opts is not None:
            splat_opts = dict(splat_opts)
            splat_opts["show_legend"] = False
        if binned_splat_opts is not None:
            binned_splat_opts = dict(binned_splat_opts)
            binned_splat_opts["show_legend"] = False
        if point_opts is not None:
            point_opts = dict(point_opts)
            point_opts["show_legend"] = False
        if cell_opts is not None:
            cell_opts["show_legend"] = False

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

    for i, spec in enumerate(image_specs):
        spec = dict(spec)
        channel = spec.pop("channel")
        alpha = float(spec.pop("alpha", 0.7 if i else 1.0))
        color = spec.pop("color", None)
        blend = spec.pop("blend", "normal" if i == 0 else "screen")
        if color is not None and "cmap" not in spec:
            spec["cmap"] = _color_cmap(color)
        elif i > 0 and "cmap" not in spec:
            color = _default_image_color(channel, i - 1)
            spec["cmap"] = _color_cmap(color)

        xdata.plot_image(
            channel,
            bounds=bounds,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            verbose=False,
            **spec,
        )
        if i > 0 or blend in {"screen", "add", "additive", "transparent"}:
            _make_latest_image_signal_overlay(ax, alpha=alpha, color=color)
        else:
            _dim_latest_scalar_image(ax, alpha=alpha)

    if splat_opts is not None:
        splat_opts = dict(splat_opts)
        genes = splat_opts.pop("genes", None)
        splat_opts.setdefault("show_legend", False)
        splat_opts.setdefault("splat_alpha", 0.8)
        xdata.plot_splat(genes=genes, bounds=bounds, ax=ax, **splat_opts)

    if binned_splat_opts is not None:
        binned_splat_opts = dict(binned_splat_opts)
        genes = binned_splat_opts.pop("genes", None)
        binned_splat_opts.setdefault("show_legend", False)
        binned_splat_opts.setdefault("splat_alpha", 0.8)
        xdata.plot_binned_splat(genes=genes, bounds=bounds, ax=ax, **binned_splat_opts)

    if point_opts is not None:
        point_opts = dict(point_opts)
        genes = point_opts.pop("genes", None)
        xdata.plot_points(genes=genes, bounds=bounds, ax=ax, **point_opts)

    if cell_opts is not None:
        cell_opts.setdefault("bounds", bounds)
        cell_opts.setdefault("ax", ax)
        cell_opts.setdefault("facecolor", "none")
        cell_opts.setdefault("edge_alpha", 0.2)
        xdata.plot_cells(**cell_opts)

    if title is not None:
        ax.set_title(title)
    if not show_axis:
        ax.set_axis_off()
    ax.set_aspect("equal")
    if show_combined_legend:
        _draw_combined_legend(
            ax,
            legend_sections,
            legend_loc=legend_loc,
            legend_title=legend_title,
        )
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    return ax
