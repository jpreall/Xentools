from __future__ import annotations

from typing import Optional

import numpy as np


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
    point_opts = _normalize_gene_layer(points, _POINT_OPTION_KEYS)
    cell_opts = {} if cells is True else (dict(cells) if isinstance(cells, dict) else None)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)

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
    return ax
