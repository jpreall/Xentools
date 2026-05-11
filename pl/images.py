from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as pl
import numpy as np

from ._shared import _load_local_module

try:
    from ..io.read.images import _source_series_level_arrays
    from ..io.write.images import _roi_bounds_in_pixels, _scale_bounds_for_level
except ImportError:
    _images_read_mod = _load_local_module("_xentools_io_read_images_for_pl", "io/read/images.py")
    _images_write_mod = _load_local_module("_xentools_io_write_images_for_pl", "io/write/images.py")

    _source_series_level_arrays = _images_read_mod._source_series_level_arrays
    _roi_bounds_in_pixels = _images_write_mod._roi_bounds_in_pixels
    _scale_bounds_for_level = _images_write_mod._scale_bounds_for_level


__all__ = ["show_ome_tiff"]


def show_ome_tiff(
    image_path,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip_percentile: float = 99.5,
    z_index: Optional[int] = None,
    pixel_size: float = 0.2125,
    roi=None,
    level: Optional[int] = None,
    micron_coords: bool = False,
    ax=None,
    verbose: bool = True,
):
    import tifffile

    if dpi is None:
        dpi = pl.rcParams["figure.dpi"]
    if figsize is None:
        figsize = pl.rcParams["figure.figsize"]

    display_px = max(int(figsize[0] * dpi), int(figsize[1] * dpi))

    with tifffile.TiffFile(image_path) as tif:
        page0 = tif.pages[0]
        subifd_pages = list(page0.pages)
        try:
            series = tif.series[0]
            level_arrays, zarr_store = _source_series_level_arrays(series)
            use_zarr_levels = True
        except Exception as exc:
            message = str(exc)
            if "multi-file pyramids" not in message:
                raise
            use_zarr_levels = False
            zarr_store = None
            level_arrays = [page0] + subifd_pages

        if use_zarr_levels and len(level_arrays) == 1 and len(subifd_pages) > 0:
            if zarr_store is not None:
                zarr_store.close()
                zarr_store = None
            use_zarr_levels = False
            level_arrays = [page0] + subifd_pages

        try:
            full_shape = level_arrays[0].shape

            if roi is not None:
                base_bounds = _roi_bounds_in_pixels(roi, pixel_size, full_shape)
                source_px = max(
                    base_bounds[1] - base_bounds[0],
                    base_bounds[3] - base_bounds[2],
                )
            else:
                base_bounds = None
                source_px = max(full_shape[-2:])

            if level is not None:
                best_level = min(level, len(level_arrays) - 1)
            else:
                best_level = 0
                for i in range(len(level_arrays)):
                    if source_px // (2**i) >= display_px:
                        best_level = i

            level_array = level_arrays[best_level]
            level_shape = level_array.shape

            if verbose:
                mode = "zarr" if use_zarr_levels else "tiff-subifd"
                print(
                    f"[show_ome_tiff] pyramid level {best_level}/{len(level_arrays)-1}  "
                    f"({level_shape[-2]} × {level_shape[-1]} px) [{mode}]"
                )

            if base_bounds is not None:
                lx0, lx1, ly0, ly1 = _scale_bounds_for_level(base_bounds, best_level)
                ly0 = max(0, ly0)
                ly1 = min(level_shape[-2], ly1)
                lx0 = max(0, lx0)
                lx1 = min(level_shape[-1], lx1)
                if use_zarr_levels:
                    if level_array.ndim == 2:
                        img = np.asarray(level_array[ly0:ly1, lx0:lx1])
                    elif z_index is None:
                        img = np.asarray(level_array[:, ly0:ly1, lx0:lx1]).max(axis=0)
                    else:
                        img = np.asarray(level_array[z_index, ly0:ly1, lx0:lx1])
                else:
                    arr = level_array.asarray()
                    if arr.ndim == 2:
                        img = np.asarray(arr[ly0:ly1, lx0:lx1])
                    elif z_index is None:
                        img = np.asarray(arr[:, ly0:ly1, lx0:lx1]).max(axis=0)
                    else:
                        img = np.asarray(arr[z_index, ly0:ly1, lx0:lx1])
            else:
                if use_zarr_levels:
                    if level_array.ndim == 2:
                        img = np.asarray(level_array)
                    elif z_index is None:
                        img = np.asarray(level_array).max(axis=0)
                    else:
                        img = np.asarray(level_array[z_index])
                else:
                    arr = level_array.asarray()
                    if arr.ndim == 2:
                        img = np.asarray(arr)
                    elif z_index is None:
                        img = np.asarray(arr).max(axis=0)
                    else:
                        img = np.asarray(arr[z_index])
        finally:
            if zarr_store is not None:
                zarr_store.close()

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = float(np.percentile(img, clip_percentile))

    own_fig = ax is None
    if own_fig:
        _, ax = pl.subplots(figsize=figsize, dpi=dpi)

    if micron_coords:
        if base_bounds is not None:
            im_extent = [
                base_bounds[0] * pixel_size,
                base_bounds[1] * pixel_size,
                base_bounds[2] * pixel_size,
                base_bounds[3] * pixel_size,
            ]
        else:
            im_extent = [0.0, full_shape[-1] * pixel_size, 0.0, full_shape[-2] * pixel_size]
        img = img[::-1]
        imshow_kwargs = dict(origin="lower", extent=im_extent)
    else:
        imshow_kwargs = {}

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", **imshow_kwargs)
    ax.axis("off")

    if own_fig:
        pl.tight_layout()

    return ax
