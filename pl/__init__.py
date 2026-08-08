"""
Public plotting namespace for xentools.

Import plotting functions here so users can access them as:

    xentools.pl.splat(...)
    xentools.pl.show_ome_tiff(...)
"""

from .boundaries import plot_boundaries, plot_cells
from .colors import generate_palette, plot_palette
from .images import show_aligned_image, show_ome_tiff
from .housekeeping import plot_housekeeping_diagnostics
from .niches import niche_heatmap, niche_map
from .render import render
from .transcripts import (
    bin_expression,
    create_binned_image,
    create_bins,
    create_multilayer_image,
    plot_binned_greyscale,
    plot_binned_rgb,
    plot_binned_splat,
    points,
    rasterize,
    rasterize_rgb,
    splat,
)

plot_image = show_ome_tiff
plot_points = points
plot_splat = splat

__all__ = [
    "bin_expression",
    "create_binned_image",
    "create_bins",
    "create_multilayer_image",
    "generate_palette",
    "niche_heatmap",
    "niche_map",
    "plot_binned_greyscale",
    "plot_housekeeping_diagnostics",
    "plot_binned_rgb",
    "plot_binned_splat",
    "plot_boundaries",
    "plot_cells",
    "plot_image",
    "plot_palette",
    "plot_points",
    "plot_splat",
    "points",
    "rasterize",
    "rasterize_rgb",
    "render",
    "show_ome_tiff",
    "show_aligned_image",
    "splat",
]
