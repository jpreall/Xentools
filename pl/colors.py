"""Color palette helpers for plotting."""

from __future__ import annotations

__all__ = ["generate_palette", "plot_palette"]

from ._save import save_figure


def generate_palette(n, lightness=0.5, sat_min=0.5, sat_max=1.0, preview=False):
    """
    Generate ``n`` visually distinct HEX colors in HLS space.
    """
    import colorsys
    import random

    palette = []
    hues = [i / n for i in range(n)]
    for h in hues:
        saturation = random.uniform(sat_min, sat_max)
        r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
        palette.append("#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255)))

    if preview:
        plot_palette(palette)

    return palette


def plot_palette(palette, save=None, save_kwargs=None):
    """
    Plot a palette of colors as a horizontal strip.
    """
    import matplotlib.pyplot as plt

    n = len(palette)
    fig, ax = plt.subplots(figsize=(n, 2))
    del fig
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save_figure(ax, save=save, save_kwargs=save_kwargs)
    plt.show()
    return ax
