"""Geometry and coordinate utility helpers."""

from __future__ import annotations

import numpy as np

__all__ = ["frame", "ROI_to_pixels", "um_to_pixels"]


def ROI_to_pixels(ROI, pixel_size):
    """
    Return integer pixel bounds for a legacy ROI coordinate array.
    """
    xmin, xmax = int(ROI[:, 0].min() / pixel_size), int(ROI[:, 0].max() / pixel_size)
    ymin, ymax = int(ROI[:, 1].min() / pixel_size), int(ROI[:, 1].max() / pixel_size)
    return xmin, xmax, ymin, ymax


def um_to_pixels(arr, pixel_size: float = 0.2125) -> np.ndarray:
    """
    Convert array-like numerical input from microns to pixels.
    """
    arr = np.asarray(arr, dtype=float)
    return np.round(arr / pixel_size).astype(int)


def frame(transcripts_df):
    """
    Return transcript coordinate bounds as ``[[xmin, xmax], [ymin, ymax]]``.
    """
    xmin, xmax = transcripts_df["x_location"].min(), transcripts_df["x_location"].max()
    ymin, ymax = transcripts_df["y_location"].min(), transcripts_df["y_location"].max()
    return np.array([[xmin, xmax], [ymin, ymax]])
