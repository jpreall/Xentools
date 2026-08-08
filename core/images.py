"""Transform-aware external image metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["AlignedImage"]


@dataclass(repr=False)
class AlignedImage:
    """External OME-TIFF registered to the Xenium morphology pixel frame."""

    name: str
    path: str
    source_to_fixed: np.ndarray
    source_shape: tuple[int, int]
    source_axes: str
    source_pixel_size: tuple[float | None, float | None]
    fixed_shape: tuple[int, int]
    fixed_pixel_size: float
    alignment_file: str | None = None
    channel_names: list[str] = field(default_factory=list)
    keypoint_count: int = 0
    keypoint_rmse_px: float | None = None
    keypoint_max_error_px: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.path = str(Path(self.path).expanduser().resolve())
        self.source_to_fixed = np.asarray(self.source_to_fixed, dtype=float)
        if self.source_to_fixed.shape != (3, 3):
            raise ValueError("source_to_fixed must be a 3×3 homogeneous transform.")

    @property
    def source_to_global(self):
        """Affine mapping source pixels to platform-neutral global microns."""
        pixel_size = float(self.fixed_pixel_size)
        fixed_to_global = np.array(
            [[pixel_size, 0, 0], [0, pixel_size, 0], [0, 0, 1]],
            dtype=float,
        )
        return fixed_to_global @ self.source_to_fixed

    @property
    def micron_bounds(self):
        """Axis-aligned display bounds of the transformed source image."""
        height, width = self.source_shape
        corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).T
        mapped = self.source_to_global @ corners
        mapped = mapped[:2] / mapped[2]
        return (
            float(mapped[0].min()),
            float(mapped[0].max()),
            float(mapped[1].min()),
            float(mapped[1].max()),
        )

    def __repr__(self):
        height, width = self.source_shape
        quality = "not validated"
        if self.keypoint_rmse_px is not None:
            quality = f"RMSE {self.keypoint_rmse_px:.3g} px ({self.keypoint_count} points)"
        return (
            f"AlignedImage(name={self.name!r}, shape={height}×{width}, "
            f"axes={self.source_axes!r}, alignment={quality})"
        )
