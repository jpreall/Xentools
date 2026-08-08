"""Lightweight coordinate systems and homogeneous transform registry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CoordinateSystem", "TransformRegistry"]


@dataclass(frozen=True)
class CoordinateSystem:
    """Named 2D coordinate system with explicit axes, units, and orientation."""

    name: str
    axes: tuple[str, str] = ("x", "y")
    units: tuple[str, str] = ("micrometer", "micrometer")
    y_direction: str = "down"

    def __post_init__(self):
        if self.y_direction not in {"down", "up"}:
            raise ValueError("y_direction must be 'down' or 'up'.")


class TransformRegistry:
    """Map spatial elements from intrinsic coordinates into named systems."""

    def __init__(self, global_system=None):
        self.systems = {}
        self._transforms = {}
        self.add_system(global_system or CoordinateSystem("global"))

    def add_system(self, system):
        if not isinstance(system, CoordinateSystem):
            raise TypeError("system must be a CoordinateSystem.")
        self.systems[system.name] = system
        return system

    def register(self, element, matrix, *, source=None, target="global", metadata=None, overwrite=False):
        """Register a finite, invertible 3×3 transform for one element."""
        element = str(element)
        if target not in self.systems:
            raise KeyError(f"Unknown target coordinate system {target!r}.")
        if element in self._transforms and not overwrite:
            raise ValueError(f"Transform for {element!r} already exists; pass overwrite=True.")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("matrix must be a finite 3×3 homogeneous transform.")
        if not np.allclose(matrix[2], [0, 0, 1], atol=1e-10):
            raise ValueError("matrix last row must be [0, 0, 1].")
        if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
            raise ValueError("matrix must be invertible.")
        self._transforms[element] = {
            "source": source or element,
            "target": target,
            "matrix": matrix,
            "metadata": {} if metadata is None else dict(metadata),
        }
        return matrix

    def get(self, element, *, inverse=False):
        """Return an element-to-target transform, or its inverse."""
        if element not in self._transforms:
            raise KeyError(f"No transform registered for {element!r}.")
        matrix = self._transforms[element]["matrix"]
        return np.linalg.inv(matrix) if inverse else matrix.copy()

    def apply(self, element, coordinates, *, inverse=False):
        """Apply a registered transform to an ``(n, 2)`` coordinate array."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2).")
        homogeneous = np.column_stack([coordinates, np.ones(len(coordinates))])
        mapped = (self.get(element, inverse=inverse) @ homogeneous.T).T
        return mapped[:, :2] / mapped[:, 2, None]

    def info(self, element):
        if element not in self._transforms:
            raise KeyError(element)
        item = self._transforms[element]
        return {**item, "matrix": item["matrix"].copy()}

    def __contains__(self, element):
        return element in self._transforms

    def __repr__(self):
        global_system = self.systems["global"]
        lines = [
            "TransformRegistry",
            (
                f"  global: axes={global_system.axes}, units={global_system.units}, "
                f"y_direction={global_system.y_direction!r}"
            ),
            f"  {len(self._transforms)} registered elements",
        ]
        for element, item in self._transforms.items():
            lines.append(f"  {element!r}: {item['source']} → {item['target']}")
        return "\n".join(lines)
