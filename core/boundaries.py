"""Boundary data model helpers for xentools."""

from __future__ import annotations

__all__ = ["LazyBoundaryGeoDataFrame"]


class LazyBoundaryGeoDataFrame:
    """
    Deferred boundary loader that materializes a GeoDataFrame on first access.
    """

    def __init__(self, loader, label: str = "boundaries"):
        self._loader = loader
        self._label = label
        self._data = None

    def _materialize(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    def __getattr__(self, name):
        return getattr(self._materialize(), name)

    def __getitem__(self, key):
        return self._materialize()[key]

    def __len__(self):
        return len(self._materialize())

    def __iter__(self):
        return iter(self._materialize())

    def __repr__(self):
        if self._data is None:
            return f"LazyBoundaryGeoDataFrame({self._label}, unloaded)"
        return repr(self._data)
