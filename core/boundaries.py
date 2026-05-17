"""Boundary data model helpers for xentools."""

from __future__ import annotations

__all__ = ["LazyBoundaryGeoDataFrame"]


class LazyBoundaryGeoDataFrame:
    """
    Deferred boundary loader that materializes a GeoDataFrame on first access.
    """

    def __init__(self, loader, label: str = "boundaries", subset_loader=None, id_subset_loader=None):
        self._loader = loader
        self._subset_loader = subset_loader
        self._id_subset_loader = id_subset_loader
        self._label = label
        self._data = None

    def _materialize(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    @property
    def loaded(self) -> bool:
        """Whether the full boundary GeoDataFrame has been materialized."""
        return self._data is not None

    def query_bounds(self, bounds):
        """
        Return boundaries intersecting/queryable within micron bounds.

        This does not materialize the full dataset when a bounded loader is
        available. If the full GeoDataFrame was already loaded, or no bounded
        loader was supplied, filtering falls back to the materialized object.
        """
        if bounds is None:
            return self._materialize()
        if self._data is None and self._subset_loader is not None:
            return self._subset_loader(bounds)

        gdf = self._materialize()
        xmin, xmax, ymin, ymax = bounds
        cx = gdf.geometry.centroid.x
        cy = gdf.geometry.centroid.y
        return gdf[(cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)]

    def query_ids(self, ids):
        """
        Return boundaries for specific cell IDs without full materialization
        when the backing loader supports ID predicates.
        """
        ids = [str(i) for i in ids]
        if self._data is None and self._id_subset_loader is not None:
            return self._id_subset_loader(ids)

        gdf = self._materialize()
        keep = [i for i in ids if i in gdf.index]
        return gdf.loc[keep]

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
