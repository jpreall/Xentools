# Boundary Loading Performance Plan

## Current Bottleneck

`XenData.__init__()` currently materializes cell and nucleus boundaries during object construction when boundary parquet files are available.

The parquet path does two expensive things immediately:

1. Reads all boundary vertex rows from disk.
2. Groups by `cell_id` and constructs one Shapely polygon per cell or nucleus in Python.

This makes initialization sensitive to disk throughput. Loading from an internal NVMe SSD is noticeably faster than loading from an external USB SSD because the full boundary files are read and converted up front.

## Current Code Path

Boundary selection happens in `XenData.__init__()`:

```python
self.cell_boundaries = import_segmentation_xenium_parquet(cell_boundaries_file)
self.nucleus_boundaries = import_segmentation_xenium_parquet(nuc_boundaries_file)
```

The reader currently does:

```python
boundaries_df = pd.read_parquet(boundaries_file)
boundaries_df.groupby("cell_id").apply(create_polygon)
```

So the startup cost is both I/O-heavy and Python object-construction-heavy.

## Recommended Short-Term Fix

Add lazy parquet boundary loading.

`lazy_boundaries` currently applies only to `cells.zarr.zip`. The same deferred-loading pattern should be extended to `cell_boundaries.parquet` and `nucleus_boundaries.parquet`.

Desired behavior:

```python
xdata = xentools.XenData(folder, lazy_boundaries=True)
```

This should register lazy loaders for both parquet and Zarr boundary sources. Boundaries would only materialize when accessed, plotted, subsetted, or exported.

## Longer-Term Options

### Converted GeoParquet Cache

Create a one-time converted geometry cache:

```text
cell_boundaries.geoparquet
nucleus_boundaries.geoparquet
```

This would store actual polygon geometries instead of vertex rows, avoiding repeated `groupby(...).apply(Polygon)` conversion on later loads.

### ROI-Aware Partial Loading

If an ROI is known early, avoid constructing all polygons. Use cell centroids from `cells.parquet`, `adata.obs`, or `cells.zarr.zip` summaries to identify candidate cells, then construct only the required boundary polygons.

### Faster Polygon Construction

Benchmark Shapely 2 vectorized/ragged polygon construction against the current pandas `groupby().apply(create_polygon)` path.

## Suggested Implementation Order

1. Extend `LazyBoundaryGeoDataFrame` to support parquet boundary loaders.
2. Make `lazy_boundaries=True` apply to both parquet and Zarr sources.
3. Ensure `__repr__` reports boundary availability without forcing materialization.
4. Add an optional GeoParquet cache path for repeated workflows.
5. Explore ROI-aware partial loading for very large Atera-scale datasets.

## Open Questions

- Should lazy boundaries become the default for large datasets?
- Should the package auto-create a converted GeoParquet cache after first load?
- Where should a converted cache live: inside the Xenium output folder or an `xentools` sidecar folder?
- Should plotting functions materialize full boundaries or support viewport/ROI-aware boundary loading?
