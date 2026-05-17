# xentools

`xentools` is a lightweight analysis and visualization toolkit for 10x Xenium-style spatial transcriptomics outputs, with working support for both standard Xenium bundles and basic Atera whole-transcriptome Zarr bundles.

This snapshot is being frozen as a usable baseline for:
- loading Xenium and Atera datasets into a `XenData` object
- reading transcript, cell, boundary, image, and clustering data
- plotting morphology overlays, splats, and boundaries
- cropping data to ROIs, including GeoJSON-driven ROI workflows

## Current Status

Working:
- Xenium data loading and plotting
- Atera Zarr bundle loading
- Atera `splat()` and ROI-based plotting workflows
- lazy boundary registration for faster startup; cell/nucleus boundary polygons
  are materialized only when needed, and ROI-bounded boundary plots load only
  the relevant polygons when possible
- basic package-level helpers exposed through `xentools` and `xentools.io`

Pending / not release-ready:
- Xenium Explorer bundle export
- exact `transcripts.zarr.zip` compatibility with Xenium Explorer
- broader test coverage and packaging cleanup

The Xenium Explorer export code is present internally as an active work-in-progress, but it should be treated as experimental rather than a stable public API.

## Installation

Create an environment with the project dependencies, then install the package in editable mode if desired.

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

```python
import xentools

xd = xentools.XenData("/path/to/xenium_or_atera_output")
ax = xd.splat(["CCND1", "NHERF1", "OR4F17"])
arr = xd.splat(["CCND1", "NHERF1", "OR4F17"], return_array=True)
```

By default, cell and nucleus boundaries are lazy. This keeps initialization
fast for large Xenium outputs; plotting a bounded region loads only the
boundaries needed for that view when possible. Set `lazy_boundaries=False` if
you explicitly want all boundary polygons materialized at initialization.

Load and subset to a GeoJSON ROI at initialization:

```python
xd = xentools.XenData(
    "/path/to/xe_outs",
    roi_file="/path/to/roi.geojson",
    crop_to_selection=0,
)
```

Track ROIs without subsetting, then focus plots through the active ROI:

```python
xd = xentools.XenData(
    "/path/to/xe_outs",
    roi_file="/path/to/roi.geojson",
)
xd.set_active_roi(0)
ax = xd.splat(["CCND1", "NHERF1", "OR4F17"])
```

Create a real subset only when you want a reduced object:

```python
xsub = xd.subset_to_roi(0, inplace=False)
```

Overlay boundaries on DAPI:

```python
ax = xd.show_image("DAPI", figsize=(10, 10))
xd.plot_boundaries(kind="cell", color_by="Cluster", ax=ax)
```

Fill cell polygons by expression of one gene, or by summed expression of a gene
set:

```python
ax = xd.plot_cells(genes="EPCAM", cmap="magma")
ax = xd.plot_cells(genes=["EPCAM", "KRT19"], cmap="magma")
```

## Notes

- ROI files are commonly interpreted in pixel units and scaled internally by the dataset pixel size.
- Large Atera datasets can be loaded lazily through `transcripts.zarr.zip`; transcript access is tile-based rather than fully materialized by default.
- The repository currently contains internal code paths for Explorer export, but the public status of that feature should be considered pending.

## Repository Scope

This GitHub snapshot is intended to preserve the current working Atera-capable state of the codebase. Large local reference datasets, caches, editor settings, and scratch comparison notes are intentionally not part of the published repository state.
