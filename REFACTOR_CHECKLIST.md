# xentools Refactor Checklist

## Goal

Break the monolithic `xentools.py` file into smaller, logically grouped modules
without breaking the current public API during the transition.

Primary constraints:

- keep `XenData`, `LazyTranscripts`, ROI classes, and top-level helpers importable
  as users expect today
- avoid a large, risky one-shot rewrite
- keep behavior stable while steadily shrinking `xentools.py`

## Desired end state

Proposed package structure:

```text
xentools/
  __init__.py
  core/
    xendata.py
    transcripts.py
    rois.py
    boundaries.py
  io/
    read/
      xenium.py
      zarr.py
      images.py
    write/
      xenium.py
      zarr.py
      spatialdata.py
  plotting/
    images.py
    transcript_rasters.py
    boundaries.py
  analysis/
    binning.py
    graph.py
    niches.py
    normalization.py
  utils/
    colors.py
    geometry.py
    metadata.py
```

This exact layout can change, but the key principle should stay the same:

- `core`: object model / orchestration
- `io`: file readers/writers and discovery
- `analysis`: data transformations and computations
- `plotting`: display helpers
- `utils`: low-level reusable helpers

## Refactor rules

### 1. Keep `xentools.py` as a compatibility facade during migration

Do not remove it immediately.

Instead:

- move implementation into new modules
- re-import / re-export old public names from `xentools.py`
- preserve user-facing imports as long as possible

### 2. Move by domain, not by arbitrary line chunks

Bad split:

- “move lines 1–1000 to module A”

Good split:

- “move all ROI-related logic to `core/rois.py`”

### 3. Avoid circular imports aggressively

In particular:

- `core.xendata` will be the most import-sensitive module
- `plotting` should not be imported by lower-level modules
- `io` and `analysis` should generally not depend on `XenData`

### 4. Prefer helper functions that work on plain objects

When possible:

- accept arrays / DataFrames / AnnData / file paths
- avoid requiring full `XenData` inside lower-level modules

This keeps modules testable and reusable.

## Phase 0: Prep work

### Checklist

- [x] Create a real package layout if not already present
- [x] Decide whether `xentools.py` remains the compatibility entrypoint or becomes thin and imports from package modules
- [x] Add a simple smoke test plan for import stability
- [x] Identify any currently user-facing imports that must remain stable

### Deliverable

- A committed module skeleton and a statement of compatibility expectations

## Phase 1: Move ROI logic

This is the safest high-value first move because ROI code is already fairly self-contained.

### Move to `core/rois.py`

- [x] `ROI`
- [x] `ROIClass`
- [x] `ROICollection`
- [x] `read_ROI_from_csv`
- [x] `read_ROI_from_geojson`
- [x] `_coerce_roi_geometry`
- [x] `_coerce_roi`
- [x] `_geometry_to_roi_points`
- [x] `_normalize_roi_property`
- [x] `_geojson_class_name`
- [x] `_geojson_selection_name`
- [x] `_make_roi_name`
- [x] `_register_roi_record`
- [x] `_select_geojson_features`
- [x] `_match_geojson_row`
- [x] `_import_roi_records`
- [x] `_resolve_roi_selection_selector`
- [x] `_roi_bounds_um`
- [x] `_shapely_bounds`

### Follow-up

- [x] Update `XenData` to import ROI pieces from the new module
- [x] Re-export public ROI classes from `xentools.py`

### Deliverable

- ROI code removed from the monolith
- no user-facing API break

## Phase 2: Move transcript storage and format selection

This is the next major seam because `LazyTranscripts` is conceptually distinct.

### Move to `core/transcripts.py`

- [x] `LazyTranscripts`
- [x] `_detect_transcripts_format`
- [x] `_count_transcripts_in_bundle`
- [x] `_load_zarr_gene_names`
- [x] `_normalize_feature_selection`
- [x] `_bin_transcript_dataframe`

### Consider also moving here

- [x] `_encode_xenium_cell_ids`

Keep here only if it is mainly transcript/cell-zarr related; otherwise move it to
`utils/metadata.py` or `io/read/zarr.py`.

### Follow-up

- [x] Update `XenData` imports
- [x] Ensure `create_binned_adata()` still works
- [x] Re-export `LazyTranscripts` from `xentools.py`

### Deliverable

- transcript storage logic separated from `XenData`

## Phase 3: Move Zarr readers

These are clear `io` concerns.

### Move to `io/read/zarr.py`

- [x] `_open_zarr_group_compat`
- [x] `_read_zarr_adata`
- [x] `_read_analysis_zarr`

### Decide where these belong

- [x] `_load_zarr_gene_names` if not already moved to `core/transcripts.py`
- [x] `_encode_xenium_cell_ids` if still shared here

Resolved as:

- `_load_zarr_gene_names` stays in `core/transcripts.py`
- `_encode_xenium_cell_ids` moved to `utils/metadata.py` so both `core` and `io`
  can import it without a cycle

### Follow-up

- [x] Update `XenData` initialization to call the new reader module
- [x] Check import cycles with `LazyTranscripts`

### Deliverable

- all Zarr read-path details isolated from the main object class

## Phase 4: Move boundary loading and conversion

Boundary code now exists in several places and should be centralized.

### Move to `core/boundaries.py` or `io/read/xenium.py`

Recommended split:

- `core/boundaries.py`
  - [x] `LazyBoundaryGeoDataFrame`
- `io/read/xenium.py`
  - [x] `create_polygon`
  - [x] `import_segmentation_xenium_parquet`
  - [x] `import_segmentation_xenium_zarr`
- `io/write/xenium.py`
  - [x] `_geometry_to_boundary_rows`
  - [x] `_prepare_boundary_dataframe`

### Follow-up

- [x] Update `XenData.__init__`
- [x] Update any export helpers using boundaries

### Deliverable

- boundary data source handling becomes explicit and centralized

## Phase 4.5: IO hierarchy

Separate read and write concerns so future targets like SpatialData do not
accumulate in a flat `io` namespace.

### Layout

- [x] Create `io/read/`
- [x] Create `io/write/`
- [x] Move canonical read implementations under `io/read/`
- [x] Keep top-level `io/*.py` shims for compatibility
- [x] Scaffold `io/write/xenium.py`
- [x] Add first real writer implementation under `io/write/`

### Deliverable

- canonical IO layout reflects read/write separation
- compatibility preserved for older imports

## Phase 5: Move image IO and image export helpers

This is a big section and a strong candidate for modularization.

### Move to `io/read/images.py`

- [x] `_parse_ome_xml`
- [x] `_detect_linked_protein_images`
- [x] `_image_extent_um`
- [x] `_source_series_level_arrays`

### Move to `io/write/images.py`

- [x] `_build_pyramid_levels`
- [x] `_write_pyramidal_ome_tiff`
- [x] `_write_pyramidal_ome_tiff_from_levels`
- [x] `_linked_ome_xml`
- [x] `_write_linked_pyramidal_ome_tiffs`
- [x] `_write_linked_pyramidal_ome_tiffs_from_levels`
- [x] `_write_morphology_for_slice`
- [x] `_write_protein_images_for_slice`
- [x] `_cropped_shape_from_bounds`
- [x] `_iter_cropped_tiles`
- [x] `_scale_bounds_for_level`
- [x] `_roi_bounds_in_pixels`
- [x] `_pixel_aligned_bounds_um`
- [x] `_write_ome_tiff`

### Follow-up

- [x] Update `XenData.write_xenium_explorer()`
- [x] Update `XenData.write_geo_submission()`
- [ ] Ensure protein pyramid fallback still works

### Deliverable

- image IO/export code leaves the monolith

## Phase 6: Move plotting helpers

The file currently mixes plotting, IO, and analysis too heavily.

User-facing direction:

- prefer a flat top-level plotting namespace: `xentools.pl`
- canonical plotting implementations should live in the `pl/` package
- object methods like `XenData.splat()` should remain as convenience wrappers

### Move to `pl/`

- [x] `show_ome_tiff`
- [x] `create_bins`
- [x] `bin_expression`
- [x] `create_binned_image`
- [x] `create_multilayer_image`
- [x] `rasterize_rgb`
- [x] `plot_binned_rgb`
- [x] `plot_binned_greyscale`
- [x] `splat`
- [x] expose `xentools.pl`

### Follow-up

- [x] Keep `XenData.show_image()` and `XenData.splat()` as thin wrappers over shared plotting functions
- [x] Make `XenData.plot_boundaries()` a thin wrapper over `pl.plot_boundaries()`
- [x] Decide whether `plot_boundaries()` should become a full standalone renderer or stay object-oriented

Resolved as:

- canonical implementation is `xentools.pl.plot_boundaries(xdata, ...)`
- `XenData.plot_boundaries()` remains a convenience wrapper
- the plotting function accepts an xdata-like object rather than requiring boundary arrays directly for now, because it still needs coordinated access to boundaries, `adata.obs`, and `active_roi`

### Deliverable

- plotting logic separated from object model and IO

## Phase 7: Move analysis helpers

### Move to `analysis/normalization.py`

- [x] consolidate `to_TP10k` / `TP10K` into `normalize_tp10k`
- [x] remove ambiguous legacy normalization helpers

### Move to `analysis/binning.py`

- [x] any residual binning helpers not already moved
- [x] potentially shared logic used by `create_binned_adata()`

Resolved as:

- `_bin_transcript_dataframe` moved to `analysis/binning.py`
- `core.transcripts` imports/re-exports it for `LazyTranscripts.to_xarray_bins()`
- `XenData.create_binned_adata()` is now a thin wrapper over `analysis.binning.create_binned_adata()`
- stale duplicate `create_bins`, `bin_expression`, and `create_binned_image` definitions removed from `xentools.py`; canonical implementations live in `pl/transcripts.py`

### Move to `analysis/graph.py`

- [x] `_apply_weights`
- [x] `build_spatial_graph`

### Move to `analysis/niches.py`

- [x] `build_niches`
- [x] `evaluate_niche_k_values`

### Deliverable

- actual computations separate from file and plotting concerns

## Phase 8: Slim down `XenData`

At this point, `XenData` should mostly orchestrate rather than implement.

### Target responsibilities for `XenData`

Keep in `core/xendata.py`:

- [ ] initialization and source selection
- [ ] object state
- [ ] thin wrapper methods calling lower-level modules
- [ ] summary / repr methods
- [ ] user-facing convenience behavior

Progress:

- [x] `XenData.write_xenium_explorer()` is now a thin wrapper over `io.write.xenium.write_xenium_explorer()`
- [x] `XenData.write_geo_submission()` is now a thin wrapper over `io.write.xenium.write_geo_submission()`
- [x] export coordinate rebasing helpers moved with writer orchestration
- [x] `XenData.create_binned_adata()` is now a thin wrapper over `analysis.binning.create_binned_adata()`
- [x] `XenData.rasterize()` is now a thin wrapper over `pl.rasterize()`
- [x] `XenData.write_ome_tiff()` is now a thin wrapper over `io.write.images.write_ome_tiff()`
- [x] `experiment.xenium` parsing and image discovery moved to `io.read.metadata`
- [x] transcript source selection/loading moved to `io.read.transcripts`
- [x] boundary source selection/loading moved to `io.read.boundaries`
- [x] cell matrix, clusters, and gene panel loading moved to `io.read.cells`
- [x] reader orchestration moved to `io.read.loader.load_xenium_folder()`

### Responsibilities to remove from `XenData`

- [ ] direct file parsing
- [x] heavy binning implementation
- [x] heavy rasterization implementation
- [x] image-level TIFF details
- [ ] ROI parsing internals

### Deliverable

- `XenData` becomes readable and maintainable

## Phase 9: Public API cleanup

After the implementation has stabilized:

- [ ] decide what should be exported from package root
- [ ] decide whether some old helpers should become private
- [ ] deprecate accidental public symbols if needed
- [ ] fix stale names and typos such as mismatched `__all__` entries

### Deliverable

- coherent public surface area

## Recommended order of execution

Suggested practical order:

1. ROI module
2. transcript module
3. Zarr reader module
4. boundary module
5. image IO module
6. plotting modules
7. analysis modules
8. slim `XenData`
9. API cleanup

This order reduces risk because it starts with the most self-contained pieces and
delays the most entangled refactors until better structure already exists.

## Testing checklist for each phase

For every module move:

- [ ] `conda run -n xentools python -m compileall ...`
- [ ] import smoke test for `xentools`
- [ ] construct a `XenData` object on test data
- [ ] test ROI import / subset if ROI code moved
- [ ] test `show_image()` if image code moved
- [ ] test `create_binned_adata()` if transcript/binning code moved
- [ ] test `write_xenium_explorer()` if IO/export code moved

## Good stopping points

Useful intermediate milestones:

### Milestone A

- ROI + transcript modules extracted
- `xentools.py` substantially smaller

### Milestone B

- Zarr readers + boundaries extracted
- initialization logic clearer

### Milestone C

- plotting and image IO extracted
- `XenData` mostly wrappers

### Milestone D

- monolith reduced to compatibility facade or retired entirely

## Immediate next step

Best first PR:

- create `core/rois.py`
- move all ROI code there
- update `XenData` and top-level exports

This gives a meaningful reduction in file size with relatively low risk and creates
the basic pattern for the remaining refactor.
