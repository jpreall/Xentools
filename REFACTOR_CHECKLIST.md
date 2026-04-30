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
    xenium.py
    zarr_read.py
    zarr_write.py
    images.py
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

- [ ] Create a real package layout if not already present
- [ ] Decide whether `xentools.py` remains the compatibility entrypoint or becomes thin and imports from package modules
- [ ] Add a simple smoke test plan for import stability
- [ ] Identify any currently user-facing imports that must remain stable

### Deliverable

- A committed module skeleton and a statement of compatibility expectations

## Phase 1: Move ROI logic

This is the safest high-value first move because ROI code is already fairly self-contained.

### Move to `core/rois.py`

- [ ] `ROI`
- [ ] `ROIClass`
- [ ] `ROICollection`
- [ ] `read_ROI_from_csv`
- [ ] `read_ROI_from_geojson`
- [ ] `_coerce_roi_geometry`
- [ ] `_coerce_roi`
- [ ] `_geometry_to_roi_points`
- [ ] `_normalize_roi_property`
- [ ] `_geojson_class_name`
- [ ] `_geojson_selection_name`
- [ ] `_make_roi_name`
- [ ] `_register_roi_record`
- [ ] `_select_geojson_features`
- [ ] `_match_geojson_row`
- [ ] `_import_roi_records`
- [ ] `_resolve_roi_selection_selector`
- [ ] `_roi_bounds_um`
- [ ] `_shapely_bounds`

### Follow-up

- [ ] Update `XenData` to import ROI pieces from the new module
- [ ] Re-export public ROI classes from `xentools.py`

### Deliverable

- ROI code removed from the monolith
- no user-facing API break

## Phase 2: Move transcript storage and format selection

This is the next major seam because `LazyTranscripts` is conceptually distinct.

### Move to `core/transcripts.py`

- [ ] `LazyTranscripts`
- [ ] `_detect_transcripts_format`
- [ ] `_count_transcripts_in_bundle`
- [ ] `_load_zarr_gene_names`
- [ ] `_normalize_feature_selection`
- [ ] `_bin_transcript_dataframe`

### Consider also moving here

- [ ] `_encode_xenium_cell_ids`

Keep here only if it is mainly transcript/cell-zarr related; otherwise move it to
`utils/metadata.py` or `io/zarr_read.py`.

### Follow-up

- [ ] Update `XenData` imports
- [ ] Ensure `create_binned_adata()` still works
- [ ] Re-export `LazyTranscripts` from `xentools.py`

### Deliverable

- transcript storage logic separated from `XenData`

## Phase 3: Move Zarr readers

These are clear `io` concerns.

### Move to `io/zarr_read.py`

- [ ] `_open_zarr_group_compat`
- [ ] `_read_zarr_adata`
- [ ] `_read_analysis_zarr`

### Decide where these belong

- [ ] `_load_zarr_gene_names` if not already moved to `core/transcripts.py`
- [ ] `_encode_xenium_cell_ids` if still shared here

### Follow-up

- [ ] Update `XenData` initialization to call the new reader module
- [ ] Check import cycles with `LazyTranscripts`

### Deliverable

- all Zarr read-path details isolated from the main object class

## Phase 4: Move boundary loading and conversion

Boundary code now exists in several places and should be centralized.

### Move to `core/boundaries.py` or `io/xenium.py`

Recommended split:

- `core/boundaries.py`
  - [ ] `LazyBoundaryGeoDataFrame`
- `io/xenium.py` or `io/zarr_read.py`
  - [ ] `create_polygon`
  - [ ] `import_segmentation_xenium_parquet`
  - [ ] `import_segmentation_xenium_zarr`
  - [ ] `_geometry_to_boundary_rows`
  - [ ] `_prepare_boundary_dataframe`

### Follow-up

- [ ] Update `XenData.__init__`
- [ ] Update any export helpers using boundaries

### Deliverable

- boundary data source handling becomes explicit and centralized

## Phase 5: Move image IO and image export helpers

This is a big section and a strong candidate for modularization.

### Move to `io/images.py`

- [ ] `_parse_ome_xml`
- [ ] `_detect_linked_protein_images`
- [ ] `_image_extent_um`
- [ ] `_source_series_level_arrays`
- [ ] `_build_pyramid_levels`
- [ ] `_write_pyramidal_ome_tiff`
- [ ] `_write_pyramidal_ome_tiff_from_levels`
- [ ] `_linked_ome_xml`
- [ ] `_write_linked_pyramidal_ome_tiffs`
- [ ] `_write_linked_pyramidal_ome_tiffs_from_levels`
- [ ] `_write_morphology_for_slice`
- [ ] `_write_protein_images_for_slice`
- [ ] `_cropped_shape_from_bounds`
- [ ] `_iter_cropped_tiles`
- [ ] `_scale_bounds_for_level`
- [ ] `_roi_bounds_in_pixels`
- [ ] `_pixel_aligned_bounds_um`

### Follow-up

- [ ] Update `XenData.write_xenium_explorer()`
- [ ] Update `XenData.write_geo_submission()`
- [ ] Ensure protein pyramid fallback still works

### Deliverable

- image IO/export code leaves the monolith

## Phase 6: Move plotting helpers

The file currently mixes plotting, IO, and analysis too heavily.

### Move to `plotting/images.py`

- [ ] `show_ome_tiff`

### Move to `plotting/transcript_rasters.py`

- [ ] `create_bins`
- [ ] `bin_expression`
- [ ] `create_binned_image`
- [ ] `create_multilayer_image`
- [ ] `rasterize_rgb`
- [ ] `plot_binned_rgb`
- [ ] `plot_binned_greyscale`
- [ ] `splat`

### Move to `plotting/boundaries.py`

- [ ] any lower-level boundary rendering helpers if extracted from `XenData.plot_boundaries`

### Follow-up

- [ ] Keep `XenData.show_image()`, `XenData.splat()`, `XenData.plot_boundaries()` as thin wrappers

### Deliverable

- plotting logic separated from object model and IO

## Phase 7: Move analysis helpers

### Move to `analysis/normalization.py`

- [ ] `to_TP10k`
- [ ] `TP10K`

### Move to `analysis/binning.py`

- [ ] any residual binning helpers not already moved
- [ ] potentially shared logic used by `create_binned_adata()`

### Move to `analysis/graph.py`

- [ ] `_apply_weights`
- [ ] `build_spatial_graph`

### Move to `analysis/niches.py`

- [ ] `build_niches`

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

### Responsibilities to remove from `XenData`

- [ ] direct file parsing
- [ ] heavy binning implementation
- [ ] heavy rasterization implementation
- [ ] image-level TIFF details
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
