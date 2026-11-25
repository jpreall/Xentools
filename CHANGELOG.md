# Changelog

## Unreleased

### Added

- Lazy Zarr splats support opt-in tile reading and processing with `workers`. Use `xd.splat(genes, workers=4)` to enable threading. The default is `workers=1` for serial processing.

### Performance

- Splat rasterization accumulates transcript counts in one integer buffer and converts to `float32` at the end, avoiding a full-image histogram for each transcript range.
- Gene and quality ranges that share a Zarr chunk are read together. This reduces repeated reads and decompression for gene sets spread across the panel while preserving the selected rows.
