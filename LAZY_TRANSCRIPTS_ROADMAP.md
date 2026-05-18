# LazyTranscripts Roadmap

This checklist captures the next steps for making `LazyTranscripts` a clear,
purpose-built alternative to general Dask/SpatialData workflows for fast native
Xenium/Atera bundle inspection.

## Positioning

- [x] Document the intended scope: fast ROI inspection, gene-subset plotting,
      native bundle access, and low-overhead interactive workflows.
- [x] Document non-goals: full scverse/napari interop, transform-heavy
      multimodal data management, and arbitrary whole-dataset computation.
- [x] Add a "when to use parquet vs zarr" section.
- [x] Add a "when to use xentools vs SpatialData/Dask" section.

## Streaming and Whole-Dataset Work

- [x] Add `LazyTranscripts.iter_tiles()` to yield `(tile_key, DataFrame)` pairs.
- [x] Use `iter_tiles()` as the primitive for out-of-core reductions.
- [ ] Consider helper reductions for common whole-slide tasks:
      per-gene counts, bounded density maps, and QC summaries.

## Query Diagnostics

- [x] Add a query diagnostic method that reports candidate tile count,
      candidate transcript count, returned transcript count, and coordinate-mask
      survival fraction.
- [ ] Use diagnostics to measure whether tile slop is significant for typical
      ROI sizes.
- [ ] Decide whether a finer secondary index is warranted only after measuring
      real datasets.

## Cache Behavior

- [x] Replace the simple query-result dict with an LRU cache.
- [x] Add a byte cap such as `cache_max_bytes`.
- [x] Add `cache_info()` so users can inspect cache size and entry count.
- [x] Document that the cache is intended for interactive reuse, not long
      whole-slide iteration.

## Quality and Metadata

- [x] Document that zarr-backed transcript access exposes high/low quality
      partitions from the native zarr layout, not arbitrary per-transcript QV
      thresholds.
- [x] Direct users to `transcripts.parquet` when they need rich per-transcript
      metadata such as cell assignment, distance-to-nucleus, or custom QV
      filtering.

## Concurrency

- [x] Document that `LazyTranscripts` is optimized for single-process
      interactive use.
- [x] Clarify that each query opens its own zarr zip store, but the in-object
      cache is not designed as a thread-safe shared cache.
- [ ] Revisit parallel tile reads only after measuring I/O behavior on local
      SSDs, external drives, and network storage.
