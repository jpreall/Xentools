# Transcript Metadata Sidecar Plan

## Why this exists

`xentools` currently has two useful but incomplete transcript representations:

- `LazyTranscripts` from `transcripts.zarr.zip`
  - efficient for spatial tile-based transcript retrieval
  - currently exposes only a reduced transcript table view
- `transcripts.parquet`
  - contains richer per-transcript metadata such as `cell_id` and `nucleus_distance`
  - expensive to scan repeatedly for large datasets

The long-term problem is how to preserve rich metadata access without giving up the
scalability benefits of the Zarr-backed lazy access path, especially for very large
Atera datasets.

## What we learned

### 1. `transcripts.zarr.zip` contains more metadata than `LazyTranscripts` currently returns

From inspection of `files/xenium_v1_testdata/transcripts.zarr.zip`, each level-0 tile
contains per-transcript arrays including:

- `location`
- `gene_identity`
- `quality_score`
- `codeword_identity`
- `id`
- `uuid`
- `status`
- `valid`

So the current `LazyTranscripts.query()` is leaving useful fields on the table.

### 2. `nucleus_distance` does not appear to be present in the Zarr transcript tiles

The example archive and 10x documentation both suggest that `nucleus_distance` is not
stored in `transcripts.zarr.zip`, and remains parquet-only.

### 3. The test `transcripts.parquet` is only partially favorable for spatial pruning

Observed in `files/xenium_v1_testdata/transcripts.parquet`:

- row groups are strongly clustered by `x_location`
- row groups span nearly the full `y_location` range
- parquet statistics for relevant columns do not appear to be recorded

Practical interpretation:

- row order is somewhat spatially coherent
- but automatic bbox pruning from parquet alone is not reliable enough for future
  very large datasets

### 4. Recomputing metadata from boundaries is the wrong long-term default

Approximate re-derivation from geometry is possible, but:

- more expensive than reading stored metadata
- likely to disagree with pipeline-native assignments
- especially poor for `nucleus_distance`

So if we want scale and fidelity, stored metadata should remain the source of truth.

## Proposed direction

Create a **tile-aligned parquet sidecar** that stores only the transcript metadata
missing from the Zarr-backed lazy representation.

### Core idea

Keep:

- Zarr for fast spatial transcript access

Add:

- a sidecar parquet dataset containing minimal transcript metadata, partitioned by
  the same level-0 tile coordinates used by `transcripts.zarr.zip`

### Candidate columns

Minimal sidecar columns:

- `transcript_id`
- `fov_index` or `fov_name`
- `cell_id`
- `nucleus_distance`
- `overlaps_nucleus`

Optional:

- `x_location`
- `y_location`

Those are only useful for debugging or validation; they should not be required if the
sidecar is tile-aligned and keyed correctly.

### Partitioning scheme

Preferred layout:

```text
transcripts.metadata.parquet/
  grid_col=0/grid_row=0/part-0.parquet
  grid_col=0/grid_row=1/part-0.parquet
  ...
```

This should match the level-0 tile logic used by `transcripts.zarr.zip`.

### Join key

Best-case join key is the same identity stored in the tile `id` array:

- transcript ID
- FOV index

If `transcript_id` is globally unique, use just that. If not, use a composite key:

- `(transcript_id, fov_index)`

Do not join by floating-point positions.

## Intended query flow

For metadata-enriched transcript queries:

1. Use `LazyTranscripts` to identify candidate level-0 tiles from the bbox.
2. Read only the matching sidecar parquet partitions.
3. Join sidecar metadata onto the queried transcripts by transcript identity.
4. Apply metadata-driven filters such as:
   - exclude unassigned transcripts
   - threshold by `nucleus_distance`

This keeps query cost proportional to the requested spatial region, not the total size
of the transcript table.

## API ideas

No implementation decision yet, but likely options:

### Option A: enrich in `query()`

```python
trans.query(..., include_metadata=["cell_id", "nucleus_distance"])
```

Pros:

- convenient for users

Cons:

- increases complexity of the core query path

### Option B: separate enrichment step

```python
df = trans.query(...)
df = trans.enrich_metadata(df, columns=["cell_id", "nucleus_distance"])
```

Pros:

- cleaner separation of concerns
- easier to cache and reason about

Cons:

- slightly less convenient

Current leaning: Option B is architecturally cleaner.

## Build strategy

The sidecar could be produced by:

- a dedicated helper in `xentools`
- export-time preprocessing
- an explicit CLI/utility script for one-time dataset preparation

High-level build algorithm:

1. Read `transcripts.parquet` once.
2. Compute the same level-0 tile indices used by the Zarr bundle.
3. Select only required metadata columns.
4. Write partitioned parquet by `grid_col`, `grid_row`.

Pseudo-shape:

```python
df["grid_col"] = (df["x_location"] / tile_size_um).astype(int)
df["grid_row"] = (df["y_location"] / tile_size_um).astype(int)

meta = df[
    [
        "transcript_id",
        "fov_name",
        "cell_id",
        "nucleus_distance",
        "overlaps_nucleus",
        "grid_col",
        "grid_row",
    ]
]

meta.to_parquet(sidecar_dir, partition_cols=["grid_col", "grid_row"])
```

## What we already changed instead

Before implementing the sidecar, `XenData(..., transcript_source="parquet")` was added
so users can force `.trans` to use the rich parquet-backed transcript table when needed.

Also, `create_binned_adata()` now prefers `transcripts.parquet` in metadata-dependent
cases when `self.trans` is a `LazyTranscripts` object and parquet is available.

These changes solve the immediate usability issue, but not the long-term scaling issue.

## Open questions for later

- Should sidecar generation be automatic, optional, or user-invoked?
- Should `LazyTranscripts.query()` eventually expose more of the Zarr-native metadata
  such as `quality_score`, `status`, `valid`, and `fov_index`?
- Should the sidecar include only parquet-only fields, or also normalize some Zarr
  metadata for a unified transcript table API?
- How should sidecar presence be detected and versioned?
- Should sidecar reads be cached per tile?

## Current conclusion

Do not implement this yet.

When revisiting:

- keep Zarr as the primary spatial transcript store
- do not recompute `nucleus_distance` from geometry as a default path
- prefer a tile-aligned metadata sidecar over repeated scans of monolithic parquet
