# LazyTranscripts Structure

`LazyTranscripts` is xentools' lazy, tile-aware representation of transcript-level data stored in
`transcripts.zarr.zip`. It is designed for datasets where loading every transcript into memory at
initialization would be slow, wasteful, or impossible.

The class lives in `core/transcripts.py` and is normally reached through:

```python
xdata = xentools.XenData("/path/to/xenium/outs")
xdata.trans
```

When `xdata.trans` is a `LazyTranscripts` object, transcript data remains on disk until a query,
plot, or conversion method asks for a specific subset.

## Why This Exists

Transcript-level Xenium and Atera datasets can contain tens to hundreds of millions of rows. A
standard `pandas.DataFrame` representation is convenient, but it requires materializing the whole
transcript table in memory.

`LazyTranscripts` makes a different tradeoff:

- keep a lightweight spatial index in memory
- read only candidate spatial tiles from `transcripts.zarr.zip`
- use the explicit tile map in `grids/.zattrs` when available
- use per-gene tile offsets to avoid scanning unnecessary genes
- return ordinary `pandas.DataFrame` or `xarray.DataArray` objects only for requested subsets

This keeps initialization and localized queries practical for large datasets.

## Backing Store

`LazyTranscripts` reads Xenium/Atera transcript data from:

```text
transcripts.zarr.zip
```

The relevant layout is a spatial tile pyramid:

```text
transcripts.zarr.zip
└── grids
    └── 0
        ├── 0,0
        │   ├── location
        │   ├── gene_identity
        │   └── gene_offset
        ├── 1,0
        │   ├── location
        │   ├── gene_identity
        │   └── gene_offset
        └── ...
```

Only level `grids/0` is currently used for transcript queries. Each tile contains arrays for the
transcripts in that spatial tile.

The core arrays used by `LazyTranscripts` are:

| Array | Meaning |
|---|---|
| `location` | Per-transcript coordinates. xentools uses columns 0 and 1 as `x_location` and `y_location`. |
| `gene_identity` | Integer gene/codeword identity for each transcript. |
| `gene_offset` | Per-gene slice offsets into the tile's transcript arrays. |

`gene_identity` values are mapped back to gene names using the ordered gene list loaded during
`XenData` initialization.

The explicit spatial map is stored in:

```text
grids/.zattrs
```

The most important fields are:

| Attribute | Meaning |
|---|---|
| `grid_keys` | Tile keys for each pyramid level, including level 0. |
| `grid_number_objects` | Transcript/object counts aligned to `grid_keys`. |
| `grid_size` | Tile width/height in microns, usually `250.0`. |
| `number_levels` | Number of spatial grid levels in the transcript Zarr. |

For level 0, a tile key such as `"12,7"` is interpreted as `(column=12, row=7)`.
If `grid_size = 250`, that tile covers:

```text
x: 12 * 250 to 13 * 250
y:  7 * 250 to  8 * 250
```

## Initialization

Creating a `LazyTranscripts` object does not load all transcript rows. Instead, it builds a compact
tile index.

```python
trans = LazyTranscripts(
    zarr_path="transcripts.zarr.zip",
    gene_names=gene_names,
    cache_threshold=5_000_000,
)
```

During initialization, xentools:

1. Reads `grids/.zattrs`.
2. Extracts level-0 `grid_keys`, `grid_number_objects`, and `grid_size`.
3. Parses each tile key as `(column, row)`.
4. Computes exact tile bounds from `column * grid_size` and `row * grid_size`.
5. Builds a mapping from gene name to gene index.

The result is a small in-memory structure that looks conceptually like:

```python
{
    "0,0": {"n": 199, "x0": 0.0, "x1": 250.0, "y0": 0.0, "y1": 250.0},
    "1,0": {"n": 1204, "x0": 250.0, "x1": 500.0, "y0": 0.0, "y1": 250.0},
    ...
}
```

This index is enough to identify candidate tiles for rectangular spatial queries without reading
the full transcript table.

If the explicit grid metadata is missing or malformed, `LazyTranscripts` falls back to the older
method: scanning tile array metadata, reading one representative coordinate per tile, and estimating
tile spacing from those sampled coordinates. That fallback exists for compatibility but is not the
preferred path.

## Spatial Query Flow

The main access method is:

```python
df = xdata.trans.query(
    xmin=1000,
    xmax=1500,
    ymin=2000,
    ymax=2500,
    genes=["EPCAM", "KRT19"],
    quality="high",
)
```

The query proceeds in two stages.

First, `LazyTranscripts` selects candidate tiles from the explicit tile index. It converts the
requested micron bounds to tile columns and rows using `grid_size`, then keeps only keys that are
actually present in the Zarr metadata. Boundary-adjacent tiles may be included conservatively; the
precise coordinate mask later removes transcripts outside the requested bounds.

Second, it opens `transcripts.zarr.zip` and reads only those candidate tiles. Within each tile:

- If genes are specified, `gene_offset` is used to read only the relevant transcript slices.
- If no genes are specified, all transcript coordinates in candidate tiles are read.
- A precise coordinate mask is applied to keep only transcripts inside the requested bounds.
- Integer gene identities are converted back to feature names.

The result is a regular DataFrame:

```text
x_location    y_location    feature_name
1201.4        2304.7        EPCAM
1228.0        2310.2        KRT19
...
```

The DataFrame currently contains:

- `x_location`
- `y_location`
- `feature_name`

It intentionally does not currently expose every metadata field stored in `transcripts.zarr.zip`.

## Gene Filtering

The Zarr tiles are sorted/indexed by gene. This is why gene-filtered queries are efficient.

For a requested gene, `LazyTranscripts` looks up the gene's integer ID and uses the tile's
`gene_offset` array to find the relevant row ranges. Those ranges are read directly from
`location` and `gene_identity`.

This means:

- querying a small gene set is much cheaper than reading a full tile
- plotting a few marker genes can remain practical on large datasets
- missing genes are ignored if none of the requested names exist in the Zarr gene index

## Quality Modes

`LazyTranscripts.query()` accepts a `quality` argument:

```python
quality="high"  # default
quality="low"
quality="all"
```

Internally, `gene_offset` has separate ranges for low- and high-quality transcript groups. xentools
uses those ranges as follows:

| Mode | Ranges read |
|---|---|
| `"high"` | high-quality transcript slices only |
| `"low"` | low-quality transcript slices only |
| any other value, conventionally `"all"` | both low- and high-quality slices |

For most analysis and plotting, `"high"` is the conservative default. Use `"all"` when reproducing
total transcript density or when low-quality calls should be included.

## Caching

Query results may be cached in memory.

Each query is keyed by:

- rounded bounds
- selected genes
- quality mode

If the result has fewer rows than `cache_threshold`, it is stored in `_query_cache`.

```python
xdata.trans.clear_cache()
```

Caching helps repeated plotting or interactive exploration of the same ROI. Large query results are
not cached by default because that would defeat the purpose of lazy loading.

The default cache threshold is controlled by `XenData(..., cache_threshold=...)`.

## DataFrame Conversion

To materialize transcripts into a DataFrame:

```python
df = xdata.trans.to_dataframe(quality="high")
```

This is equivalent to querying without spatial bounds or gene filters. It can therefore load a very
large number of rows and should be used carefully on large datasets.

For large datasets, prefer bounded queries:

```python
df = xdata.trans.query(
    xmin=1000,
    xmax=2000,
    ymin=1000,
    ymax=2000,
    genes=["EPCAM", "KRT19"],
)
```

## Binned xarray Conversion

`LazyTranscripts` can also produce a binned `xarray.DataArray`:

```python
arr = xdata.trans.to_xarray_bins(
    bin_size=5,
    genes=["EPCAM", "KRT19"],
    bounds=(1000, 2000, 1000, 2000),
    quality="all",
)
```

The returned array has dimensions:

```text
(feature_name, y, x)
```

and coordinates:

- `feature_name`: selected genes/features
- `y`: bin center y-coordinate in microns
- `x`: bin center x-coordinate in microns

This is useful when the desired output is an image-like expression cube rather than a transcript
point table.

## Relationship To XenData

`XenData` chooses the transcript representation during initialization.

For Zarr-format bundles, `XenData` can use:

- `LazyTranscripts` from `transcripts.zarr.zip`
- a fully materialized `pandas.DataFrame` from `transcripts.parquet`

The current default is adaptive: when a Zarr bundle also has `transcripts.parquet` and the dataset
has fewer transcripts than `eager_transcript_threshold`, xentools can use the richer parquet table.
For larger datasets, lazy Zarr access avoids loading the full transcript table.

The source can be overridden:

```python
xdata = xentools.XenData(path, transcript_source="zarr")
xdata = xentools.XenData(path, transcript_source="parquet")
```

## Strengths

`LazyTranscripts` works well for:

- very large transcript sets
- ROI-restricted plotting
- marker-gene visualization
- binned expression images
- workflows where most operations touch a small spatial region or gene subset

Its main advantage is that I/O scales with the queried region and gene set rather than with the
entire dataset.

## Limitations

`LazyTranscripts` is not a drop-in replacement for `pandas.DataFrame`.

Important limitations:

- It is not subscriptable like `df["feature_name"]`.
- It does not currently return all transcript metadata available in `transcripts.zarr.zip`.
- It does not include parquet-only metadata such as cell assignment or distance-to-nucleus.
- Whole-dataset queries can still be expensive because they intentionally materialize a DataFrame.
- Candidate tile selection is tile-level, so queries still apply final coordinate filtering after loading tile slices.

If code needs full pandas semantics, explicitly materialize a DataFrame for a bounded query or use
`transcript_source="parquet"` when initializing `XenData`.

## Future Directions

Possible improvements:

- expose additional Zarr-native transcript metadata from each tile
- add optional sidecar metadata for efficient cell assignment and distance-to-nucleus lookup
- provide richer query return schemas with opt-in columns
- improve integration with Datashader for large-scale point rendering
- support more output formats for binned transcript cubes
- make tile-level indexing more explicit and inspectable

The current design intentionally keeps the lazy layer narrow: it handles efficient transcript
retrieval and leaves richer representations to explicit conversion methods.
