LazyTranscripts
===============

:class:`xentools.LazyTranscripts` is xentools' lazy, tile-aware representation
of transcript-level data stored in ``transcripts.zarr.zip``. It is designed for
datasets where loading every transcript into memory at initialization would be
slow, wasteful, or impractical.

Most users access it through:

.. code-block:: python

   import xentools

   xdata = xentools.XenData("/path/to/xe_outs")
   xdata.trans

If ``xdata.trans`` is a ``LazyTranscripts`` object, transcript rows remain on
disk until a query, plot, or conversion method asks for a subset.

Why It Exists
-------------

Transcript-level Xenium and Atera datasets can contain tens to hundreds of
millions of rows. A normal ``pandas.DataFrame`` is convenient but requires
materializing the whole transcript table in memory.

``LazyTranscripts`` makes a different tradeoff:

* keep a compact spatial tile index in memory
* read only candidate spatial tiles from ``transcripts.zarr.zip``
* use explicit tile metadata from ``grids/.zattrs`` when available
* use per-gene tile offsets to avoid scanning unnecessary genes
* return ordinary ``pandas.DataFrame`` objects for requested subsets

This is not intended to be a general replacement for Dask or SpatialData. It is
a small query engine for the access pattern xentools prioritizes: open a native
Xenium/Atera bundle quickly, select a region, select a few genes, and plot or
inspect the result with minimal overhead.

When to Use LazyTranscripts
---------------------------

Use zarr-backed ``LazyTranscripts`` when you want:

* fast startup on large native output bundles
* spatial bounding-box queries
* marker-gene or gene-signature plotting
* ROI-level inspection without converting the dataset first
* low-overhead results as ordinary ``pandas.DataFrame`` objects

Prefer ``transcripts.parquet`` when you need:

* rich per-transcript metadata such as cell assignment or distance to nucleus
* arbitrary per-transcript filtering on columns not exposed in the zarr tile
  arrays
* flexible QV thresholds beyond the high/low partitions represented by the
  native zarr offsets
* a complete in-memory transcript table for smaller datasets

Prefer SpatialData, Dask, or another ecosystem-native representation when you
need:

* whole-dataset transcript computations that do not fit in memory
* automatic parallelism across many chunks or samples
* napari/scverse interoperability
* a unified multimodal object with explicit coordinate transforms for images,
  labels, points, shapes, and tables

Backing Store
-------------

The relevant transcript zarr layout is a spatial tile pyramid:

.. code-block:: text

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

Only level ``grids/0`` is currently used for precise transcript queries.

The key arrays are:

``location``
    Per-transcript coordinates. xentools uses columns 0 and 1 as
    ``x_location`` and ``y_location`` by default. Column 2 is available as
    ``z_location`` when ``include_z=True`` is requested.

``gene_identity``
    Integer gene or codeword identity for each transcript.

``gene_offset``
    Per-gene slice offsets into the tile's transcript arrays.

Spatial Query Flow
------------------

The main access method is:

.. code-block:: python

   df = xdata.trans.query(
       xmin=1000,
       xmax=1500,
       ymin=2000,
       ymax=2500,
       genes=["EPCAM", "KRT19"],
       quality="high",
   )

Set ``include_z=True`` when the transcript z coordinate is needed:

.. code-block:: python

   df = xdata.trans.query(
       xmin=1000,
       xmax=1500,
       ymin=2000,
       ymax=2500,
       genes=["EPCAM", "KRT19"],
       quality="high",
       include_z=True,
   )

Queries proceed in two stages.

First, ``LazyTranscripts`` selects candidate tiles from the tile index. The
requested micron bounds are converted to tile columns and rows using the zarr
grid size, then only keys that are present in the backing store are considered.

Second, the selected tiles are read. If genes are specified, ``gene_offset`` is
used to read only the relevant transcript slices. A precise coordinate mask
then removes transcripts outside the requested bounds.

The result is a regular DataFrame with columns such as:

.. code-block:: text

   x_location    y_location    feature_name

With ``include_z=True``, the returned DataFrame includes:

.. code-block:: text

   x_location    y_location    z_location    feature_name

Streaming Tile Iteration
------------------------

For out-of-core workflows, use :meth:`LazyTranscripts.iter_tiles` to stream
matching tile results without concatenating the full query into one DataFrame:

.. code-block:: python

   totals = {}
   for tile_key, df in xdata.trans.iter_tiles(genes=["EPCAM", "KRT19"], quality="all"):
       counts = df["feature_name"].value_counts()
       for gene, count in counts.items():
           totals[gene] = totals.get(gene, 0) + int(count)

``iter_tiles`` yields ``(tile_key, DataFrame)`` pairs and skips empty candidate
tiles by default. It is the preferred primitive for reductions that need to
touch a large portion of a zarr-backed transcript store.

Query Diagnostics
-----------------

Use :meth:`LazyTranscripts.diagnose_query` to measure how efficiently the tile
index prunes a proposed query:

.. code-block:: python

   info = xdata.trans.diagnose_query(
       xmin=1000,
       xmax=1500,
       ymin=2000,
       ymax=2500,
       genes=["EPCAM", "KRT19"],
       quality="all",
   )
   info

The returned dictionary includes:

``candidate_tiles``
    Number of tiles selected by the spatial index.

``candidate_transcripts``
    Number of transcripts considered before precise coordinate masking. For
    gene-filtered queries, this uses ``gene_offset`` to count only requested
    gene/quality slices.

``returned_transcripts``
    Number of transcripts remaining after coordinate masking.

``survival_fraction``
    ``returned_transcripts / candidate_transcripts``. Low values indicate that
    the ROI is small relative to the underlying tile size and that substantial
    tile slop is being read and masked out.

Quality Modes
-------------

``LazyTranscripts.query()`` accepts a ``quality`` argument:

``"high"``
    Read high-quality transcript slices only. This is the conservative default.

``"low"``
    Read low-quality transcript slices only.

``"all"``
    Read both low- and high-quality transcript slices.

Caching
-------

Query results may be cached in memory. The cache is intended to speed up
interactive work where the same ROI and gene set are queried repeatedly for
plotting.

Query results below ``cache_threshold`` rows are stored in an in-object LRU
cache. The cache also has a byte-level cap controlled by ``cache_max_bytes``.
By default this cap is ``512_000_000`` bytes. Set ``cache_max_bytes=None`` to
disable byte-level eviction, or use a smaller value to constrain notebook memory
use more aggressively.

Inspect the cache with:

.. code-block:: python

   xdata.trans.cache_info()

Clear cached query results with:

.. code-block:: python

   xdata.trans.clear_cache()

Design Limitations
------------------

``LazyTranscripts`` currently prioritizes fast spatial/gene queries and plotting
over exposing every transcript metadata column. For richer per-transcript
metadata such as cell assignment or distance to nucleus, loading
``transcripts.parquet`` may still be preferable when the file is small enough.

The object is optimized for single-process interactive use. Each query opens
its own zarr zip store, but the in-object cache is not designed as a
thread-safe shared cache for parallel workers.
