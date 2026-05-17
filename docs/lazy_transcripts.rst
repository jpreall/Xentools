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
    ``x_location`` and ``y_location``.

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

Design Limitations
------------------

``LazyTranscripts`` currently prioritizes fast spatial/gene queries and plotting
over exposing every transcript metadata column. For richer per-transcript
metadata such as cell assignment or distance to nucleus, loading
``transcripts.parquet`` may still be preferable when the file is small enough.
