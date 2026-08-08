Core Concepts
=============

XenData
-------

:class:`xentools.XenData` is the main user-facing object. It represents a
single Xenium-style output directory and exposes the pieces users usually need
for analysis and visualization:

``xdata.adata``
    Cell-by-gene :class:`anndata.AnnData` object built from the Xenium
    cell-feature matrix.

``xdata.trans``
    Transcript-level data. This may be a regular ``pandas.DataFrame`` for
    smaller datasets or a lazy :class:`xentools.LazyTranscripts` object for
    large ``transcripts.zarr.zip``-backed datasets.

``xdata.cell_boundaries`` and ``xdata.nucleus_boundaries``
    Cell and nucleus boundary polygons. These can be regular GeoDataFrames or
    lazy boundary objects that materialize only the requested region.

``xdata.images`` and ``xdata.protein_images``
    Discovered morphology and protein image files.

``xdata.rois``
    Named ROI objects loaded from GeoJSON/CSV files or created manually.

``xdata.transforms``
    Registry mapping each image, point table, and shape collection from its
    intrinsic coordinates into the canonical ``global`` coordinate system.

ROI API Naming
--------------

ROI classes follow normal Python class naming (``ROI``, ``ROIGroup``,
``ROIClass``, and ``ROICollection``). Functions, methods, and attributes use
lowercase names. The canonical collection is ``xdata.rois``.

The earlier mixed-case aliases were removed. The principal replacements are:

===============================  =============================
Removed name                     Replacement
===============================  =============================
``xdata.ROIs``                   ``xdata.rois``
``xdata.import_ROI()``           ``xdata.import_rois()``
``xdata.crop_to_ROI()``          ``xdata.subset_to_roi()``
``xdata.cropped_roi``            ``xdata.subset_roi``
``xdata.assign_cells_to_ROIs()`` ``xdata.assign_cells_to_rois()``
``xdata.assign_bins_to_ROIs()``  ``xdata.assign_bins_to_rois()``
``read_ROI_from_csv()``          ``read_roi_from_csv()``
``read_ROI_from_geojson()``      ``read_roi_from_geojson()``
``ROI_to_pixels()``              ``roi_to_pixels()``
===============================  =============================

Lazy Loading
------------

xentools uses lazy loading where it is most useful for fast exploratory work:

Transcripts
    Large zarr-backed transcript tables can be queried by spatial bounds and
    gene names without reading the whole transcript table.

Boundaries
    Cell and nucleus boundary polygons are registered at initialization but can
    be materialized only when plotted or explicitly requested.

Images
    OME-TIFF image pyramid levels are selected on demand for display.

Transcript Source Selection
---------------------------

When both ``transcripts.parquet`` and ``transcripts.zarr.zip`` are present,
xentools can choose between eager parquet loading and lazy zarr loading. The
default behavior is to eagerly load reasonably sized parquet transcript tables
and use lazy zarr access for very large datasets.

This is a deliberate tradeoff. Parquet is the richer representation for
per-transcript metadata and arbitrary column filtering. The zarr-backed
``LazyTranscripts`` path is faster for localized spatial/gene queries because
it uses the native tile grid and ``gene_offset`` arrays rather than scanning a
row-partitioned table.

Xentools and SpatialData
------------------------

xentools and SpatialData solve overlapping but different problems.

Use xentools when the priority is to open a fresh Xenium/Atera output directory
directly, inspect ROIs quickly, and make transcript/image/boundary plots without
first creating another large on-disk object.

Use SpatialData when the priority is ecosystem interoperability, explicit
coordinate transforms across many modalities, napari/scverse workflows, or
larger analysis pipelines that benefit from Dask-backed composability.

Boundary Source Selection
-------------------------

Boundaries are usually read from ``cell_boundaries.parquet`` and
``nucleus_boundaries.parquet`` when available. If those files are absent,
xentools can fall back to boundary polygons stored in ``cells.zarr.zip``.

Cell Matrix Compatibility
-------------------------

Different Xenium software versions have used different
``cell_feature_matrix.zarr.zip`` layouts. xentools supports both newer nested
``cell_features/csc`` sparse arrays and older flat ``cell_features`` sparse
arrays.

Coordinate Systems
------------------

xentools uses a platform-neutral coordinate system named ``global`` for
Xenium, Atera, and imported modalities. Its axes are ``x`` and ``y``, its
units are micrometers, and Y increases downward in image convention. Plotting
uses descending Matplotlib Y limits; stored geometries and transforms are not
reflected merely for display.

Each spatial element retains its intrinsic coordinates and has a homogeneous
3×3 transform into ``global``. Native transcript and ROI coordinates therefore
use identity transforms, while image pixels use pixel-to-global transforms.
Inspect these mappings with ``xdata.transforms`` or apply one directly:

.. code-block:: python

   print(xdata.transforms)
   xy_global = xdata.transforms.apply("image:H&E", xy_pixels)

A bounds tuple is ordered as:

.. code-block:: python

   (xmin, xmax, ymin, ymax)

The same global convention is used by transcript queries, splat plots, image
display, ROIs, and boundary overlays. Cropping changes only the view bounds; it
never changes an element's transform.
