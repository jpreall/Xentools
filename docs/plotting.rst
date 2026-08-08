Plotting
========

xentools exposes plotting functions through ``xentools.pl`` and also provides
convenience methods on :class:`xentools.XenData`.

For example, these are equivalent in spirit:

.. code-block:: python

   xdata.plot_splat(["EPCAM", "KRT19"])
   xentools.pl.plot_splat(xdata, ["EPCAM", "KRT19"])

Older shorter names such as ``xdata.splat()``, ``xdata.points()``, and
``xdata.show_image()`` remain available as aliases.

Saving Figures
--------------

Most plotting functions accept ``save`` and ``save_kwargs``:

.. code-block:: python

   ax = xdata.plot_splat(
       ["EPCAM", "KRT19"],
       save="figures/epithelial_splat.png",
       save_kwargs={"dpi": 300},
   )

``save`` can be a string or path-like object. Parent directories are created
automatically. ``save_kwargs`` is forwarded to Matplotlib's
``Figure.savefig``; by default xentools uses ``bbox_inches="tight"``.

Composite Rendering
-------------------

``render`` is the high-level compositor for common exploratory views. It puts
image layers on the bottom, splats on top of images with transparent
zero-signal pixels, then point and cell-boundary overlays last.

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2},
       splat={
           "genes": ["C7", "Epcam", "Tagln"],
           "gains": [3, 3, 3],
       },
       points={
           "genes": ["Prss3"],
           "s": 0.4,
       },
       cells=True,
       bounds=(500, 1500, 900, 1900),
       save="figures/composite.png",
   )

For multiple image channels, pass ``images``. The first image is treated as
the background; later images are converted into transparent signal overlays so
dark pixels do not cover the lower layers.

.. code-block:: python

   ax = xdata.render(
       images={
           "DAPI": {"level": 2},
           "18S": {"level": 2, "color": "green", "alpha": 0.6},
       },
       splat={"genes": ["C7", "Epcam", "Tagln"], "gains": [3, 3, 3]},
       bounds=(500, 1500, 900, 1900),
   )

The lower-level plotting functions below remain useful when you want manual
control over every layer. See :doc:`rendering` for a fuller guide to the
compositor interface and layer option schemas.

Transcript Splats
-----------------

``plot_splat`` rasterizes transcript coordinates into one or more image
channels. It is useful for visualizing marker genes or gene signatures over
spatial regions.

.. code-block:: python

   ax = xdata.plot_splat(
       genes=["EPCAM", "COL1A1", "PTPRC"],
       sigma_um=2.0,
       gains=(1.0, 1.5, 2.0),
   )

When ``genes`` is a dictionary, each dictionary key becomes a channel label and
the listed genes are summed into that channel.

.. code-block:: python

   ax = xdata.plot_splat(
       genes={
           "R": ["EPCAM", "KRT19"],
           "G": ["COL1A1", "DCN"],
           "B": ["PTPRC", "CD3D"],
       }
   )

For zarr-backed lazy transcripts, ``plot_splat`` uses a direct zarr-to-raster
path so it does not need to build an intermediate transcript DataFrame. This is
especially useful for Atera-scale datasets and ROI-focused plotting.

By default, ``plot_splat`` uses ``pixel_size_um="auto"`` and targets roughly
1 million raster pixels. This keeps large-region notebook plotting responsive.
To control output size directly, set ``target_pixels``:

.. code-block:: python

   ax = xdata.plot_splat(
       genes=["EPCAM", "COL1A1", "PTPRC"],
       target_pixels=8_000_000,
   )

For high-resolution figure generation, pass an explicit spatial resolution.
Numeric ``pixel_size_um`` always overrides ``target_pixels``:

.. code-block:: python

   ax = xdata.plot_splat(
       genes=["EPCAM", "COL1A1", "PTPRC"],
       pixel_size_um=1.0,
   )

When auto-resolution chooses a coarser value, Xentools emits an informational
warning. Experienced users can suppress these messages globally:

.. code-block:: python

   import xentools

   xentools.settings.verbosity = 0

Large gene signatures can still be expensive because each additional gene adds
more zarr slices to read. To keep interactive plotting responsive, splats clip
each gene-set channel to the first 50 genes by default and issue a warning when
this happens. If you intentionally want every gene in a large signature, opt in
explicitly:

.. code-block:: python

   ax = xdata.plot_splat(
       genes=large_signature_dict,
       force_all_genes=True,
   )

You can also change or disable the cap:

.. code-block:: python

   ax = xdata.plot_splat(
       genes=large_signature_dict,
       max_signature_genes=100,
   )

   ax = xdata.plot_splat(
       genes=large_signature_dict,
       max_signature_genes=None,
   )

By default, ``splat`` returns a matplotlib axes object. Use ``return_array`` to
return image arrays for downstream processing:

.. code-block:: python

   display = xdata.plot_splat(["EPCAM"], return_array=True)
   raw = xdata.plot_splat(["EPCAM"], return_array="raw")

Binned Splats
-------------

``plot_binned_splat`` draws the same kind of RGB gene or gene-set density image
from a precomputed ``xdata.binned_adata`` matrix. This is useful when repeatedly
plotting large tissue regions because the expensive transcript-to-bin
aggregation is done once.

.. code-block:: python

   xdata.create_binned_adata(bin_size=5)

   ax = xdata.plot_binned_splat(
       genes={
           "R": ["EPCAM", "KRT19"],
           "G": ["COL1A1", "DCN"],
           "B": ["PTPRC", "CD3D"],
       },
       gains=(1.0, 1.5, 2.0),
   )

Like ``plot_splat``, it accepts ``ax`` for manual compositing and
``return_array=True`` or ``return_array="raw"`` for array output. It can also be
used directly in the high-level compositor:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       binned_splat={"genes": ["EPCAM", "COL1A1", "PTPRC"], "gains": [2, 2, 2]},
       cells=True,
   )

Transcript Points
-----------------

``plot_points`` draws individual transcript molecules as matplotlib scatter
markers. Use this for focused ROIs where seeing single molecules is useful;
for dense regions or whole-slide views, ``plot_splat`` is usually faster and
more readable.

.. code-block:: python

   ax = xdata.plot_points(
       genes=["EPCAM", "KRT19"],
       max_points=50_000,
       s=6,
       alpha=0.7,
   )

When ``genes`` is a dictionary, each key becomes a point category:

.. code-block:: python

   ax = xdata.plot_points(
       genes={
           "epithelial": ["EPCAM", "KRT19"],
           "immune": ["PTPRC", "CD3D"],
       },
       markers={"epithelial": "o", "immune": "^"},
   )

To avoid accidentally sending millions of artists to matplotlib,
``plot_points`` randomly samples to ``max_points=100_000`` by default. Pass
``max_points=None`` to draw every matching transcript. For DataFrame/parquet
transcripts with a ``cell_id`` column, ``plot_points`` defaults to assigned
cell transcripts only; pass ``assigned_only=False`` to include unassigned
transcripts.

Images
------

``plot_image`` displays morphology or protein OME-TIFF channels:

.. code-block:: python

   ax = xdata.plot_image("DAPI", level=3)

If the image is pyramidal, choose a level appropriate for display. Higher
levels are typically lower resolution and faster to display.

External aligned images
~~~~~~~~~~~~~~~~~~~~~~~

H&E and separately acquired immunofluorescence OME-TIFFs can be registered
from alignment files exported by Xenium Explorer:

.. code-block:: python

   he = xdata.import_aligned_image(
       "20260803_117451_HE.ome.tif",
       alignment_file="117451_HE_alignment_files.zip",
       name="H&E",
   )

When ``alignment_file="auto"`` (the default), xentools searches beside the
image and Xenium output and matches numeric sample identifiers, including IDs
that differ only by leading zeros. Ambiguous matches produce an error asking
for an explicit path.

The returned ``AlignedImage`` records the affine transform, source dimensions,
OME channel names, transformed bounds, and keypoint residuals:

.. code-block:: python

   print(he.keypoint_rmse_px, he.keypoint_max_error_px)
   ax = xdata.show_image("H&E")
   xdata.plot_boundaries(ax=ax, edgecolor="cyan")

The source pyramid remains unchanged. Display reads only the required source
region and applies the affine in micron coordinates. For a registered
multi-channel image, choose a source channel by name or index:

.. code-block:: python

   xdata.show_image("IF", source_channel="CD3", cmap="magma")

ROI outlines can be composed on any xentools image or transcript axes. All
layers use the same native global coordinates, so plotting an ROI does not
apply a separate coordinate reflection:

.. code-block:: python

   ax = xdata.show_image("DAPI")
   xdata.rois.plot(
       ax,
       edgecolor="yellow",
       linewidth=1,
       labels=True,
       label_kwargs={"fontsize": 8, "color": "white"},
   )

   ax = xdata.splat(["EPCAM", "KRT19"])
   xdata.rois["all_liver"].plot(ax, edgecolor="cyan")

The global Y axis follows image convention (values increase downward), and
xentools displays it with descending Matplotlib Y limits.

Boundaries
----------

``plot_boundaries`` overlays cell or nucleus outlines. It is designed to work
both as a standalone plot and as an overlay on a splat or image axes.

.. code-block:: python

   ax = xdata.plot_splat(["EPCAM", "KRT19"])
   xdata.plot_boundaries(kind="cell", ax=ax, color_by="Cluster")

Boundary legends default to positions outside the plot where possible. The
default boundary edge alpha is intentionally low so overlays remain readable.

Cell Fill Plots
---------------

``plot_cells`` fills cell polygons by annotation or expression value.

Color cells by an ``obs`` annotation:

.. code-block:: python

   ax = xdata.plot_cells(color_by="Cluster")

Color cells by gene expression:

.. code-block:: python

   ax = xdata.plot_cells(genes="EPCAM", cmap="magma")

Color cells by summed expression of a gene list:

.. code-block:: python

   ax = xdata.plot_cells(genes=["EPCAM", "KRT19"], cmap="magma")

Niche Plots
-----------

After computing niches with ``xentools.build_niches()``, visualize them with:

.. code-block:: python

   ax = xdata.niche_heatmap(key_added="niche")
   ax = xdata.niche_map(key_added="niche")

Layering
--------

For most composite plots, prefer ``render``. For manual composition, plotting
functions return standard matplotlib axes:

.. code-block:: python

   ax = xdata.plot_image("DAPI", level=3, figsize=(8, 8))
   xdata.plot_splat(["EPCAM", "KRT19"], ax=ax, show_legend=False)
   xdata.plot_points(["CD3D"], ax=ax, max_points=20_000, color="cyan")
   xdata.plot_boundaries(kind="cell", ax=ax, edge_alpha=0.2)
