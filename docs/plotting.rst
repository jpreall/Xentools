Plotting
========

xentools exposes plotting functions through ``xentools.pl`` and also provides
convenience methods on :class:`xentools.XenData`.

For example, these are equivalent in spirit:

.. code-block:: python

   xdata.splat(["EPCAM", "KRT19"])
   xentools.pl.splat(xdata, ["EPCAM", "KRT19"])

Transcript Splats
-----------------

``splat`` rasterizes transcript coordinates into one or more image channels.
It is useful for visualizing marker genes or gene signatures over spatial
regions.

.. code-block:: python

   ax = xdata.splat(
       genes=["EPCAM", "COL1A1", "PTPRC"],
       sigma_um=2.0,
       pixel_size_um=1.0,
       gains=(1.0, 1.5, 2.0),
   )

When ``genes`` is a dictionary, each dictionary key becomes a channel label and
the listed genes are summed into that channel.

.. code-block:: python

   ax = xdata.splat(
       genes={
           "R": ["EPCAM", "KRT19"],
           "G": ["COL1A1", "DCN"],
           "B": ["PTPRC", "CD3D"],
       }
   )

By default, ``splat`` returns a matplotlib axes object. Use ``return_array`` to
return image arrays for downstream processing:

.. code-block:: python

   display = xdata.splat(["EPCAM"], return_array=True)
   raw = xdata.splat(["EPCAM"], return_array="raw")

Images
------

``show_image`` displays morphology or protein OME-TIFF channels:

.. code-block:: python

   ax = xdata.show_image("DAPI", level=3)

If the image is pyramidal, choose a level appropriate for display. Higher
levels are typically lower resolution and faster to display.

Boundaries
----------

``plot_boundaries`` overlays cell or nucleus outlines. It is designed to work
both as a standalone plot and as an overlay on a splat or image axes.

.. code-block:: python

   ax = xdata.splat(["EPCAM", "KRT19"])
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

Because plotting functions return standard matplotlib axes, simple manual
composition is usually enough:

.. code-block:: python

   ax = xdata.show_image("DAPI", level=3, figsize=(8, 8))
   xdata.splat(["EPCAM", "KRT19"], ax=ax, show_legend=False)
   xdata.plot_boundaries(kind="cell", ax=ax, edge_alpha=0.2)
