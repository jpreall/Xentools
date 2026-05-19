Composite Rendering
===================

``render`` is xentools' high-level compositor for exploratory spatial plots.
It is intended for the common workflow of combining morphology/protein images,
transcript-density splats, single-transcript points, and cell boundaries in one
ROI-focused view.

The same function is available as:

.. code-block:: python

   xdata.render(...)
   xentools.render(xdata, ...)
   xentools.pl.render(xdata, ...)

Why Use Render?
---------------

The lower-level plotting functions all return Matplotlib axes, so manual
composition is possible. But manual composition is easy to get subtly wrong:
opaque image artists can cover transcript layers, transcript rasters can cover
the DAPI background, and multiple image channels need different blending
behavior.

``render`` applies a consistent layer order:

1. Image channels first.
2. Transcript splats over images, with zero-signal pixels transparent.
3. Transcript points over splats.
4. Cell boundaries last.

Basic Example
-------------

.. code-block:: python

   b = (500, 1500, 900, 1900)

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       splat={
           "genes": ["C7", "Epcam", "Tagln"],
           "gains": [3, 3, 3],
       },
       points={
           "genes": ["Prss3"],
           "s": 0.4,
           "color": "cyan",
       },
       cells=True,
       bounds=b,
   )

Image Layers
------------

Use ``image`` for one channel:

.. code-block:: python

   ax = xdata.render(image="DAPI", bounds=b)

Pass a dictionary when you need options:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.35},
       bounds=b,
   )

Use ``images`` for multiple channels:

.. code-block:: python

   ax = xdata.render(
       images={
           "DAPI": {"level": 2, "alpha": 0.4},
           "18S": {"level": 2, "color": "green", "alpha": 0.6},
       },
       bounds=b,
   )

The first image is treated as the background. Additional images are converted
into transparent signal overlays so dark pixels do not hide the background.

Common image options:

``level``
    Pyramid level to display. Higher levels are lower resolution and faster.

``alpha``
    For the first/background image, dims brightness without making black pixels
    gray. For later image overlays, controls maximum signal opacity.

``color``
    Colorize a grayscale image layer, useful for protein/fluorescence channels.

``cmap``
    Matplotlib colormap. If ``color`` is supplied, xentools creates a black-to-
    color colormap automatically.

``vmin``, ``vmax``, ``clip_percentile``
    Display intensity controls passed to ``plot_image``.

Splat Layer
-----------

Use a gene list:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       splat=["C7", "Epcam", "Tagln"],
       bounds=b,
   )

Use a dictionary for splat options:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       splat={
           "genes": ["C7", "Epcam", "Tagln"],
           "gains": [3, 3, 3],
           "pixel_size_um": 1.0,
           "sigma_um": 2.0,
           "splat_alpha": 0.8,
       },
       bounds=b,
   )

Use a gene-set dictionary when each display channel should summarize multiple
genes:

.. code-block:: python

   ax = xdata.render(
       splat={
           "genes": {
               "R": ["EPCAM", "KRT19"],
               "G": ["COL1A1", "DCN"],
               "B": ["PTPRC", "CD3D"],
           },
           "gains": [1, 1.5, 2],
       },
       bounds=b,
   )

Common splat options:

``genes``
    Gene, list of genes, or dictionary of channel name to gene list.

``gains``
    Per-channel brightness scaling.

``pixel_size_um``
    Raster pixel size in microns.

``sigma_um``
    Gaussian smoothing radius in microns.

``splat_alpha``
    Maximum opacity of the splat overlay when drawn over images.

Points Layer
------------

Use ``points`` to draw individual transcript molecules:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2},
       points={"genes": ["Prss3"], "s": 0.4, "color": "cyan"},
       bounds=b,
   )

Common point options:

``genes``
    Gene, list of genes, or gene-set dictionary.

``max_points``
    Maximum number of transcript markers to draw. Defaults to 100,000 in
    ``plot_points``. Set to ``None`` only for small ROIs.

``s``
    Marker size.

``color`` or ``palette``
    Point color controls.

``marker`` or ``markers``
    Marker style controls.

``assigned_only``
    For parquet/DataFrame-backed transcripts with ``cell_id``, defaults to
    assigned transcripts only. Pass ``False`` to include unassigned transcripts.

Cell Boundaries
---------------

Pass ``cells=True`` for default cell boundaries:

.. code-block:: python

   ax = xdata.render(
       splat=["EPCAM", "KRT19"],
       cells=True,
       bounds=b,
   )

Pass a dictionary for options:

.. code-block:: python

   ax = xdata.render(
       splat=["EPCAM", "KRT19"],
       cells={"edge_alpha": 0.1, "linewidth": 0.3},
       bounds=b,
   )

Because boundaries are drawn last, low ``edge_alpha`` values usually produce
the clearest overlays.

Bounds and ROIs
---------------

``bounds`` should be supplied as ``(xmin, xmax, ymin, ymax)`` in microns:

.. code-block:: python

   ax = xdata.render(bounds=(500, 1500, 900, 1900), splat=["EPCAM"])

If ``bounds`` is omitted, ``render`` uses the active ROI when one is set:

.. code-block:: python

   xdata.set_active_roi("Tumor")
   ax = xdata.render(image="DAPI", splat=["EPCAM"], cells=True)

If no active ROI is set, the full transcript frame is used.

When to Use Lower-Level Plotting
--------------------------------

Use ``render`` for most exploratory composite views. Use the lower-level
functions when you need to manually control artist order or return arrays:

.. code-block:: python

   ax = xdata.plot_image("DAPI", bounds=b)
   xdata.plot_splat(["EPCAM"], ax=ax, bounds=b)
   arr = xdata.plot_splat(["EPCAM"], bounds=b, return_array=True)

