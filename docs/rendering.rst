Composite Rendering
===================

``render`` is xentools' high-level compositor for exploratory spatial plots.
It combines morphology/protein images, transcript-density splats,
single-transcript points, and cell boundaries in one ROI-focused view.

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
       save="figures/dapi_splat_points_cells.png",
       save_kwargs={"dpi": 300},
   )

API Reference
-------------

.. function:: xdata.render(*, image=None, images=None, splat=None, binned_splat=None, points=None, cells=None, bounds=None, ax=None, figsize=(8, 8), dpi=None, level=None, background="black", show_axis=False, title=None, legend=True, legend_loc="outside right", legend_title=None, legend_max_items=30, save=None, save_kwargs=None)

   Render a composite spatial view from common XenData layer types.

   The module-level form is ``xentools.pl.render(xdata, ...)`` or
   ``xentools.render(xdata, ...)``. The ``xdata.render(...)`` method is the
   most convenient form for interactive use.

Parameters
~~~~~~~~~~

``image`` : str, dict, list, tuple, or None, default ``None``
    Single image-layer specification. A string is interpreted as an image
    channel, for example ``image="DAPI"``. A dictionary with a ``"channel"``
    key can include image display options, for example
    ``image={"channel": "DAPI", "level": 2, "alpha": 0.5}``.
    Lists and tuples are also accepted and are normalized like ``images``.

``images`` : dict, list, tuple, or None, default ``None``
    One or more image-layer specifications. A dictionary whose keys are channel
    names is the recommended form for multiple images, for example
    ``images={"DAPI": {"level": 2}, "18S": {"color": "green"}}``. The first
    image is treated as the background. Later images are converted to
    transparent signal overlays by default.

``splat`` : str, sequence, dict, or None, default ``None``
    Transcript-density layer specification. A string, list, tuple, or
    channel-to-gene dictionary is interpreted as the ``genes`` argument to
    :meth:`~xentools.XenData.plot_splat`. A dictionary containing recognized
    splat option keys is interpreted as a full options dictionary, for example
    ``splat={"genes": ["EPCAM", "KRT19"], "gains": [2, 1]}``.

``binned_splat`` : str, sequence, dict, or None, default ``None``
    Binned transcript-density layer specification. This uses
    ``xdata.binned_adata`` via :meth:`~xentools.XenData.plot_binned_splat`,
    so run ``xdata.create_binned_adata()`` first. It accepts the same simple
    gene forms as ``splat``.

``points`` : str, sequence, dict, or None, default ``None``
    Single-transcript point layer specification. Simple gene forms are
    accepted, or pass a dictionary containing ``"genes"`` plus options for
    :meth:`~xentools.XenData.plot_points`, for example
    ``points={"genes": ["CD3D"], "s": 0.5, "max_points": 20000}``.

``cells`` : bool, dict, or None, default ``None``
    Cell-boundary layer specification. If ``True``, draw default cell
    boundaries. If a dictionary, pass those options to
    :meth:`~xentools.XenData.plot_cells`, for example
    ``cells={"edge_alpha": 0.1, "linewidth": 0.3}``. If ``None`` or ``False``,
    omit cell boundaries.

``bounds`` : tuple or None, default ``None``
    Spatial window as ``(xmin, xmax, ymin, ymax)`` in microns. If omitted,
    ``render`` uses the active ROI when one is set. If no active ROI is set,
    it uses the full transcript frame.

``ax`` : matplotlib.axes.Axes or None, default ``None``
    Existing Matplotlib axes to draw into. If omitted, ``render`` creates a new
    figure and axes.

``figsize`` : tuple, default ``(8, 8)``
    Figure size in inches when ``ax`` is omitted.

``dpi`` : int or None, default ``None``
    Figure DPI when ``ax`` is omitted.

``level`` : int or None, default ``None``
    Default image pyramid level applied to image layers that do not specify
    their own ``level``. Higher levels are lower resolution and usually faster.

``background`` : str, default ``"black"``
    Figure and axes background color when ``ax`` is omitted.

``show_axis`` : bool, default ``False``
    Whether to show axis ticks and labels.

``title`` : str or None, default ``None``
    Optional axes title.

``legend`` : bool or str, default ``True``
    Whether to draw a single combined legend for all rendered layers. This is
    enabled by default because ``render`` is intended to make figure-ready
    composites. Use ``False`` or string values such as ``"off"`` or
    ``"none"`` to suppress it. When enabled, lower-level categorical legends
    from ``plot_splat``, ``plot_points``, and ``plot_cells`` are suppressed so
    the final plot has one coordinated legend.

``legend_loc`` : str, default ``"outside right"``
    Combined legend placement. Supported outside placements are
    ``"outside right"``, ``"outside left"``, ``"outside bottom"``, and
    ``"outside top"``. Other values are passed through to Matplotlib as normal
    legend locations, for example ``"upper right"``.

``legend_title`` : str or None, default ``None``
    Optional title for the combined legend.

``legend_max_items`` : int, default ``30``
    Maximum number of entries shown within each categorical legend section. If
    a section has more entries, ``render`` adds a compact ``"... N more"``
    line.

``save`` : str, path-like, or None, default ``None``
    Optional output path. The completed composite is saved after all layers are
    drawn. Parent directories are created automatically.

``save_kwargs`` : dict or None, default ``None``
    Extra keyword arguments forwarded to Matplotlib's ``Figure.savefig``.
    xentools defaults to ``bbox_inches="tight"`` unless overridden.

Returns
~~~~~~~

``matplotlib.axes.Axes``
    The axes containing the completed composite.

Layer Option Reference
----------------------

Image Options
~~~~~~~~~~~~~

Image options are passed inside ``image`` or ``images`` dictionaries.

``channel`` : str
    Image channel name. Required when using ``image={...}``; inferred from the
    dictionary key when using ``images={"DAPI": {...}}``.

``level`` : int, optional
    Pyramid level to display. Higher levels are lower resolution and faster.

``alpha`` : float, default depends on layer position
    For the first/background image, values below 1 dim brightness without
    turning black pixels gray. For later image overlays, controls maximum
    signal opacity. The first image defaults to ``1.0``; later image overlays
    default to ``0.7``.

``color`` : str or RGB-like, optional
    Colorize a grayscale image layer, useful for protein or fluorescence
    channels. If supplied, xentools creates a black-to-color colormap.

``cmap`` : str or matplotlib colormap, optional
    Colormap passed to :meth:`~xentools.XenData.plot_image`. Ignored when
    ``color`` is supplied unless explicitly overridden by the caller.

``vmin``, ``vmax`` : float, optional
    Display intensity limits.

``clip_percentile`` : float, optional
    Percentile-based upper intensity clipping used by ``plot_image``.

``blend`` : {"normal", "screen", "add", "additive", "transparent"}, optional
    Blend behavior for the image layer. The first image defaults to
    ``"normal"``. Later images default to ``"screen"`` so dark pixels do not
    hide the layers underneath.

Splat Options
~~~~~~~~~~~~~

Splat options are passed inside the ``splat`` dictionary.

``genes`` : str, sequence, dict, or None
    Gene or genes to rasterize. A dictionary maps channel names to gene lists,
    for example ``{"R": ["EPCAM", "KRT19"], "G": ["COL1A1"], "B": ["PTPRC"]}``.

``gains`` : float or sequence, default inherited from ``plot_splat``
    Brightness scaling. A scalar applies to all channels. A sequence applies
    per channel.

``pixel_size_um`` : float, optional
    Raster pixel size in microns.

``sigma_um`` : float, optional
    Gaussian smoothing radius in microns.

``smooth`` : bool, optional
    Whether to smooth the rasterized transcript counts.

``global_norm`` : bool, optional
    If ``True``, normalize all channels together. If ``False``, normalize each
    channel independently.

``splat_alpha`` : float, default ``0.8`` in ``render``
    Maximum opacity of the splat overlay when drawn over other layers.

``show_legend`` : bool, default disabled when ``legend=True``
    Whether to draw the lower-level splat legend. In normal composites,
    ``render`` disables this and represents the splat in the combined legend.

Binned Splat Options
~~~~~~~~~~~~~~~~~~~~

Binned splat options are passed inside the ``binned_splat`` dictionary. This
layer uses ``xdata.binned_adata`` instead of transcript coordinates, so run
``xdata.create_binned_adata()`` first.

``genes`` : str, sequence, dict, or None
    Gene or genes to draw from the binned matrix. A dictionary maps channel
    names to gene lists, as with ``splat``.

``gains`` : float or sequence, default inherited from ``plot_binned_splat``
    Brightness scaling. A scalar applies to all channels. A sequence applies
    per channel.

``sigma_um`` : float, optional
    Gaussian smoothing radius in microns applied to the binned image.

``smooth`` : bool, optional
    Whether to smooth the binned counts.

``global_norm`` : bool, optional
    If ``True``, normalize all channels together. If ``False``, normalize each
    channel independently.

``splat_alpha`` : float, default ``0.8`` in ``render``
    Maximum opacity of the binned splat overlay when drawn over other layers.

Point Options
~~~~~~~~~~~~~

Point options are passed inside the ``points`` dictionary.

``genes`` : str, sequence, dict, or None
    Gene or genes to draw as individual transcript markers. A gene-set
    dictionary can be used to assign genes to named color groups.

``max_points`` : int or None, optional
    Maximum number of transcript markers to draw. ``plot_points`` defaults to
    100,000. If more transcripts are selected, a random subset is shown. Use
    ``None`` only for small ROIs.

``s`` : float, optional
    Marker size passed to Matplotlib ``scatter``.

``alpha`` : float, optional
    Marker opacity.

``color`` : color-like, optional
    Single color for all points.

``palette`` : dict or sequence, optional
    Gene or gene-set color mapping.

``marker`` or ``markers`` : str or dict, optional
    Marker style. By default, circular markers are drawn without marker edges
    to reduce visual noise.

``assigned_only`` : bool, optional
    For parquet/DataFrame-backed transcripts with ``cell_id``, defaults to
    assigned transcripts only. Pass ``False`` to include unassigned
    transcripts.

Cell Options
~~~~~~~~~~~~

Cell options are passed inside the ``cells`` dictionary.

``color_by`` : str, optional
    Observation column used to color cells.

``genes`` : str or sequence, optional
    Gene or genes used to color cells by expression. Multiple genes are summed.

``facecolor`` : color-like, default ``"none"`` in ``render``
    Cell fill color when not coloring by metadata or expression.

``face_alpha`` : float, optional
    Cell fill opacity.

``edgecolor`` : color-like, optional
    Boundary edge color.

``edge_alpha`` : float, default ``0.2`` in ``render``
    Boundary edge opacity.

``linewidth`` : float, optional
    Boundary edge width.

``cmap`` : str or matplotlib colormap, optional
    Colormap for continuous cell coloring.

``show_legend`` : bool, optional
    Whether to show the lower-level categorical cell legend. In normal
    composites, ``render`` disables this and represents categorical cell
    colors in the combined legend. Continuous cell-expression coloring still
    uses the lower-level colorbar.

Combined Legends
~~~~~~~~~~~~~~~~

By default, ``render`` creates one outside legend that summarizes all visible
layers:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.35},
       splat={"genes": ["C7", "Epcam", "Tagln"], "gains": [3, 3, 3]},
       points={"genes": ["Prss3"], "s": 0.4, "color": "cyan"},
       cells={"color_by": "Cluster", "face_alpha": 0.2, "edge_alpha": 0.1},
       bounds=b,
   )

The legend is organized into sections such as ``Images``, ``Transcript
density``, ``Transcript points``, and ``Cells: Cluster``. Image legend swatches
show the channel color at full strength even when the plotted image is dimmed
with ``alpha``; the goal is to identify the layer, not to reproduce the exact
blend opacity. This avoids the common failure mode where a cell-boundary legend
replaces a gene-density legend, or where multiple legends compete for space
inside the plot.

Move or suppress the combined legend with ``legend_loc`` and ``legend``:

.. code-block:: python

   ax = xdata.render(..., legend_loc="outside bottom")
   ax = xdata.render(..., legend=False)

Usage Patterns
--------------

Single Image
~~~~~~~~~~~~

Use ``image`` for one channel:

.. code-block:: python

   ax = xdata.render(image="DAPI", bounds=b)

Pass a dictionary when you need options:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.35},
       bounds=b,
   )

Multiple Images
~~~~~~~~~~~~~~~

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

Transcript Splats
~~~~~~~~~~~~~~~~~

Use a gene list:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       splat=["C7", "Epcam", "Tagln"],
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

Transcript Points
~~~~~~~~~~~~~~~~~

Use ``points`` to draw individual transcript molecules:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2},
       points={"genes": ["Prss3"], "s": 0.4, "color": "cyan"},
       bounds=b,
   )

Cell Boundaries
~~~~~~~~~~~~~~~

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

Saving Figures
--------------

``save`` writes the completed composite after all layers are drawn:

.. code-block:: python

   ax = xdata.render(
       image={"channel": "DAPI", "level": 2, "alpha": 0.4},
       splat={"genes": ["EPCAM", "KRT19"], "gains": [2, 2]},
       cells=True,
       bounds=b,
       save="figures/tumor_region.png",
       save_kwargs={"dpi": 300},
   )

Parent directories are created automatically. ``save_kwargs`` is forwarded to
Matplotlib's ``Figure.savefig`` and defaults to ``bbox_inches="tight"``.

When to Use Lower-Level Plotting
--------------------------------

Use ``render`` for most exploratory composite views. Use the lower-level
functions when you need to manually control artist order or return arrays:

.. code-block:: python

   ax = xdata.plot_image("DAPI", bounds=b)
   xdata.plot_splat(["EPCAM"], ax=ax, bounds=b)
   arr = xdata.plot_splat(["EPCAM"], bounds=b, return_array=True)
