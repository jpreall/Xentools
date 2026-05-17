Quickstart
==========

Load a Dataset
--------------

Create a :class:`xentools.XenData` object from a Xenium-style output folder:

.. code-block:: python

   import xentools

   xdata = xentools.XenData("/path/to/xe_outs")
   xdata

``XenData`` reads the cell-feature matrix, cell metadata, image metadata,
cluster assignments when present, transcript metadata, and boundary metadata.
Large transcript or boundary tables may remain lazy until queried.

Plot Transcript Splats
----------------------

Plot one or more genes as smoothed transcript-density channels:

.. code-block:: python

   ax = xdata.splat(["CCND1", "NHERF1", "OR4F17"])

Use a dictionary to combine gene sets into channels:

.. code-block:: python

   ax = xdata.splat(
       {
           "Epithelial": ["EPCAM", "KRT19"],
           "Stroma": ["COL1A1", "DCN"],
           "Immune": ["PTPRC", "CD3D"],
       },
       gains=(1.0, 1.5, 2.0),
   )

Return the rendered array instead of the axes:

.. code-block:: python

   arr = xdata.splat(["EPCAM", "KRT19"], return_array=True)

Show Images
-----------

Display a morphology or protein image channel:

.. code-block:: python

   ax = xdata.show_image("DAPI", level=3)

Overlay Boundaries
------------------

Cell and nucleus boundaries are lazy by default. Overlay them on an existing
axes:

.. code-block:: python

   ax = xdata.splat(["EPCAM", "KRT19"])
   xdata.plot_boundaries(kind="cell", ax=ax, color_by="Cluster")

Plot Cells by Expression
------------------------

Fill cell polygons by expression of one gene or the summed expression of a gene
set:

.. code-block:: python

   ax = xdata.plot_cells(genes="EPCAM", cmap="magma")
   ax = xdata.plot_cells(genes=["EPCAM", "KRT19"], cmap="magma")

Work with ROIs
--------------

Load a GeoJSON ROI file at initialization:

.. code-block:: python

   xdata = xentools.XenData(
       "/path/to/xe_outs",
       roi_file="/path/to/annotation.geojson",
   )

Set the active ROI and plot within that region:

.. code-block:: python

   xdata.set_active_roi("Tumor")
   ax = xdata.splat(["EPCAM", "KRT19"])
   xdata.plot_boundaries(kind="cell", ax=ax)

Create a new ROI manually:

.. code-block:: python

   roi = xentools.ROI.from_bounds(2000, 12000, 2000, 4000, name="custom")
   xdata.rois["custom"] = roi
   xdata.set_active_roi("custom")
