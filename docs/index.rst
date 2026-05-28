xentools
========

``xentools`` is a lightweight toolkit for quickly reading, inspecting, and
visualizing 10x Xenium-style spatial transcriptomics outputs.

The package is intended for the early exploratory phase after a Xenium or
Atera run finishes: load the output directory, inspect the cell-level matrix,
plot transcript splats, overlay boundaries, and work with ROIs without first
converting the dataset into a larger specialized container.

.. note::

   xentools is under active development. Core loading and plotting workflows are
   usable, while Xenium Explorer export and full write-back compatibility should
   still be treated as experimental.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   concepts
   analysis
   plotting
   rendering
   gene_sets
   lazy_transcripts
   api

Key Ideas
---------

Fast startup
    Transcript and boundary data can be represented lazily so large output
    directories become usable quickly.

Direct Xenium bundle access
    The package reads directly from standard files such as
    ``transcripts.parquet``, ``transcripts.zarr.zip``, ``cells.zarr.zip``,
    ``cell_feature_matrix.zarr.zip``, and morphology OME-TIFF files.

Matplotlib-first visualization
    Plotting functions return normal matplotlib axes, which makes it easy to
    layer transcript splats, DAPI/protein images, boundaries, and annotations.

ROI-aware workflows
    GeoJSON or manually defined ROIs can be stashed in a ``XenData`` object and
    used to focus plotting or subset data.
