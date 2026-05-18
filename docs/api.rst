API Reference
=============

This page lists the main public modules and functions. The API is still being
stabilized; prefer the documented high-level entry points where possible.

Top-Level API
-------------

.. autosummary::

   xentools.XenData
   xentools.LazyTranscripts
   xentools.LazyBoundaryGeoDataFrame
   xentools.ROI
   xentools.ROICollection
   xentools.splat
   xentools.show_ome_tiff
   xentools.plot_boundaries
   xentools.plot_cells
   xentools.build_niches
   xentools.evaluate_niche_k_values
   xentools.neighborhood_composition
   xentools.normalize_tp10k

XenData
-------

.. autoclass:: xentools.XenData
   :show-inheritance:

Plotting Namespace
------------------

.. autosummary::

   xentools.pl.splat
   xentools.pl.show_ome_tiff
   xentools.pl.plot_boundaries
   xentools.pl.plot_cells
   xentools.pl.rasterize
   xentools.pl.plot_binned_rgb

Analysis Namespace
------------------

.. autosummary::

   xentools.analysis.build_niches
   xentools.analysis.evaluate_niche_k_values
   xentools.analysis.neighborhood_composition
   xentools.analysis.build_spatial_graph
   xentools.analysis.normalize_tp10k
   xentools.analysis.create_binned_adata

Core Objects
------------

.. autosummary::

   xentools.ROI
   xentools.ROICollection
   xentools.LazyTranscripts
   xentools.LazyTranscripts.query
   xentools.LazyTranscripts.iter_tiles
   xentools.LazyTranscripts.diagnose_query
   xentools.LazyTranscripts.cache_info
   xentools.LazyTranscripts.clear_cache
   xentools.LazyBoundaryGeoDataFrame
