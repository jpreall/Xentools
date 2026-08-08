API Reference
=============

This page lists the main public modules and functions. The API is still being
stabilized; prefer the documented high-level entry points where possible.

Top-Level API
-------------

.. autosummary::

   xentools.XenData
   xentools.AlignedImage
   xentools.CoordinateSystem
   xentools.TransformRegistry
   xentools.LazyTranscripts
   xentools.LazyBoundaryGeoDataFrame
   xentools.ROI
   xentools.ROICollection
   xentools.ROIGroup
   xentools.CoordinateSystem
   xentools.TransformRegistry
   xentools.settings
   xentools.plot_image
   xentools.plot_points
   xentools.plot_splat
   xentools.plot_binned_splat
   xentools.render
   xentools.points
   xentools.splat
   xentools.show_ome_tiff
   xentools.show_aligned_image
   xentools.plot_boundaries
   xentools.plot_cells
   xentools.build_niches
   xentools.evaluate_niche_k_values
   xentools.neighborhood_composition
   xentools.find_housekeeping_genes
   xentools.plot_housekeeping_diagnostics
   xentools.normalize_tp10k
   xentools.gene_panel_to_dataframe

Gene Sets Namespace
-------------------

.. autosummary::

   xentools.gene_sets.pick
   xentools.gene_sets.GeneSetPicker
   xentools.gene_sets.build_gene_set_h5
   xentools.gene_sets.load_curated_gene_sets
   xentools.gene_sets.prepare_for_xendata
   xentools.gene_sets.convert_gene_sets_species
   xentools.gene_sets.coverage_summary

XenData
-------

.. autoclass:: xentools.XenData
   :show-inheritance:

Plotting Namespace
------------------

.. autosummary::

   xentools.pl.splat
   xentools.pl.points
   xentools.pl.plot_splat
   xentools.pl.plot_binned_splat
   xentools.pl.plot_points
   xentools.pl.plot_image
   xentools.pl.render
   xentools.pl.show_ome_tiff
   xentools.pl.show_aligned_image
   xentools.pl.plot_boundaries
   xentools.pl.plot_cells
   xentools.pl.plot_housekeeping_diagnostics
   xentools.pl.rasterize
   xentools.pl.plot_binned_rgb

Analysis Namespace
------------------

.. autosummary::

   xentools.analysis.build_niches
   xentools.analysis.evaluate_niche_k_values
   xentools.analysis.neighborhood_composition
   xentools.analysis.find_housekeeping_genes
   xentools.analysis.build_spatial_graph
   xentools.analysis.normalize_tp10k
   xentools.analysis.create_binned_adata

Core Objects
------------

.. autosummary::

   xentools.ROI
   xentools.ROICollection
   xentools.ROIGroup
   xentools.LazyTranscripts
   xentools.LazyTranscripts.query
   xentools.LazyTranscripts.iter_tiles
   xentools.LazyTranscripts.diagnose_query
   xentools.LazyTranscripts.cache_info
   xentools.LazyTranscripts.clear_cache
   xentools.LazyBoundaryGeoDataFrame
