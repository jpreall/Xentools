Analysis
========

xentools analysis helpers are intentionally small and ROI-aware. They are meant
to answer common first-pass questions about a tissue feature or TMA core without
requiring users to split the original Xenium run into separate Analyzer outputs.

Logical ROI Groups
------------------

Related tissue pieces can be grouped without deleting or modifying their
original ROIs:

.. code-block:: python

   liver = xdata.create_roi_group(
       "all_liver",
       ["liver_piece_1", "liver_piece_2"],
       metadata={"tissue": "liver"},
   )

The group retains its member names and exposes their geometric union. Existing
ROI-aware operations accept the group name, including activation, cropping,
and housekeeping analysis:

.. code-block:: python

   xdata.set_active_roi("all_liver")
   liver_only = xdata.subset_to_roi("all_liver", inplace=False)

   ranking = xdata.find_housekeeping_genes(
       rois=["all_liver", "all_kidney"],
   )

Disjoint pieces are represented as a ``MultiPolygon`` and are tested by exact
polygon containment. Cropped image bounds may still include the slide space
between distant pieces. The union and its membership metadata are GeoJSON-ready;
writing a cropped group as a Xenium Explorer dataset remains a separate export
workflow.

Housekeeping Gene Discovery
---------------------------

``find_housekeeping_genes`` uses raw counts to rank genes that are well
detected, moderately to highly expressed, and stable both within and between
tissue pieces. It does not apply library-size normalization. Within-tissue
stability is measured by Poisson-adjusted Pearson dispersion, while
between-tissue consistency uses variance of log-transformed tissue means. For a
``XenData`` object, it uses all registered ROIs by default:

Raw counts are resolved from ``adata.X`` by default. If ``adata.X`` has been
normalized or transformed, ``layer="auto"`` searches count-like layers named
``counts``, ``raw_counts``, or ``raw`` (then other layers), followed by a
compatible ``adata.raw``. The selected source is recorded in
``ranking.attrs["count_source"]``. Before modifying ``adata.X``, preserve counts
with ``xdata.adata.layers["counts"] = xdata.adata.X.copy()``.

.. code-block:: python

   ranking = xdata.find_housekeeping_genes(n_genes=20)
   genes = ranking.index[ranking["selected"]].tolist()

Review the expression/stability tradeoff and consistency across tissues with:

.. code-block:: python

   axes = xdata.plot_housekeeping_diagnostics(ranking)

The first panel shows all genes in expression-versus-variance space, the second
shows mean expression of the selected genes in every ROI, and the third shows
their detection range across ROIs. Selected genes should sit toward the
high-expression, low-variance region while showing even heatmap colors and
consistently high detection.

Zero-count genes are omitted from the log-scaled selection scatter. To suppress
additional low-count rejected genes without changing the ranking, set
``min_plot_mean_count``, for example
``xdata.plot_housekeeping_diagnostics(ranking, min_plot_mean_count=0.05)``.

If no genes pass the selection criteria, the plotting function reports how
many genes passed each filter and suggests which thresholds to revisit. You can
still inspect borderline genes without changing the ranking:

.. code-block:: python

   borderline = ranking.sort_values("score", ascending=False).index[:10]
   xdata.plot_housekeeping_diagnostics(ranking, genes=borderline)

Choose particular TMA cores by name with ``rois=["core_1", "core_2"]``. The
returned table includes per-ROI means and detection rates plus separate within-
and between-ROI variance, making it easy to review why each gene ranked well.

For an AnnData object, define tissue pieces through an observation column:

.. code-block:: python

   ranking = xentools.find_housekeeping_genes(
       adata,
       groupby="tissue",
       n_genes=15,
   )

Cell Neighborhood Composition
-----------------------------

``neighborhood_composition`` computes the composition of each cell's local
spatial neighborhood by cluster, cell type, or any categorical column in
``xdata.adata.obs``.

When called on a :class:`xentools.XenData` object, it defaults to the active
ROI:

.. code-block:: python

   xdata.set_active_roi("Core_1")

   comp = xdata.neighborhood_composition(
       label_key="Cluster",
       n_neighbors=30,
       normalize="prop",
   )

The returned DataFrame is indexed by cell ID. Columns are the categories from
``label_key``. With ``normalize="prop"``, each row gives the fraction of that
cell's neighbors belonging to each category. The ``"_neighbor_total"`` column
stores the number or total weight of neighbors before normalization.

To ignore the active ROI and use the full object:

.. code-block:: python

   comp = xdata.neighborhood_composition(
       label_key="Cluster",
       roi=None,
   )

To use a specific named ROI:

.. code-block:: python

   comp = xdata.neighborhood_composition(
       label_key="Cluster",
       roi="Tumor",
   )

Neighborhood Definition
-----------------------

The default neighborhood is k-nearest neighbors:

.. code-block:: python

   comp = xdata.neighborhood_composition(n_neighbors=30)

Alternatively, use a fixed radius in microns:

.. code-block:: python

   comp = xdata.neighborhood_composition(use_radius=50)

The helper uses the same spatial graph construction code as
``build_niches()``. For most ROI-level exploratory work, start with
``normalize="prop"`` so the output reflects neighborhood composition rather
than local cell density.

Niche Visualization
-------------------

After running ``build_niches()``, use ``niche_heatmap`` to inspect the average
neighborhood composition of each niche label:

.. code-block:: python

   xentools.build_niches(
       xdata.adata,
       label_key="Cluster",
       n_neighbors=30,
       k_niches=5,
       key_added="niche",
   )

   ax = xdata.niche_heatmap(key_added="niche")

The heatmap rows are niche labels such as ``"niche_k5"`` and columns are the
cell labels used to build the neighborhood-composition matrix.

To map niche labels spatially:

.. code-block:: python

   ax = xdata.niche_map(key_added="niche")

``niche_map`` defaults to the active ROI when one is set. Use ``roi=None`` to
plot all cells:

.. code-block:: python

   ax = xdata.niche_map(key_added="niche", roi=None)

You can also map a continuous neighborhood-composition column by passing one of
the labels stored in ``adata.uns["niche_celltypes"]``:

.. code-block:: python

   ax = xdata.niche_map(color="Tumor", key_added="niche")
