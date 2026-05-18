Analysis
========

xentools analysis helpers are intentionally small and ROI-aware. They are meant
to answer common first-pass questions about a tissue feature or TMA core without
requiring users to split the original Xenium run into separate Analyzer outputs.

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
