from __future__ import annotations

import matplotlib.axes


def _build_small_niches(xdata):
    import xentools

    xentools.build_niches(
        xdata.adata,
        label_key="Cluster",
        n_neighbors=3,
        k_niches=3,
        key_added="niche",
    )


def test_niche_heatmap_plots_mean_composition_by_niche(xdata):
    _build_small_niches(xdata)

    ax = xdata.niche_heatmap(key_added="niche")

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
    assert ax.get_ylabel().startswith("niche_k")


def test_niche_map_plots_categorical_niche_labels(xdata):
    _build_small_niches(xdata)

    ax = xdata.niche_map(key_added="niche", roi=None, s=2)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert ax.get_legend() is not None


def test_niche_map_uses_native_global_coordinates(xdata):
    _build_small_niches(xdata)
    ax = xdata.niche_map(key_added="niche", roi=None, use_boundaries=False, s=2)

    plotted = ax.collections[0].get_offsets()
    coords = xdata.adata.obsm["spatial"]
    assert plotted[0, 0] == coords[0, 0]
    assert plotted[0, 1] == coords[0, 1]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]


def test_niche_map_plots_continuous_composition_column(xdata):
    _build_small_niches(xdata)
    first_celltype = xdata.adata.uns["niche_celltypes"][0]

    ax = xdata.niche_map(color=first_celltype, key_added="niche", roi=None, s=2)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.collections) == 1
    assert len(ax.figure.axes) == 2


def test_niche_map_uses_edge_free_cell_polygons_by_default(xdata):
    _build_small_niches(xdata)

    ax = xdata.niche_map(key_added="niche", roi=None)
    collection = ax.collections[0]

    assert len(collection.get_paths()) > 0
    assert collection.get_linewidths()[0] == 0
    assert collection.get_edgecolors()[0, 3] == 0


def test_niche_heatmap_accepts_neighborhood_dataframe(xdata):
    import xentools

    comp = xdata.neighborhood_composition(label_key="Cluster", roi=None, n_neighbors=3)

    ax = xentools.pl.niche_heatmap(comp)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
