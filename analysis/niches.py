"""Spatial niche-composition helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import sparse

def _load_local_module(module_name, relative_path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .graph import build_spatial_graph
except ImportError:
    build_spatial_graph = _load_local_module(
        "_xentools_analysis_graph_for_niches",
        "analysis/graph.py",
    ).build_spatial_graph

__all__ = ["build_niches", "evaluate_niche_k_values"]

_K_NICHES_REQUIRED = object()


def build_niches(
    adata,
    *,
    xy_key: Optional[str] = "spatial",
    label_key: str = "graphclust",
    use_radius: Optional[float] = None,
    n_neighbors: int = 30,
    symmetrize: Literal["none", "max", "mean"] = "max",
    weight_scheme: Optional[Literal["binary", "inverse", "gaussian"]] = "binary",
    sigma: Optional[float] = None,
    normalize: Literal["none", "prop", "zscore"] = "prop",
    k_niches=_K_NICHES_REQUIRED,
    key_added: str = "niche",
):
    """
    Build spatial niche-composition features and cluster them into niche labels.

    The niche feature matrix describes the composition of each cell's spatial
    neighborhood after multiplying the spatial adjacency graph by a one-hot
    cell-type matrix. Rows are cells; columns are categories from
    ``adata.obs[label_key]``.

    Results are written to ``adata.obsm[f"{key_added}_X"]``,
    ``adata.uns[f"{key_added}_celltypes"]``, and
    ``adata.obsp["spatial_connectivities"]``. If ``k_niches`` is an integer,
    k-means niche labels are also written to
    ``adata.obs[f"{key_added}_k{k_niches}"]``.

    ``k_niches`` must be specified explicitly. Set ``k_niches=None`` to compute
    the continuous neighbor-composition feature matrix without assigning
    categorical niche labels.

    Parameters
    ----------
    normalize
        How to normalize the neighborhood-composition matrix before clustering.

        ``"none"``
            Use raw neighbor counts or weighted sums. This preserves neighborhood
            size/density, so dense regions can separate from sparse regions even
            if they have similar cell-type proportions.

        ``"prop"``
            Row-normalize each cell's neighborhood vector to sum to 1. This
            clusters cells by relative neighborhood composition and largely
            ignores how many neighbors contributed. This is the default and is
            usually the most interpretable choice for categorical tissue niches.

        ``"zscore"``
            Standardize each cell-type column across cells after computing raw
            neighborhood composition. This emphasizes enrichment/depletion of
            each cell type relative to its global abundance, which can help rare
            cell types influence niche clustering. It is more sensitive to noise
            and should be inspected carefully.
    """
    if k_niches is _K_NICHES_REQUIRED:
        raise ValueError(
            "k_niches must be specified explicitly. Use an integer such as "
            "k_niches=5 to assign categorical niche labels, or use "
            "k_niches=None to compute only the continuous niche-composition "
            "features without clustering."
        )
    if k_niches is not None and k_niches <= 1:
        raise ValueError("k_niches must be an integer greater than 1.")

    if xy_key is not None and xy_key in adata.obsm:
        coords = np.asarray(adata.obsm[xy_key], dtype=float)
        if coords.shape[1] > 2:
            coords = coords[:, :2]
    else:
        coords = np.asarray(adata.obs[["x", "y"]].values, dtype=float)

    graph = build_spatial_graph(
        coords,
        use_radius=use_radius,
        n_neighbors=n_neighbors,
        include_self=False,
        symmetrize=symmetrize,
        weight_scheme=weight_scheme,
        sigma=sigma,
    )
    adata.obsp["spatial_connectivities"] = graph

    celltypes = adata.obs[label_key].astype("category")
    celltype_names = list(celltypes.cat.categories)
    onehot_df = pd.get_dummies(celltypes, sparse=True)
    onehot = sparse.csr_matrix(onehot_df.values)

    composition = (graph @ onehot).astype(float)

    if normalize == "prop":
        row_sums = np.asarray(composition.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        composition = sparse.diags(1.0 / row_sums) @ composition
    elif normalize == "zscore":
        col_means = np.asarray(composition.mean(axis=0)).ravel()
        col_sqmeans = np.asarray(composition.multiply(composition).mean(axis=0)).ravel()
        col_stds = np.sqrt(np.maximum(col_sqmeans - col_means**2, 1e-12))
        composition = composition - sparse.csr_matrix(np.broadcast_to(col_means, composition.shape))
        invstd = 1.0 / np.where(col_stds == 0, 1.0, col_stds)
        composition = composition @ sparse.diags(invstd)

    adata.obsm[f"{key_added}_X"] = (
        composition.toarray() if sparse.issparse(composition) else composition
    )
    adata.uns[f"{key_added}_celltypes"] = celltype_names

    if k_niches is None:
        print(
            f"k_niches=None: wrote adata.obsm['{key_added}_X'], "
            f"adata.uns['{key_added}_celltypes'], and "
            "adata.obsp['spatial_connectivities']; no categorical niche labels were assigned."
        )
        return adata

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(k_niches), n_init="auto", random_state=0)
    labels = km.fit_predict(adata.obsm[f"{key_added}_X"])
    adata.obs[f"{key_added}_k{int(k_niches)}"] = pd.Categorical(labels.astype(str))

    return adata


def evaluate_niche_k_values(
    adata,
    k_values=range(3, 13),
    *,
    key: str = "niche_X",
    random_state: int = 0,
    n_init="auto",
    sample_size: Optional[int] = 5000,
    store_labels: bool = False,
    key_added: str = "niche",
    plot: bool = True,
    ax=None,
    figsize=(7, 4),
    show: bool = True,
    silhouette_ylim=None,
) -> pd.DataFrame:
    """
    Evaluate candidate numbers of niche clusters using an existing niche matrix.

    Parameters
    ----------
    adata
        AnnData object containing ``adata.obsm[key]``. Run
        ``build_niches(..., k_niches=None)`` first if needed.
    k_values
        Integer or iterable of integer k values to evaluate.
    key
        Key in ``adata.obsm`` containing the continuous niche-composition matrix.
        Default ``"niche_X"``.
    random_state
        Random seed passed to k-means.
    n_init
        Passed to ``sklearn.cluster.KMeans``.
    sample_size
        Maximum number of cells used for silhouette scoring. Set to ``None`` to
        use all cells. Inertia and niche-size summaries always use all cells.
    store_labels
        If True, store each tested clustering in
        ``adata.obs[f"{key_added}_k{k}"]``.
    key_added
        Prefix used when ``store_labels=True``.
    plot
        If True, plot inertia and silhouette as a function of k.
    ax
        Optional Matplotlib axes for the inertia curve. The silhouette curve is
        drawn on a secondary y-axis using ``ax.twinx()``. If None, a new figure
        and axes are created.
    figsize
        Figure size used when ``ax=None``.
    show
        If True, call ``matplotlib.pyplot.show()`` after plotting. Set to False
        for scripts/tests that should create the plot without displaying it.
    silhouette_ylim
        Optional ``(ymin, ymax)`` tuple for the silhouette y-axis. If None, the
        axis is autoscaled to the observed silhouette scores with modest padding.

    Returns
    -------
    pandas.DataFrame
        One row per k with inertia, silhouette score, and niche-size summaries.

        ``inertia``
            The k-means within-cluster sum of squared distances. Lower values
            mean cells are closer to their assigned niche centroids in
            niche-composition space. Inertia almost always decreases as k
            increases, so it is most useful for finding an "elbow" where adding
            more niches provides diminishing improvement. Do not choose the k
            with the lowest inertia by itself, because that will usually favor
            larger k.

        ``silhouette``
            A separation score comparing each cell's distance to its assigned
            niche versus the nearest alternative niche. Values range from -1 to
            1. Higher is generally better: values near 1 indicate well-separated
            clusters, values near 0 indicate overlapping/ambiguous clusters, and
            negative values suggest some cells may fit better in another
            cluster. Silhouette can favor simple, compact cluster structure and
            may under-rank biologically meaningful gradients or rare niches, so
            it should be interpreted alongside niche maps and cluster sizes.

        ``min_size``, ``max_size``, ``median_size``, ``min_fraction``,
        ``max_fraction``, and ``empty_clusters``
            Cluster-balance summaries. These are useful for identifying k values
            that create tiny, unstable, or effectively unused niches.
    """
    if key not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{key}'] not found. Run build_niches(..., k_niches=None) "
            "or build_niches(..., k_niches=<int>) before evaluating k values."
        )

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix = np.asarray(adata.obsm[key], dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"adata.obsm['{key}'] must be a 2D matrix.")

    n_obs = matrix.shape[0]
    rows = []
    rng = np.random.default_rng(random_state)

    if isinstance(k_values, int):
        k_values = [k_values]

    for k in k_values:
        k = int(k)
        if k <= 1:
            raise ValueError("All k_values must be integers greater than 1.")
        if k >= n_obs:
            raise ValueError("All k_values must be smaller than the number of observations.")

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(matrix)
        counts = np.bincount(labels, minlength=k)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            silhouette = np.nan
        elif sample_size is None or n_obs <= sample_size:
            silhouette = float(silhouette_score(matrix, labels))
        else:
            sample_idx = rng.choice(n_obs, size=int(sample_size), replace=False)
            sample_labels = labels[sample_idx]
            sample_unique = np.unique(sample_labels)
            if len(sample_unique) < 2 or len(sample_unique) >= len(sample_labels):
                silhouette = np.nan
            else:
                silhouette = float(silhouette_score(matrix[sample_idx], sample_labels))

        if store_labels:
            adata.obs[f"{key_added}_k{k}"] = pd.Categorical(labels.astype(str))

        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": silhouette,
                "min_size": int(counts.min()),
                "max_size": int(counts.max()),
                "median_size": float(np.median(counts)),
                "min_fraction": float(counts.min() / n_obs),
                "max_fraction": float(counts.max() / n_obs),
                "empty_clusters": int((counts == 0).sum()),
            }
        )

    results = pd.DataFrame(rows)

    if plot:
        _plot_niche_k_evaluation(
            results,
            ax=ax,
            figsize=figsize,
            show=show,
            silhouette_ylim=silhouette_ylim,
        )

    return results


def _plot_niche_k_evaluation(
    results: pd.DataFrame,
    *,
    ax=None,
    figsize=(7, 4),
    show=True,
    silhouette_ylim=None,
):
    """Plot inertia and silhouette from ``evaluate_niche_k_values`` results."""
    import matplotlib.pyplot as plt

    if results.empty:
        raise ValueError("Cannot plot empty niche k evaluation results.")

    plot_df = results.sort_values("k")

    if ax is None:
        fig, ax_inertia = plt.subplots(figsize=figsize)
    else:
        ax_inertia = ax
        fig = ax_inertia.figure

    ax_silhouette = ax_inertia.twinx()

    inertia_line = ax_inertia.plot(
        plot_df["k"],
        plot_df["inertia"],
        marker="o",
        color="#375a7f",
        label="Inertia",
    )[0]
    silhouette_line = ax_silhouette.plot(
        plot_df["k"],
        plot_df["silhouette"],
        marker="o",
        color="#c44e52",
        label="Silhouette",
    )[0]

    ax_inertia.set_xlabel("Number of niches (k)")
    ax_inertia.set_ylabel("K-means inertia", color=inertia_line.get_color())
    ax_silhouette.set_ylabel("Silhouette score", color=silhouette_line.get_color())
    ax_inertia.tick_params(axis="y", labelcolor=inertia_line.get_color())
    ax_silhouette.tick_params(axis="y", labelcolor=silhouette_line.get_color())
    if silhouette_ylim is not None:
        ax_silhouette.set_ylim(*silhouette_ylim)
    else:
        finite = plot_df["silhouette"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            ymin = float(finite.min())
            ymax = float(finite.max())
            span = ymax - ymin
            pad = max(0.02, span * 0.15)
            ax_silhouette.set_ylim(max(-1.0, ymin - pad), min(1.0, ymax + pad))
    ax_inertia.set_xticks(plot_df["k"])
    ax_inertia.set_title("Niche k evaluation")
    ax_inertia.grid(True, axis="both", alpha=0.25)

    ax_inertia.legend(
        [inertia_line, silhouette_line],
        ["Inertia", "Silhouette"],
        loc="best",
        frameon=False,
    )

    fig.tight_layout()

    if show:
        plt.show()

    return ax_inertia, ax_silhouette
