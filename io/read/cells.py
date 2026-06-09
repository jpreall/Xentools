"""Cell feature matrix, gene panel, and cluster readers."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd


def _load_local_module(module_name, relative_path):
    """Load a repository module by file path when imported outside a package."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .zarr import _read_analysis_zarr, _read_zarr_adata
except ImportError:
    _zarr_read = _load_local_module("_xentools_io_read_zarr_for_cells", "zarr.py")
    _read_analysis_zarr = _zarr_read._read_analysis_zarr
    _read_zarr_adata = _zarr_read._read_zarr_adata

try:
    from ...analysis.normalization import normalize_tp10k
    from ...utils.metadata import _annotate_adata_count_metrics
except ImportError:
    normalize_tp10k = _load_local_module(
        "_xentools_analysis_normalization_for_cells",
        os.path.join("..", "..", "analysis", "normalization.py"),
    ).normalize_tp10k
    _annotate_adata_count_metrics = _load_local_module(
        "_xentools_utils_metadata_for_cells",
        os.path.join("..", "..", "utils", "metadata.py"),
    )._annotate_adata_count_metrics


__all__ = [
    "CellMatrixLoadResult",
    "_make_gene_panel_df",
    "load_xenium_cell_matrix",
    "read_classic_analysis_clusters",
    "read_xen_panel",
    "read_xenium_to_anndata",
]


@dataclass
class CellMatrixLoadResult:
    """Cell-level data loaded from a Xenium output folder."""

    adata: object
    clusters: pd.DataFrame
    gene_panel: dict | None


def read_xen_panel(gene_panel_file):
    """Read a Xenium gene panel JSON file."""
    with open(gene_panel_file) as f:
        return json.load(f)


def _make_gene_panel_df(gene_panel_dict):
    """Convert a Xenium gene panel dictionary to a DataFrame."""
    out = {}
    for target in gene_panel_dict["payload"]["targets"]:
        gene_id = None
        if "id" in target["type"]["data"]:
            gene_id = target["type"]["data"]["id"]
        gene_name = target["type"]["data"]["name"]
        panel_identity = target["source"]["identity"]

        out[gene_name] = {
            "Gene_ID": gene_id,
            "Description": target["type"]["descriptor"],
            "Coverage": target["info"]["gene_coverage"],
            "Panel_ID": panel_identity["design_id"],
            "Panel_Name": panel_identity["name"],
        }
        if "version" in panel_identity:
            out[gene_name]["Panel_Version"] = panel_identity["version"]

    return pd.DataFrame.from_dict(out, orient="index")


def read_classic_analysis_clusters(xenium_folder, verbose=True):
    """Read the classic graph-clustering CSV used by older Xenium outputs."""
    clusters_file = os.path.join(
        xenium_folder,
        "analysis/clustering/gene_expression_graphclust/clusters.csv",
    )
    if verbose:
        print("Reading Clusters")
    if os.path.exists(clusters_file):
        clusters = pd.read_csv(clusters_file, index_col=0)
        clusters.index = clusters.index.astype("str")
        clusters["Cluster"] = clusters["Cluster"].astype("str")
        return clusters
    return pd.DataFrame(columns=["Cluster"])


def _read_all_classic_cluster_annotations(xenium_folder, obs_names, verbose=True):
    if verbose:
        print("Importing Xenium Ranger cluster annotations")
    cluster_files = glob.glob(os.path.join(xenium_folder, "analysis/clustering/*/*csv"))
    result = pd.DataFrame()
    for path in cluster_files:
        cluster_name = os.path.basename(os.path.dirname(path)).replace("gene_expression_", "")
        cluster_df = pd.read_csv(path, index_col=0)
        cluster_df.index = cluster_df.index.astype("str")
        clusters = cluster_df["Cluster"].rename(cluster_name).astype("str").astype("category")
        result = pd.concat([result, clusters], axis=1)
    return result.reindex(obs_names)


def read_xenium_to_anndata(xenium_output_folder, include_non_gene_features=False, verbose=True):
    """Read a classic Xenium cell feature matrix into AnnData."""
    import scanpy as sc

    xdir = xenium_output_folder
    try:
        adata = sc.read_10x_h5(f"{xdir}/cell_feature_matrix.h5")
    except FileNotFoundError:
        if os.path.exists(f"{xdir}/cell_feature_matrix/"):
            adata = sc.read_10x_mtx(f"{xdir}/cell_feature_matrix/", gex_only=False)
        else:
            raise

    feature_names = pd.Index(adata.var_names.astype(str))
    total_mask = feature_names == "Total transcripts"
    if total_mask.any():
        total_idx = np.flatnonzero(total_mask)[0]
        total_values = adata.X[:, total_idx]
        if hasattr(total_values, "toarray"):
            total_values = total_values.toarray()
        adata.obs["total_transcripts"] = np.asarray(total_values).ravel().astype(np.int64)

    if "feature_types" in adata.var:
        feature_types = adata.var["feature_types"].astype(str).str.lower()
        gene_mask = feature_types.isin(["gene", "gene expression"]).to_numpy()
    else:
        gene_mask = ~feature_names.str.contains(
            "codeword|controlprobe|control_codeword",
            case=False,
            regex=True,
        )

    keep_features = ~total_mask if include_non_gene_features else (gene_mask & ~total_mask)
    if not keep_features.all():
        adata = adata[:, np.asarray(keep_features)].copy()

    cells = pd.read_parquet(f"{xdir}/cells.parquet")
    cells.index = cells["cell_id"].astype("str")
    adata.obsm["spatial"] = cells.loc[:, ["x_centroid", "y_centroid"]].values

    umap_file = os.path.join(xdir, "analysis/umap/gene_expression_2_components/projection.csv")
    try:
        umap_coords = pd.read_csv(umap_file, index_col=0)
        umap_coords.index = umap_coords.index.astype(str)
        adata.obsm["X_umap"] = umap_coords.reindex(adata.obs_names).to_numpy()
    except FileNotFoundError:
        if verbose:
            print("No UMAP coordinates found.")

    xrange = adata.obsm["spatial"][:, 0].max() - adata.obsm["spatial"][:, 0].min()
    yrange = adata.obsm["spatial"][:, 1].max() - adata.obsm["spatial"][:, 1].min()
    adata.uns["aspect_ratio"] = xrange / yrange

    panel_file = os.path.join(xdir, "gene_panel.json")
    gene_panel = read_xen_panel(panel_file)
    try:
        species = gene_panel["payload"]["panel"]["species"]
    except KeyError:
        species = "Unknown"
    adata.uns["genome"] = species

    adata.layers["counts"] = adata.X.astype("int").copy()
    _annotate_adata_count_metrics(adata, counts=adata.layers["counts"])
    adata.layers["TP10K"] = normalize_tp10k(adata.layers["counts"], log1p=True).astype("float32")

    adata.obs["n_counts"] = adata.obs["n_transcripts"].astype("int")
    adata.obs["n_genes"] = np.sum(adata.layers["counts"] > 0, axis=1).A1.astype("int")
    adata.obs["logUMIs"] = np.log(adata.obs["n_transcripts"] + 1)

    clusters = _read_all_classic_cluster_annotations(xdir, adata.obs_names, verbose=verbose)
    adata.obs = adata.obs.merge(clusters, left_index=True, right_index=True, how="left")

    if verbose:
        print(adata)
        print("Ready!")
    return adata


def load_xenium_cell_matrix(
    xenium_folder,
    *,
    bundle_format: str,
    include_non_gene_features: bool = False,
    verbose: bool = True,
) -> CellMatrixLoadResult:
    """
    Load cell-level AnnData, cluster assignments, and gene panel metadata.
    """
    panel_file = os.path.join(xenium_folder, "gene_panel.json")
    gene_panel = read_xen_panel(panel_file) if os.path.exists(panel_file) else None

    if bundle_format == "zarr":
        adata = _read_zarr_adata(
            xenium_folder,
            verbose=verbose,
            include_non_gene_features=include_non_gene_features,
        )
        clusters = _read_analysis_zarr(xenium_folder, verbose=verbose)
        if len(clusters):
            adata.obs = adata.obs.merge(clusters, left_index=True, right_index=True, how="left")
        return CellMatrixLoadResult(adata=adata, clusters=clusters, gene_panel=gene_panel)

    if bundle_format == "parquet":
        clusters = read_classic_analysis_clusters(xenium_folder, verbose=verbose)
        if verbose:
            print("Reading in AnnData object")
        adata = read_xenium_to_anndata(
            xenium_folder,
            include_non_gene_features=include_non_gene_features,
            verbose=verbose,
        )
        adata.obs = adata.obs.merge(clusters, left_index=True, right_index=True, how="left")
        if gene_panel is not None:
            # Preserve historical behavior by computing the merge target without
            # mutating var; this will be tightened in a later API cleanup pass.
            adata.var.merge(
                _make_gene_panel_df(gene_panel),
                left_index=True,
                right_index=True,
                how="left",
            )
        return CellMatrixLoadResult(adata=adata, clusters=clusters, gene_panel=gene_panel)

    raise ValueError("bundle_format must be 'zarr' or 'parquet'.")
