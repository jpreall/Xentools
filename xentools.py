#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
from typing import Union, Optional, Literal
from matplotlib.patches import Patch


def _load_local_module(module_name, relative_path):
    """Load a sibling module by file path for standalone `xentools.py` imports."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    if os.path.basename(module_path) == "__init__.py":
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[os.path.dirname(module_path)],
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .utils.misc import read_json
    from .utils.geometry import frame, um_to_pixels
except ImportError:
    read_json = _load_local_module("_xentools_utils_misc", os.path.join("utils", "misc.py")).read_json
    _geometry_utils_mod = _load_local_module("_xentools_utils_geometry", os.path.join("utils", "geometry.py"))
    frame = _geometry_utils_mod.frame
    um_to_pixels = _geometry_utils_mod.um_to_pixels


def _is_lazy_transcripts(obj) -> bool:
    """
    Detect LazyTranscripts-like objects without relying on class identity.

    Standalone ``sys.path.append(repo); import xentools`` imports can load the
    same source class under multiple module names, so ``isinstance`` is brittle.
    """
    return (
        hasattr(obj, "query")
        and callable(getattr(obj, "query"))
        and hasattr(obj, "_gene_names")
        and hasattr(obj, "_tile_meta")
    )

try:
    from .analysis.binning import create_binned_adata as _create_binned_adata
    from .analysis.graph import _apply_weights, build_spatial_graph
    from .analysis.niches import build_niches, evaluate_niche_k_values
    from .analysis.normalization import normalize_tp10k
except ImportError:
    _binning_mod = _load_local_module(
        "_xentools_analysis_binning",
        os.path.join("analysis", "binning.py"),
    )
    _graph_mod = _load_local_module(
        "_xentools_analysis_graph",
        os.path.join("analysis", "graph.py"),
    )
    _niches_mod = _load_local_module(
        "_xentools_analysis_niches",
        os.path.join("analysis", "niches.py"),
    )
    _normalization_mod = _load_local_module(
        "_xentools_analysis_normalization",
        os.path.join("analysis", "normalization.py"),
    )
    _create_binned_adata = _binning_mod.create_binned_adata
    _apply_weights = _graph_mod._apply_weights
    build_spatial_graph = _graph_mod.build_spatial_graph
    build_niches = _niches_mod.build_niches
    evaluate_niche_k_values = _niches_mod.evaluate_niche_k_values
    normalize_tp10k = _normalization_mod.normalize_tp10k

try:
    from .core.rois import (
        ROI,
        ROIClass,
        ROICollection,
        read_ROI_from_csv,
        read_ROI_from_geojson,
        _coerce_roi_geometry,
        _coerce_roi,
        _geometry_to_roi_points,
        _normalize_roi_property,
        _geojson_class_name,
        _geojson_selection_name,
        _make_roi_name,
        _register_roi_record,
        _select_geojson_features,
        _match_geojson_row,
        _import_roi_records,
        _resolve_roi_selection_selector,
        _roi_bounds_um,
        _shapely_bounds,
    )
    from .core.boundaries import LazyBoundaryGeoDataFrame
except ImportError:
    from core.rois import (
        ROI,
        ROIClass,
        ROICollection,
        read_ROI_from_csv,
        read_ROI_from_geojson,
        _coerce_roi_geometry,
        _coerce_roi,
        _geometry_to_roi_points,
        _normalize_roi_property,
        _geojson_class_name,
        _geojson_selection_name,
        _make_roi_name,
        _register_roi_record,
        _select_geojson_features,
        _match_geojson_row,
        _import_roi_records,
        _resolve_roi_selection_selector,
        _roi_bounds_um,
        _shapely_bounds,
    )
    from core.boundaries import LazyBoundaryGeoDataFrame

try:
    from .core.transcripts import (
        LazyTranscripts,
        _detect_transcripts_format,
        _count_transcripts_in_bundle,
        _encode_xenium_cell_ids,
        _load_zarr_gene_names,
    )

    from .io.read.zarr import (
        _open_zarr_group_compat,
        _read_zarr_adata,
        _read_analysis_zarr,
    )
    from .io.read.boundaries import load_xenium_boundaries
    from .io.read.cells import (
        _make_gene_panel_df as _read_make_gene_panel_df,
        load_xenium_cell_matrix,
        read_xen_panel as _read_xen_panel,
        read_xenium_to_anndata as _read_xenium_to_anndata,
    )
    from .io.read.images import (
        _image_extent_um,
        _parse_ome_xml,
        _source_series_level_arrays,
    )
    from .io.read.metadata import load_xenium_metadata
    from .io.read.transcripts import load_xenium_transcripts
    from .io.read.loader import load_xenium_folder
    from .io.write.xenium import (
        write_geo_submission as _write_geo_submission_bundle,
        write_xenium_explorer as _write_xenium_explorer_bundle,
    )
    from .io.write.images import (
        _build_pyramid_levels,
        _cropped_shape_from_bounds,
        _iter_cropped_tiles,
        _linked_ome_xml,
        _pixel_aligned_bounds_um,
        _roi_bounds_in_pixels,
        _scale_bounds_for_level,
        _write_linked_pyramidal_ome_tiffs,
        _write_linked_pyramidal_ome_tiffs_from_levels,
        write_ome_tiff as _write_ome_tiff_bundle,
        _write_pyramidal_ome_tiff,
        _write_pyramidal_ome_tiff_from_levels,
    )
except ImportError:
    _transcripts_mod = _load_local_module("_xentools_core_transcripts", "core/transcripts.py")
    _zarr_read_mod = _load_local_module("_xentools_io_read_zarr", os.path.join("io", "read", "zarr.py"))
    _xenium_write_mod = _load_local_module("_xentools_io_write_xenium", os.path.join("io", "write", "xenium.py"))
    _images_read_mod = _load_local_module("_xentools_io_read_images", os.path.join("io", "read", "images.py"))
    _images_write_mod = _load_local_module("_xentools_io_write_images", os.path.join("io", "write", "images.py"))

    LazyTranscripts = _transcripts_mod.LazyTranscripts
    _detect_transcripts_format = _transcripts_mod._detect_transcripts_format
    _count_transcripts_in_bundle = _transcripts_mod._count_transcripts_in_bundle
    _encode_xenium_cell_ids = _transcripts_mod._encode_xenium_cell_ids
    _load_zarr_gene_names = _transcripts_mod._load_zarr_gene_names

    _open_zarr_group_compat = _zarr_read_mod._open_zarr_group_compat
    _read_zarr_adata = _zarr_read_mod._read_zarr_adata
    _read_analysis_zarr = _zarr_read_mod._read_analysis_zarr
    _boundaries_read_mod = _load_local_module(
        "_xentools_io_read_boundaries",
        os.path.join("io", "read", "boundaries.py"),
    )
    load_xenium_boundaries = _boundaries_read_mod.load_xenium_boundaries
    _cells_read_mod = _load_local_module(
        "_xentools_io_read_cells",
        os.path.join("io", "read", "cells.py"),
    )
    _read_make_gene_panel_df = _cells_read_mod._make_gene_panel_df
    load_xenium_cell_matrix = _cells_read_mod.load_xenium_cell_matrix
    _read_xen_panel = _cells_read_mod.read_xen_panel
    _read_xenium_to_anndata = _cells_read_mod.read_xenium_to_anndata
    _image_extent_um = _images_read_mod._image_extent_um
    _parse_ome_xml = _images_read_mod._parse_ome_xml
    _source_series_level_arrays = _images_read_mod._source_series_level_arrays
    _metadata_read_mod = _load_local_module(
        "_xentools_io_read_metadata",
        os.path.join("io", "read", "metadata.py"),
    )
    load_xenium_metadata = _metadata_read_mod.load_xenium_metadata
    _transcripts_read_mod = _load_local_module(
        "_xentools_io_read_transcripts",
        os.path.join("io", "read", "transcripts.py"),
    )
    load_xenium_transcripts = _transcripts_read_mod.load_xenium_transcripts
    _loader_read_mod = _load_local_module(
        "_xentools_io_read_loader",
        os.path.join("io", "read", "loader.py"),
    )
    load_xenium_folder = _loader_read_mod.load_xenium_folder
    _write_geo_submission_bundle = _xenium_write_mod.write_geo_submission
    _write_xenium_explorer_bundle = _xenium_write_mod.write_xenium_explorer
    _build_pyramid_levels = _images_write_mod._build_pyramid_levels
    _cropped_shape_from_bounds = _images_write_mod._cropped_shape_from_bounds
    _iter_cropped_tiles = _images_write_mod._iter_cropped_tiles
    _linked_ome_xml = _images_write_mod._linked_ome_xml
    _pixel_aligned_bounds_um = _images_write_mod._pixel_aligned_bounds_um
    _roi_bounds_in_pixels = _images_write_mod._roi_bounds_in_pixels
    _scale_bounds_for_level = _images_write_mod._scale_bounds_for_level
    _write_linked_pyramidal_ome_tiffs = _images_write_mod._write_linked_pyramidal_ome_tiffs
    _write_linked_pyramidal_ome_tiffs_from_levels = _images_write_mod._write_linked_pyramidal_ome_tiffs_from_levels
    _write_ome_tiff_bundle = _images_write_mod.write_ome_tiff
    _write_pyramidal_ome_tiff = _images_write_mod._write_pyramidal_ome_tiff
    _write_pyramidal_ome_tiff_from_levels = _images_write_mod._write_pyramidal_ome_tiff_from_levels



def read_xen_panel(gene_panel_file):
    """
    Reads a Xenium gene panel file and returns the contents as a dictionary.
    """
    return _read_xen_panel(gene_panel_file)

def _make_gene_panel_df(gene_panel_dict):
    """
    Converts the gene panel dictionary to a DataFrame.
    """
    return _read_make_gene_panel_df(gene_panel_dict)

def read_xenium_to_anndata(xenium_output_folder, include_non_gene_features=False):
    return _read_xenium_to_anndata(
        xenium_output_folder,
        include_non_gene_features=include_non_gene_features,
    )

try:
    from .core.xendata import XenData
except ImportError:
    from core.xendata import XenData
    
## DEPRECATED
#def read_ROI_from_csv(XenAna_csv_file):
#    """
#    Xenium Explorer ROI files look like this:
#    
#    """
#    ROI = pd.read_csv(XenAna_csv_file, comment='#').select_dtypes(np.number).values
#    return ROI

def ROI_to_pixels(ROI, pixel_size):
    xmin,xmax = int(ROI[:,0].min()/pixel_size),int(ROI[:,0].max()/pixel_size)
    ymin, ymax = int(ROI[:,1].min()/pixel_size),int(ROI[:,1].max()/pixel_size)
    return xmin, xmax, ymin, ymax

def import_cell_annotations(
    xdata: XenData,
    cell_annotations_file):
    """
    Imports cell annotations from a CSV file into the XenData object.
    The CSV file should contain a column with the same name as the first column in the
    xdata.adata.obs DataFrame. The annotations will be added to the xdata.adata.obs DataFrame.
    Parameters:
    - xdata: XenData object containing transcript data.
    - cell_annotations_file: path to the CSV file containing cell annotations.  
    """
    anno = pd.read_csv(cell_annotations_file, index_col=0)
    groups_col = anno.columns[0]
    if groups_col not in xdata.adata.obs.columns:
        xdata.adata.obs = xdata.adata.obs.merge(
            anno[groups_col], left_index=True,right_index=True, how='left')
    

def rasterize_rgb(xdata, 
    genes_or_gene_sets, 
    bin_size=8, 
    fig_scale=10,
    gammas=[1, 1, 1],
    log=False,
    include_unassigned=False):
    """
    Creates an RGB image of up to 3 binned gene expression profiles.
    Accepts either:
    - A list of up to 3 genes, plotting each expression in the R, G, and B channels.
    - A dictionary of up to 3 gene sets, where each key corresponds to a channel (R, G, B),
      and the value is a list of genes to combine into that channel.

    Parameters:
    - xdata: XenData object containing transcript data.
    - genes_or_gene_sets: List of up to 3 genes or a dictionary with up to 3 gene sets.
    - bin_size: Size of the bins for rasterization.
    - fig_scale: Scale of the figure for plotting.
    - gammas: List of gamma correction values for each channel.
    - log: If True, apply log transformation to the counts before plotting.
    - include_unassigned: If True, include transcripts not assigned to any cell in the plot.

    Returns:
    - None: Displays the RGB image with a legend.
    """
    from PIL import Image
    from matplotlib.patches import Patch

    df = xdata.trans
    if include_unassigned is False:
        df = df[df['cell_id'] != 'UNASSIGNED'].copy()

    # Determine input type (list of genes or dictionary of gene sets)
    if isinstance(genes_or_gene_sets, list):
        genes = genes_or_gene_sets
        if len(genes) > 3:
            raise ValueError("Only 3 genes can be rasterized at a time.")
        if len(genes) < 1:
            raise ValueError("At least 1 gene must be rasterized.")
        gene_sets = {gene: [gene] for gene in genes}
    elif isinstance(genes_or_gene_sets, dict):
        gene_sets = genes_or_gene_sets
        genes = list(gene_sets.keys())  # Initialize genes with the keys of the dictionary
        if len(gene_sets) > 3:
            raise ValueError("Only 3 gene sets can be rasterized at a time.")
        if len(gene_sets) < 1:
            raise ValueError("At least 1 gene set must be rasterized.")
    else:
        raise ValueError("Input must be a list of genes or a dictionary of gene sets.")

    # Compute bin edges and aspect ratio
    x_min, x_max = df['x_location'].min(), df['x_location'].max()
    y_min, y_max = df['y_location'].min(), df['y_location'].max()
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    aspect_ratio = (x_edges[-1] - x_edges[0]) / (y_edges[-1] - y_edges[0])

    # Precompute bin indices (zero-based)
    x_idx = np.digitize(df['x_location'].values, x_edges) - 1
    y_idx = np.digitize(df['y_location'].values, y_edges) - 1

    # Define histogram shape (bins count in each dimension)
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    counts = np.zeros((3, shape[0], shape[1]), dtype=np.float32)

    # Process each gene set and accumulate counts
    for ch, (channel_name, gene_list) in enumerate(gene_sets.items()):
        if ch >= 3:
            break  # Only process up to 3 channels
        mask = df['feature_name'].isin(gene_list).values
        if np.any(mask):
            np.add.at(counts[ch], (y_idx[mask], x_idx[mask]), 1)

    # Apply log transformation if specified
    if log:
        counts = np.log1p(counts)

    # Normalize each channel to the range [0, 255]
    for ch in range(3):
        max_val = counts[ch].max()
        if max_val > 0:
            counts[ch] = np.round(255 * counts[ch] / max_val)
        else:
            counts[ch] = 0

    # Merge channels into a single image array
    merged = np.stack([counts[0], counts[1], counts[2]], axis=-1).clip(0, 255).astype(np.uint8)

    # Apply gamma correction to each channel if necessary
    for i in range(3):
        channel_norm = merged[:, :, i] / 255.0
        channel_corr = 255 * np.power(channel_norm, 1.0 / gammas[i])
        merged[:, :, i] = np.clip(channel_corr, 0, 255).astype(np.uint8)

    im = Image.fromarray(merged)

    # Plot the image with annotations for each gene or gene set
    plt.figure(figsize=[fig_scale * aspect_ratio, fig_scale])

    # Create a sub-function to add a legend outside the main plot
    def add_legend_outside(labels, colors, fig, ax):
        """
        Adds a legend outside the main plot for the specified labels and colors.

        Parameters:
        - labels: List of gene or gene set names.
        - colors: List of colors corresponding to the labels.
        - fig: The matplotlib figure object.
        - ax: The matplotlib axis object.
        """
        legend_handles = [Patch(color=color, label=label) for label, color in zip(labels, colors)]
        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize='medium',
            labelcolor='white',
            facecolor='black',
            edgecolor='black',
        )

    # Add the legend to the plot
    fig, ax = plt.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
    ax.imshow(im)
    ax.axis('off')

    # Define colors for the channels
    colors = ['red', 'green', 'blue'][:len(gene_sets)]
    labels = list(gene_sets.keys())
    add_legend_outside(labels, colors, fig, ax)

    plt.tight_layout()
    plt.show()

def plot_binned_rgb(xdata, 
    genes_or_gene_sets, 
    norm='per_gene',
    fig_scale=10,
    gammas=[1, 1, 1],
    log=False,
    flip=True,
    bounds: tuple|None=None, # extent bounds for plotting: (xmin, xmax, ymin, ymax)
    ):

    xmin, xmax, ymin, ymax = bounds if bounds is not None else (None, None, None, None)
    """
    Creates an RGB image of binned gene expression profiles.
    Can accept either:
    - XenData object with binned transcript data.
    - AnnData object with binned transcript data.

    Parameters:
    - xdata: XenData object containing transcript data.
    - genes_or_gene_sets: Either a list of up to 3 genes to rasterize, 
      or a dictionary of up to 3 gene sets where each key corresponds to a channel (R, G, B) 
      and the value is a list of genes to combine into that channel.
    - norm: Normalization method ('per_gene' or 'global').
    - fig_scale: Scale of the figure for plotting.
    - gammas: List of gamma correction values for each channel.
    - log: If True, apply log transformation to the counts before plotting.
    - flip: If True, flip the image upside down to match Scanpy and Xenium plotting orientation.
    
    Returns:
    - None: Displays the RGB image with a legend.
    """
    from PIL import Image
    from matplotlib.patches import Patch
    import anndata as ad

    # Check if xdata is a XenData object with binned_adata
    if hasattr(xdata, 'binned_adata'):
        data = xdata.binned_adata
    # If not, check if xdata is an AnnData object with binned transcript data in 'spatial' 
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if 'spatial' not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. "
                             "Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")
    

    # Determine input type (list of genes or dictionary of gene sets)
    if isinstance(genes_or_gene_sets, list):
        genes = genes_or_gene_sets
        if len(genes) > 3:
            raise ValueError("Only 3 genes can be rasterized at a time.")
        if len(genes) < 1:
            raise ValueError("At least 1 gene must be rasterized.")
        gene_sets = {gene: [gene] for gene in genes}
    elif isinstance(genes_or_gene_sets, dict):
        gene_sets = genes_or_gene_sets
        if len(gene_sets) > 3:
            raise ValueError("Only 3 gene sets can be rasterized at a time.")
        if len(gene_sets) < 1:
            raise ValueError("At least 1 gene set must be rasterized.")
    else:
        raise ValueError("Input must be a list of genes or a dictionary of gene sets.")
    
    # Extract spatial coordinates (assume shape (n_bins, 2): columns x and y)
    xy_coords = data.obsm['spatial']
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]

    # Determine image bounds and dimensions
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w = int(np.abs(x_max - x_min)) + 1
    h = int(np.abs(y_max - y_min)) + 1
    aspect_ratio = w / h

    # Create an empty image array (height, width, 3)
    imdata = np.zeros((h, w, 3))

    # Convert spatial coordinates into image indices starting from zero
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)

    # Populate image with expression values per gene into corresponding RGB channels
    for n, (set_name,genes) in enumerate(gene_sets.items()):
        expr = data[:, genes].X.todense().sum(1).A1
        for xi, yi, intensity in zip(x_idx, y_idx, expr):
            imdata[yi, xi, n] = intensity

    # log transform if specified
    if log:
        imdata = np.log1p(imdata)
    
    # Normalize each channel to the range [0, 255]
    # Ensure norm is either 'per_gene' or 'global'
    assert norm in ['per_gene', 'global'], "Invalid normalization method. Choose 'per_gene' or 'global'."

    global_max = imdata.max()
    for ch in range(3):
        if norm == 'per_gene':
            max_val = imdata[:, :, ch].max()
        elif norm == 'global':
            max_val = global_max
        if max_val > 0:
            imdata[:, :, ch] = np.round(255 * imdata[:, :, ch] / max_val)
        else:
            imdata[:, :, ch] = 0

    # Flip the image upside down to match Scanpy and Xenium plotting orientation
    if flip == True:
        imdata = imdata[::-1, :, :]

    # Apply gamma correction to each channel if necessary
    for i in range(3):
        channel_norm = imdata[:, :, i] / 255.0
        channel_corr = 255 * np.power(channel_norm, 1.0 / gammas[i])
        imdata[:, :, i] = np.clip(channel_corr, 0, 255).astype(np.uint8)
        
    # Convert to PIL Image
    if bounds is None:
        im = Image.fromarray(np.uint8(imdata))
    else:
        im = imdata[...,:3].astype(np.uint8)
    # Plot the image with annotations for each gene or gene set
    plt.figure(figsize=[fig_scale * aspect_ratio, fig_scale])

    # Create a sub-function to add a legend outside the main plot
    def add_legend_outside(labels, colors, fig, ax):
        """
        Adds a legend outside the main plot for the specified labels and colors.

        Parameters:
        - labels: List of gene or gene set names.
        - colors: List of colors corresponding to the labels.
        - fig: The matplotlib figure object.
        - ax: The matplotlib axis object.
        """
        legend_handles = [Patch(color=color, label=label) for label, color in zip(labels, colors)]
        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize='medium',
            labelcolor='white',
            facecolor='black',
            edgecolor='black',
        )

    # Add the legend to the plot
    fig, ax = plt.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])

    if bounds is None:
        ax.imshow(im)
    else:
        print(bounds)
        extent = [xmin, xmax, ymin, ymax]
        ax.imshow(imdata.astype(np.uint8), extent=extent)
    ax.axis('off')

    # Define colors for the channels
    colors = ['red', 'green', 'blue'][:len(gene_sets)]
    labels = list(gene_sets.keys())
    add_legend_outside(labels, colors, fig, ax)

    plt.tight_layout()
    plt.show()

def create_multilayer_image(xdata, genes, log=False):
    """
    Create a multilayer image from the binned AnnData object in xdata.
    Can accept both XenData and AnnData objects as input.
    Parameters:
    - xdata: XenData object containing the binned AnnData object.
    - genes: List of gene names to include in the image.
    - log: If True, apply log transformation to the expression data before creating the image.
    Returns:
    - imdata: A 3D numpy array representing the image, with shape (height, width, n_genes).
    """
    import anndata as ad

    # Determine if xdata is a XenData object or an AnnData object
    if isinstance(xdata, XenData):
        if not hasattr(xdata, 'binned_adata'):
            raise ValueError("XenData object does not have a binned_adata attribute. "
                             "Please create a binned AnnData object first using xdata.create_binned_adata().")
        data = xdata.binned_adata
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if 'spatial' not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. "
                             "Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")
    # Check if genes is None or a string, and convert to list if necessary
    if genes is None:
        genes = data.var_names.tolist()

    elif isinstance(genes, str):
        genes = [genes]
    # Ensure genes are present in the AnnData object
    missing_genes = [gene for gene in genes if gene not in data.var_names]
    if missing_genes:
        raise ValueError(f"The following genes are not present in the binned AnnData object: {', '.join(missing_genes)}")

    # Get spatial coordinates (assumed to be in xdata.binned_adata.obsm['spatial'])
    coords = data.obsm['spatial']
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Determine image bounds and create index arrays
    x_min, y_min = x_coords.min(), y_coords.min()
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)
    w = int(x_coords.max() - x_min) + 1
    h = int(y_coords.max() - y_min) + 1

    # Prepare output image array: one layer per gene
    n_genes = len(genes)
    imdata = np.zeros((h, w, n_genes), dtype=np.uint16)

    # Extract expression data for all genes at once
    data = data[:, genes].X
    # If data is sparse, convert to a dense array
    if hasattr(data, "toarray"):
        data = data.toarray()  # Shape: (n_cells, n_genes)

    if log:
        # Apply log2 transformation to the expression data
        data = np.log2(data + 1)
        # scale to 8-bit range
        data = np.clip(data, 0, 255).astype(np.uint8)
        
    else:
        # Clip to 8-bit or 16-bit range, depending on the dynamic range of the data
        if np.max(data) > 255:
            data = np.clip(data, 0, 2**16 - 1).astype(np.uint16)
        else:
            # If the data fits in 8 bits, convert to uint8
            data = np.clip(data, 0, 255).astype(np.uint8)

    #data = np.clip(data, 0, 2**16 - 1).astype(np.uint16)

    # Use vectorized assignment: each cell's expression for all genes is written to its corresponding spatial index
    imdata[y_idx, x_idx, :] = data

    # Flip the image upside down to match Scanpy and Xenium plotting orientation
    imdata = imdata[::-1, :, :]

    return imdata

def plot_binned_greyscale(xdata, 
    genes, 
    fig_scale=10,
    gamma=1,
    log=False,
    flip=True,
    return_img=False,
    cmap='inferno',
    ):
    """
    Creates a greyscale image of binned gene expression profiles.
    Can accept either:
    - XenData object with binned transcript data.
    - AnnData object with binned transcript data.

    Parameters:
    - xdata: XenData object containing transcript data.
    - genes: a list of genes to combine
    - fig_scale: Scale of the figure for plotting.
    - gammas: List of gamma correction values for each channel.
    - log: If True, apply log transformation to the counts before plotting.
    - flip: If True, flip the image upside down to match Scanpy and Xenium plotting orientation.
    
    Returns:
    - None: Displays the RGB image with a legend.
    """
    from PIL import Image
    from matplotlib.patches import Patch
    import anndata as ad

    # Check if xdata is a XenData object with binned_adata
    if hasattr(xdata, 'binned_adata'):
        data = xdata.binned_adata
    # If not, check if xdata is an AnnData object with binned transcript data in 'spatial' 
    elif isinstance(xdata, ad.AnnData):
        data = xdata
        if 'spatial' not in data.obsm:
            raise ValueError("AnnData object does not have spatial coordinates in obsm['spatial']. "
                             "Please ensure the AnnData object has been properly prepared.")
    else:
        raise ValueError("Input must be a XenData object or an AnnData object with binned transcript data.")
    
    if isinstance(genes, str):
        genes = [genes]
    
    # Extract spatial coordinates (assume shape (n_bins, 2): columns x and y)
    xy_coords = data.obsm['spatial']
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]

    # Determine image bounds and dimensions
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w = int(np.abs(x_max - x_min)) + 1
    h = int(np.abs(y_max - y_min)) + 1
    aspect_ratio = w / h

    # Create an empty image array (height, width, 3)
    imdata = np.zeros((h, w, 1))

    # Convert spatial coordinates into image indices starting from zero
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)

    # Populate image with expression values per gene into corresponding RGB channels
    
    from scipy.sparse import issparse
    if issparse(data[:, genes].X):
        expr = data[:, genes].X.todense().sum(1).A1
    else:
        expr = data[:, genes].X.sum(1)
    imdata[y_idx, x_idx, 0] = expr

    # log transform if specified
    if log:
        imdata = np.log1p(imdata)
    
    imdata = imdata[:,:,0]

    norm_imdata = imdata / imdata.max()
    imdata = np.round(255*(norm_imdata))

    channel_norm = imdata[:, :] / 255.0
    channel_corr = 255 * np.power(channel_norm, 1.0 / gamma)
    imdata = np.clip(channel_corr, 0, 255)

    if flip == True:
        imdata = imdata[::-1, :]
    
    if return_img:
        return imdata
    else:
        fig, ax = plt.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
        ax.imshow(imdata, cmap=cmap)
        ax.axis('off')
        plt.show()

def show_ome_tiff(image_path,
                  figsize: Optional[tuple] = None,
                  dpi: Optional[int] = None,
                  cmap: str = 'gray',
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  clip_percentile: float = 99.5,
                  z_index: Optional[int] = None,
                  pixel_size: float = 0.2125,
                  roi=None,
                  level: Optional[int] = None,
                  micron_coords: bool = False,
                  ax=None,
                  verbose: bool = True):
    """
    Display a single OME-TIFF (or any pyramidal TIFF) directly from a file path.

    Intended as a standalone development/debugging companion to ``XenData.show_image``.
    Automatically selects the coarsest pyramid level that still meets the display
    resolution so that large Xenium morphology images load quickly in notebooks.

    Parameters
    ----------
    image_path : str | Path
        Path to the OME-TIFF file (e.g. morphology.ome.tif or a ch*.ome.tif file).
    figsize : tuple, optional
        Figure size in inches (width, height). Defaults to matplotlib rcParams.
    dpi : int, optional
        Display DPI. Defaults to matplotlib rcParams.
    cmap : str
        Matplotlib colormap. Default 'gray'.
    vmin, vmax : float, optional
        Intensity range for display. vmax is auto-set via ``clip_percentile`` when None.
    clip_percentile : float
        Percentile used to auto-set vmax. Default 99.5.
    z_index : int, optional
        Which Z slice to show for multi-plane images. When None (default), a
        max-intensity projection across all Z slices is displayed.
    pixel_size : float
        Microns per pixel (only used when ``roi`` is provided for coordinate conversion).
        Default 0.2125 (standard Xenium pixel size).
    roi : shapely geometry | None
        Optional crop region in micron coordinates (e.g. a shapely Polygon).
        When provided, only the bounding box of the ROI is loaded and displayed.
    level : int, optional
        Manually override the pyramid level (0 = full resolution, higher = coarser).
        If None (default), the level is chosen automatically based on display size.
        Clamped to the number of available levels if out of range.
    micron_coords : bool
        If True, display in micron coordinates with ``origin="lower"``, matching
        the coordinate system used by ``splat()``. Required for overlay use.
        Default False (pixel coordinates, origin upper-left).
    ax : matplotlib Axes, optional
        Axes to plot into. If None, a new figure is created.
    verbose : bool
        Print the selected pyramid level info. Default True.

    Returns
    -------
    ax : matplotlib Axes
    """
    import tifffile

    if dpi is None:
        dpi = plt.rcParams['figure.dpi']
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']

    display_px = max(int(figsize[0] * dpi), int(figsize[1] * dpi))

    with tifffile.TiffFile(image_path) as tif:
        page0 = tif.pages[0]
        subifd_pages = list(page0.pages)
        try:
            series = tif.series[0]
            level_arrays, zarr_store = _source_series_level_arrays(series)
            use_zarr_levels = True
        except Exception as exc:
            message = str(exc)
            if "multi-file pyramids" not in message:
                raise
            use_zarr_levels = False
            zarr_store = None
            level_arrays = [page0] + subifd_pages

        # Some multi-file OME-TIFFs silently expose only level 0 via the OME/zarr
        # path even though TIFF SubIFDs are present. Prefer the SubIFD pyramid in
        # that case so explicit level requests can still work.
        if use_zarr_levels and len(level_arrays) == 1 and len(subifd_pages) > 0:
            if zarr_store is not None:
                zarr_store.close()
                zarr_store = None
            use_zarr_levels = False
            level_arrays = [page0] + subifd_pages

        try:
            full_shape = level_arrays[0].shape

            # Crop bounds from an optional ROI (in micron coordinates)
            if roi is not None:
                base_bounds = _roi_bounds_in_pixels(roi, pixel_size, full_shape)
                source_px = max(
                    base_bounds[1] - base_bounds[0],
                    base_bounds[3] - base_bounds[2],
                )
            else:
                base_bounds = None
                source_px = max(full_shape[-2:])

            # Coarsest pyramid level that still meets display_px (or use override)
            if level is not None:
                best_level = min(level, len(level_arrays) - 1)
            else:
                best_level = 0
                for i in range(len(level_arrays)):
                    if source_px // (2 ** i) >= display_px:
                        best_level = i

            level_array = level_arrays[best_level]
            level_shape = level_array.shape

            if verbose:
                mode = "zarr" if use_zarr_levels else "tiff-subifd"
                print(
                    f"[show_ome_tiff] pyramid level {best_level}/{len(level_arrays)-1}  "
                    f"({level_shape[-2]} × {level_shape[-1]} px) [{mode}]"
                )

            if base_bounds is not None:
                lx0, lx1, ly0, ly1 = _scale_bounds_for_level(base_bounds, best_level)
                ly0 = max(0, ly0);  ly1 = min(level_shape[-2], ly1)
                lx0 = max(0, lx0);  lx1 = min(level_shape[-1], lx1)
                if use_zarr_levels:
                    if level_array.ndim == 2:
                        img = np.asarray(level_array[ly0:ly1, lx0:lx1])
                    elif z_index is None:
                        img = np.asarray(level_array[:, ly0:ly1, lx0:lx1]).max(axis=0)
                    else:
                        img = np.asarray(level_array[z_index, ly0:ly1, lx0:lx1])
                else:
                    arr = level_array.asarray()
                    if arr.ndim == 2:
                        img = np.asarray(arr[ly0:ly1, lx0:lx1])
                    elif z_index is None:
                        img = np.asarray(arr[:, ly0:ly1, lx0:lx1]).max(axis=0)
                    else:
                        img = np.asarray(arr[z_index, ly0:ly1, lx0:lx1])
            else:
                if use_zarr_levels:
                    if level_array.ndim == 2:
                        img = np.asarray(level_array)
                    elif z_index is None:
                        img = np.asarray(level_array).max(axis=0)
                    else:
                        img = np.asarray(level_array[z_index])
                else:
                    arr = level_array.asarray()
                    if arr.ndim == 2:
                        img = np.asarray(arr)
                    elif z_index is None:
                        img = np.asarray(arr).max(axis=0)
                    else:
                        img = np.asarray(arr[z_index])
        finally:
            if zarr_store is not None:
                zarr_store.close()

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = float(np.percentile(img, clip_percentile))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if micron_coords:
        # Compute the µm extent of the displayed region so the axes use the same
        # coordinate system as splat() (origin="lower", y increases upward).
        if base_bounds is not None:
            im_extent = [
                base_bounds[0] * pixel_size, base_bounds[1] * pixel_size,
                base_bounds[2] * pixel_size, base_bounds[3] * pixel_size,
            ]
        else:
            im_extent = [0.0, full_shape[-1] * pixel_size,
                         0.0, full_shape[-2] * pixel_size]
        # Flip rows so that origin="lower" puts small physical-y at the top,
        # matching the y_idx = (ymax - y) / px rasterisation used by splat().
        img = img[::-1]
        imshow_kwargs = dict(origin='lower', extent=im_extent)
    else:
        imshow_kwargs = {}

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
              **imshow_kwargs)
    ax.axis('off')

    if own_fig:
        plt.tight_layout()

    return ax

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def splat(
    data: XenData | pd.DataFrame = None, # XenData or DataFrame with transcript data
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
    genes: Union[None, str, list[str], dict] = None,  # list of genes to use for RGB channels
    gains=(1.0, 1.0, 1.0),     # per-channel gain / brightness
    bounds=None, # extent bounds for plotting: (xmin, xmax, ymin, ymax)
    pixel_size_um=1.0,
    sigma_um=2.0,
    ax=None,
    global_norm=False, # normalize over all channels jointly
    smooth = True, # apply Gaussian smoothing. Otherwise, just bin counts.
    show_ticks=False, # show distances on axes for help in cropping
    show_legend: bool = True,
    legend_loc: str = 'outside right',
):
    """
    Render spatial transcriptomic/transcript position data into 1–N channel raster images.
    The function bins transcript coordinates into a 2D pixel grid, optionally applies
    Gaussian smoothing (in micrometers -> pixels), and produces both a raw per-channel
    float image (counts or smoothed counts) and a display image that has been
    normalized and had per-channel gains applied. Channels may be constructed from
    single genes or from gene signatures (groups of genes). If no genes are provided,
    a random subset of transcripts is sampled and plotted as a single grayscale channel.

    Parameters
    ----------
    data : XenData | pandas.DataFrame
        Source of transcript data. If a XenData instance is provided the function
        uses data.trans. If a pandas.DataFrame is provided it must contain the
        columns specified by x_col, y_col and gene_col.
    x_col : str, optional
        Column name in `data` containing the x-coordinate (µm). Default "x_location".
    y_col : str, optional
        Column name in `data` containing the y-coordinate (µm). Default "y_location".
    gene_col : str, optional
        Column name in `data` containing the gene/feature name. Default "feature_name".
    genes : None | str | list[str] | dict, optional
        Defines channels:
          - None: sample up to 100k transcripts at random and render as a single
            grayscale channel named "Random transcripts".
          - str: single gene name -> one channel.
          - list[str]: each element is treated as the name of a separate channel.
          - dict: interpreted as gene signatures, mapping channel name -> list of genes.
            Each channel displays the union of the listed genes.
    gains : scalar | sequence, optional
        Per-channel multiplicative display gains (brightness). Can be a single
        scalar applied to every channel or a sequence of length equal to the number
        of channels. Gains are applied after normalization (in "display space").
        Default (1.0, 1.0, 1.0).
    bounds : tuple(float, float, float, float), optional
        Spatial extent to rasterize: (xmin, xmax, ymin, ymax). If None, bounds are
        inferred from the data min/max coordinates.
    pixel_size_um : float, optional
        Pixel size in micrometers. Determines raster grid resolution. Default 1.0 µm.
    sigma_um : float, optional
        Gaussian smoothing sigma in micrometers. Converted to pixels by dividing by
        pixel_size_um. Used only when `smooth` is True. Default 2.0 µm.
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to draw the image. If None a new figure/axis is created.
    global_norm : bool, optional
        If True, normalize all channels jointly by the global maximum. If False,
        normalize each channel independently. Default False (per-channel normalization).
    smooth : bool, optional
        If True apply Gaussian smoothing to the binned counts. If False return raw
        binned counts. Default True.
    show_ticks : bool, optional
        If True, show axis labels and ticks in micrometers for cropping help.
        Otherwise axis ticks/labels are removed. Default False.

    Returns
    -------
    rgb : numpy.ndarray, shape (ny, nx, n_channels), dtype float32
        Raw per-channel raster data (binned counts or smoothed counts). This is
        NOT normalized or clipped. ny and nx depend on bounds and pixel_size_um.
    disp : numpy.ndarray, shape (ny, nx, n_channels), dtype float32
        Display image: per-channel normalized (either per-channel or global),
        multiplied by `gains`, and clipped to [0, 1]. If n_channels >= 3 the first
        three channels are typically used as RGB for visualization.
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the plotted image (created if `ax` was None).
    Raises
    ------
    ValueError
        If `data` is not a XenData or pandas.DataFrame, or if `gains` is provided
        as a sequence whose length does not match the number of channels.
    Notes
    -----
    - Coordinate mapping:
        x pixel index = floor((x - xmin) / pixel_size_um)
        y pixel index = floor((ymax - y) / pixel_size_um)
      The y-axis is flipped so that the returned image's origin corresponds to
      (xmin, ymin) when displayed with origin="lower" and extent=(xmin, xmax, ymin, ymax).
    - Gaussian smoothing:
        The sigma supplied (sigma_um) is converted to pixels by dividing by
        pixel_size_um before calling the Gaussian filter.
    - Normalization and gains:
        Normalization is performed before gains are applied. If global_norm is True
        all channels are divided by the global maximum across all channels; otherwise
        each channel is divided by its own maximum. Gains are then applied in
        display space and final values are clipped to [0, 1].
    - Display:
        If an axis is not provided the function creates one. For n_channels == 1
        a grayscale image is shown; for n_channels >= 3 the first three channels
        are shown as an RGB image. Interpolation is set to "nearest".
    Examples
    --------
    Simple grayscale rendering of a DataFrame `df` with default parameters:
        img, disp, ax = splat(df)
    Render three specific genes as RGB channels with custom gains and pixel size:
        rgb, disp, ax = splat(df, genes=["GeneA", "GeneB", "GeneC"],
                             gains=(0.8, 1.2, 1.0), pixel_size_um=0.5)
    Use gene signatures to create named channels:
        signatures = {"Exc": ["SLC17A7", "CAMK2A"], "Inh": ["GAD1", "GAD2"]}
        rgb, disp, ax = splat(df, genes=signatures, global_norm=True)
    """
    if isinstance(data, XenData):
        df = data.trans
    elif isinstance(data, pd.DataFrame):
        df = data
    elif _is_lazy_transcripts(data):
        df = data
    else:
        raise ValueError("data must be XenData, LazyTranscripts, or pd.DataFrame")

    if _is_lazy_transcripts(df):
        # Resolve the gene list so the query pre-filters to only needed genes
        if genes is None:
            query_genes = None
        elif isinstance(genes, str):
            query_genes = [genes]
        elif isinstance(genes, list):
            query_genes = genes
        elif isinstance(genes, dict):
            query_genes = list({g for gs in genes.values() for g in gs})
        else:
            query_genes = None

        if bounds is not None:
            qxmin, qxmax, qymin, qymax = bounds
        else:
            qxmin = qxmax = qymin = qymax = None

        df = df.query(xmin=qxmin, xmax=qxmax, ymin=qymin, ymax=qymax,
                      genes=query_genes)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    g = df[gene_col].to_numpy()

    # Bounds
    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    width_um  = xmax - xmin
    height_um = ymax - ymin
    nx = int(np.ceil(width_um  / pixel_size_um))
    ny = int(np.ceil(height_um / pixel_size_um))

    sigma_px = sigma_um / pixel_size_um
    
    gene_signatures = None # Flag to indicate if using signatures

    # Decide channels: either signatures or individual genes
    if genes is None:
        # sample 100000 transcripts randomly and plot as greyscale
        rgb_genes = set(g[np.random.choice(len(g), size=min(100000, len(g)), replace=False)])
        chan_names = ['Random transcripts']
    elif isinstance(genes, dict):
        chan_names = list(genes.keys())
        gene_signatures = genes
        #chan_names = list(rgb_genes)
    elif isinstance(genes, str):
        chan_names = [genes]
    else:
        chan_names = list(genes)

    n_channels = len(chan_names)

    # Handle gains: allow scalar or sequence
    if np.isscalar(gains):
        gains = (float(gains),) * n_channels
    else:
        gains = tuple(gains)
        if len(gains) != n_channels:
            raise ValueError(f"gains length ({len(gains)}) must match number "
                             f"of channels ({n_channels})")

    # Allocate multi-channel image
    rgb = np.zeros((ny, nx, n_channels), dtype=np.float32)

    # --- fast path: convert gene strings to integer codes ONCE ---
    # pd.factorize is hash-based O(N); avoids repeated O(N*M) np.isin per channel
    codes, uniq = pd.factorize(g)
    gene_to_code = {gene: i for i, gene in enumerate(uniq)}

    # Build channel assignment array: uniq_gene_index -> channel_index (-1 = unused)
    chan_assignment = np.full(len(uniq), -1, dtype=np.int32)
    for k, chan_name in enumerate(chan_names):
        if gene_signatures is not None:
            for gene in gene_signatures[chan_name]:
                if gene in gene_to_code:
                    chan_assignment[gene_to_code[gene]] = k
        else:
            if chan_name in gene_to_code:
                chan_assignment[gene_to_code[chan_name]] = k

    # Map every transcript to its channel in one O(N) integer-array lookup
    transcript_channel = chan_assignment[codes]

    # Compute pixel indices ONCE for all transcripts
    x_idx_all = ((x - xmin) / pixel_size_um).astype(np.int32)
    y_idx_all = ((ymax - y) / pixel_size_um).astype(np.int32)
    valid_all = (
        (x_idx_all >= 0) & (x_idx_all < nx) &
        (y_idx_all >= 0) & (y_idx_all < ny)
    )

    for k in range(n_channels):
        mask = (transcript_channel == k) & valid_all
        if not mask.any():
            continue
        # np.bincount is buffered C code, much faster than np.add.at
        flat_idx = y_idx_all[mask].astype(np.int64) * nx + x_idx_all[mask]
        rgb[..., k] = np.bincount(flat_idx, minlength=ny * nx).reshape(ny, nx).astype(np.float32)

    # Single vectorized Gaussian blur across all channels at once
    if smooth:
        rgb = gaussian_filter(rgb, sigma=[sigma_px, sigma_px, 0], mode="nearest")

    # ---- normalization step (without gains) ----
    if global_norm:
        base = rgb.copy()
        m = base.max()
        if m > 0:
            disp = base / m
        else:
            disp = base
    else:
        # per-channel normalization
        base = rgb.copy()
        disp = np.zeros_like(base)
        for k in range(n_channels):
            m = base[..., k].max()
            if m > 0:
                disp[..., k] = base[..., k] / m

    # ---- apply per-channel gains in display space ----
    gains_arr = np.array(gains, dtype=float).reshape(1, 1, -1)
    disp = disp * gains_arr

    # option A: just clip (simplest; other channels not dimmed by one bright channel)
    disp = np.clip(disp, 0, 1)

    # Plot (only meaningful if n_channels is 1 or 3)
    if ax is None:
        fig_size = (12, 12) if n_channels == 3 else (6, 6)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

    extent = [xmin, xmax, ymin, ymax]
    if n_channels == 1:
        # grayscale
        ax.imshow(disp[..., 0], extent=extent, origin="lower",
                  interpolation="nearest", cmap="gray")
    elif n_channels >= 3:
        # use first 3 channels as RGB for display
        ax.imshow(disp[..., :3], extent=extent, origin="lower",
                  interpolation="nearest")
        
    if show_ticks:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(False)

    # ── legend / title ────────────────────────────────────────────────────────
    _channel_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    if show_legend and chan_names != ['Random transcripts']:
        # Derive a representative color for each channel
        if n_channels == 1:
            # Sample the display colormap near its bright end
            legend_colors = [plt.get_cmap('gray')(0.9)[:3]]
        else:
            legend_colors = [
                _channel_colors[i] if i < len(_channel_colors)
                else plt.get_cmap('hsv')(i / n_channels)[:3]
                for i in range(n_channels)
            ]


        handles = [Patch(color=c, label=n)
                   for c, n in zip(legend_colors, chan_names)]

        legend_kwargs = dict(
            handles=handles,
            fontsize='medium',
            labelcolor='white',
            facecolor='black',
            edgecolor='black',
        )
        if legend_loc == 'outside right':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **legend_kwargs)
        elif legend_loc == 'outside left':
            ax.legend(loc='center right', bbox_to_anchor=(0, 0.5), **legend_kwargs)
        elif legend_loc == 'outside bottom':
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                      ncol=min(n_channels, 4), **legend_kwargs)
        elif legend_loc == 'outside top':
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
                      ncol=min(n_channels, 4), **legend_kwargs)
        else:
            ax.legend(loc=legend_loc, **legend_kwargs)
    else:
        # Fall back to a plain title when legend is suppressed or unnamed
        ax.set_title(", ".join(chan_names[:3]))

    return rgb, disp, ax


try:
    from . import pl as _pl_namespace
except ImportError:
    _pl_namespace = _load_local_module("_xentools_pl", os.path.join("pl", "__init__.py"))

try:
    from . import io as _io_namespace
except ImportError:
    _io_namespace = _load_local_module("_xentools_io", os.path.join("io", "__init__.py"))

try:
    from . import utils as _utils_namespace
except ImportError:
    _utils_namespace = _load_local_module("_xentools_utils", os.path.join("utils", "__init__.py"))

create_bins = _pl_namespace.create_bins
bin_expression = _pl_namespace.bin_expression
create_binned_image = _pl_namespace.create_binned_image
rasterize = _pl_namespace.rasterize
rasterize_rgb = _pl_namespace.rasterize_rgb
plot_binned_rgb = _pl_namespace.plot_binned_rgb
create_multilayer_image = _pl_namespace.create_multilayer_image
plot_binned_greyscale = _pl_namespace.plot_binned_greyscale
show_ome_tiff = _pl_namespace.show_ome_tiff
splat = _pl_namespace.splat
pl = _pl_namespace
io = _io_namespace
utils = _utils_namespace
frame = _utils_namespace.frame
um_to_pixels = _utils_namespace.um_to_pixels

__all__ = [
    "io",
    "XenData",
    "LazyTranscripts",
    "LazyBoundaryGeoDataFrame",
    "ROI",
    "ROIClass",
    "ROICollection",
    "read_xen_panel",
    "um_to_pixels",
    "read_xenium_to_anndata",
    "frame",
    "ROI_to_pixels",
    "import_cell_annotations",
    "read_ROI_from_csv",
    "read_ROI_from_geojson",
    "build_spatial_graph",
    "build_niches",
    "evaluate_niche_k_values",
    "normalize_tp10k",
    "create_bins",
    "bin_expression",
    "create_binned_image",
    "rasterize",
    "rasterize_rgb",
    "plot_binned_rgb",
    "create_multilayer_image",
    "plot_binned_greyscale",
    "show_ome_tiff",
    "splat",
    "pl",
    "utils",
]
