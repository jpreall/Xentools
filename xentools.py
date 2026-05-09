#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sys
import scanpy as sc
import json
import geopandas as gpd
import re
import uuid
import importlib.util
from typing import Union, Optional, Literal
from scipy import sparse
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
    from .io.read.xenium import (
        create_polygon,
        import_segmentation_xenium_parquet,
        import_segmentation_xenium_zarr,
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
    _xenium_io_mod = _load_local_module("_xentools_io_read_xenium", os.path.join("io", "read", "xenium.py"))
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
    create_polygon = _xenium_io_mod.create_polygon
    import_segmentation_xenium_parquet = _xenium_io_mod.import_segmentation_xenium_parquet
    import_segmentation_xenium_zarr = _xenium_io_mod.import_segmentation_xenium_zarr
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

def read_json(json_file):
    """
    Reads a JSON file and returns the contents as a dictionary.
    Note that experiment.xenium is a JSON file, so this function can be used to read it.
    """
    with open(json_file) as f:
        metadata_dict = json.load(f)
    return metadata_dict

def _make_gene_panel_df(gene_panel_dict):
    """
    Converts the gene panel dictionary to a DataFrame.
    """
    return _read_make_gene_panel_df(gene_panel_dict)

def um_to_pixels(
        arr: Union[
        np.typing.ArrayLike,        # covers list, tuple, np.ndarray
        "pd.Series",          # forward ref; avoids hard dep on pandas
        ],
        pixel_size: float = 0.2125
    ) -> np.ndarray:

    """
    Convert array-like numerical input from microns to pixels.

    Parameters
    ----------
    arr : array-like
        Numerical data in microns. Can be list, tuple, np.ndarray, or pandas Series.
    pixel_size : float, default=0.2125
        Microns per pixel.

    Returns
    -------
    np.ndarray of int
        Converted values in pixels (rounded to nearest integer).
    """
    arr = np.asarray(arr, dtype=float)  # safely converts most array-like
    return np.round(arr / pixel_size).astype(int)

def read_xenium_to_anndata(xenium_output_folder, include_non_gene_features=False):
    return _read_xenium_to_anndata(
        xenium_output_folder,
        include_non_gene_features=include_non_gene_features,
    )

def read_xen_essentials(xenium_folder, verbose = True):
    panel_file = f'{xenium_folder}/gene_panel.json'
    cellboundaries_file = f'{xenium_folder}/cell_boundaries.parquet'
    transcripts_file = f'{xenium_folder}/transcripts.parquet'
    clusters_file = f'{xenium_folder}/analysis/clustering/gene_expression_graphclust/clusters.csv'
    nucboundaries_file = f'{xenium_folder}/nucleus_boundaries.parquet'
    

    """
    if verbose:
        print('Reading Cell Boundaries')
    celldata = pd.read_parquet(cellboundaries_file)
    celldata.set_index('cell_id', inplace=True)
    celldata.index = celldata.index.astype('str')
    celldata['cell'] = celldata.index.copy()

    if verbose:
        print('Reading Nuclear Boundaries')
    nuc = pd.read_parquet(nucboundaries_file)
    nuc.set_index('cell_id', inplace=True)
    nuc.index = nuc.index.astype('str')
    nuc['cell'] = nuc.index.copy()
    """

    if verbose:
        print('Reading Clusters')
    if os.path.exists(clusters_file):
        clusters = pd.read_csv(clusters_file, index_col=0)
        clusters.index = clusters.index.astype('str')
        clusters['Cluster'] = clusters['Cluster'].astype('str')
    else:
        clusters = pd.DataFrame(columns=['Cluster'])
    #celldata['cluster'] = celldata.index.map(clusters['Cluster'].to_dict())
    #celldata['cluster'] = celldata['cluster'].astype('category')
    color_key = dict(zip(clusters['Cluster'].unique(),sc.pl.palettes.default_28))
    #celldata['color'] = celldata['cluster'].map(color_key)
    
    if verbose:
        print('Reading Transcripts')
    trans = pd.read_parquet(transcripts_file)

    ## Sanitize in case of bytes:
    sample = trans["feature_name"].iloc[:100]
    # If any are bytes, run the C‐level vectorized decode
    if sample.map(lambda x: isinstance(x, (bytes, bytearray))).any():
    # This uses the fast C implementation under the hood
        trans["feature_name"] = trans["feature_name"].str.decode("utf-8")
    # 3) Ensure pandas knows it’s a true string column
    #trans["feature_name"] = trans["feature_name"].astype("string")

    gene_panel = read_xen_panel(panel_file) if os.path.exists(panel_file) else None

    return trans, clusters, gene_panel
    #return celldata, trans, nuc, clusters, gene_panel


def frame(transcripts_df):
    xmin,xmax = transcripts_df['x_location'].min(),transcripts_df['x_location'].max()
    ymin,ymax = transcripts_df['y_location'].min(),transcripts_df['y_location'].max()
    return np.array([[xmin,xmax],[ymin,ymax]])

def generate_palette(n, lightness=0.5, sat_min=0.5, sat_max=1.0, preview = False):
    """
    Generate a palette of n HEX colors.
    
    Colors are generated in HSL space with:
      - Hues evenly spaced across the circle (0 to 1)
      - Lightness fixed to a user-specified value (default 0.5)
      - Saturation randomly sampled between sat_min and sat_max
      
    Args:
        n (int): Number of colors to generate.
        lightness (float): Fixed lightness value (0 to 1).
        sat_min (float): Minimum saturation value (0 to 1).
        sat_max (float): Maximum saturation value (0 to 1).

    Returns:
        List[str]: List of HEX color strings.
    """
    import colorsys
    import random

    palette = []
    # Evenly space hues to maximize contrast.
    hues = [i / n for i in range(n)]
    for h in hues:
        # Randomize saturation for variation.
        s = random.uniform(sat_min, sat_max)
        # colorsys uses HLS ordering: (hue, lightness, saturation)
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))
        palette.append(hex_color)
    if preview:
        plot_palette(palette)
        
    return palette
    
def plot_palette(palette):
    """
    Plot a palette of colors as a horizontal line.
    
    Args:
        palette (List[str]): List of HEX color strings.
    """
    n = len(palette)
    fig, ax = plt.subplots(figsize=(n, 2))
    
    # Draw each color as a rectangle
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Hide axes
    plt.show()


class XenData:
    def __init__(self, xenium_folder, verbose=True, roi_file=None, crop_to_selection=None,
                 cache_threshold=5_000_000,
                 transcript_source: Literal['auto', 'zarr', 'parquet']='auto',
                 eager_transcript_threshold: int=20_000_000,
                 boundary_source: Literal['auto', 'parquet', 'zarr']='auto',
                 lazy_boundaries: bool=False,
                 include_non_gene_features: bool=False):
        self.xenium_folder = xenium_folder
        self.cache_threshold = cache_threshold
        self.eager_transcript_threshold = int(eager_transcript_threshold)
        loaded = load_xenium_folder(
            xenium_folder,
            transcript_source=transcript_source,
            cache_threshold=cache_threshold,
            eager_transcript_threshold=self.eager_transcript_threshold,
            boundary_source=boundary_source,
            lazy_boundaries=lazy_boundaries,
            include_non_gene_features=include_non_gene_features,
            verbose=verbose,
        )

        transcripts = loaded.transcripts
        self.trans = transcripts.trans
        self.n_transcripts = transcripts.n_transcripts
        self._transcripts_fmt = transcripts.transcript_format

        cells = loaded.cells
        self.adata = cells.adata
        self.clusters = cells.clusters
        self.gene_panel = cells.gene_panel

        boundaries = loaded.boundaries
        self.cell_boundaries = boundaries.cell_boundaries
        self.nucleus_boundaries = boundaries.nucleus_boundaries
        self._boundary_source = boundaries.boundary_source
        self._lazy_boundaries = boundaries.lazy_boundaries

        metadata = loaded.metadata
        self.xenium_metadata = metadata.metadata
        self.pixel_size = metadata.pixel_size
        self.name = metadata.name

        self.ROIs = ROICollection()
        self.rois = self.ROIs
        self.active_roi = None
        self.subset_roi = None
        self.update_cell_names()

        self.images = metadata.images
        self.protein_images = metadata.protein_images

        if roi_file is not None:
            if verbose:
                print(f'Importing ROIs from file: {roi_file}')
            imported_roi_names = self.import_ROI(
                roi_file,
                scale_geojson=True,
                append=False,
            )
            crop_target = _resolve_roi_selection_selector(self.ROIs, imported_roi_names, crop_to_selection)
            if crop_target is not None:
                if verbose:
                    print(f'Subsetting to ROI selection/class: {crop_target}')
                self.subset_to_roi(crop_target)

    def update_cell_names(self):
        if isinstance(self.trans, LazyTranscripts):
            # Transcripts are lazy — cell list lives in adata
            if self.adata is not None:
                self._cell_names = self.adata.obs_names.tolist()
            else:
                self._cell_names = []
        elif 'cell_id' in self.trans.columns:
            self._cell_names = sorted(self.trans['cell_id'].unique())
            if 'UNASSIGNED' in self._cell_names:
                self._cell_names.remove('UNASSIGNED')
        elif self.adata is not None:
            # trans was materialized from LazyTranscripts (no cell_id col) —
            # use adata, which is already subsetted to the ROI at this point
            self._cell_names = self.adata.obs_names.tolist()
        else:
            self._cell_names = []

    @property
    def cell_names(self):
        return self._cell_names
    #@property
    #def cell_names(self):
    #    cell_names = sorted(self.trans['cell_id'].unique())
    #    cell_names.remove('UNASSIGNED')
    #    return cell_names

    @property
    def frame(self):
        """
        Returns the frame of the transcript data as a 2D numpy array.
        The frame is defined by the minimum and maximum x and y coordinates of the transcripts.
        """
        if isinstance(self.trans, LazyTranscripts):
            return self.trans.frame
        return frame(self.trans)
    
    @property
    def aspect_ratio(self):
        """
        Returns the aspect ratio of the transcript data.
        The aspect ratio is defined as the width divided by the height of the frame.
        """
        return (self.frame[0,1] - self.frame[0,0]) / (self.frame[1,1] - self.frame[1,0])

    @property
    def xmin(self):
        return self.frame[0,0]
    @property
    def xmax(self):
        return self.frame[0,1]
    @property
    def ymin(self):
        return self.frame[1,0]
    @property
    def ymax(self):
        return self.frame[1,1]

    @property
    def features(self):
        features = [name for name in self.adata.var.index if 'Codeword' not in name and 'ControlProbe' not in name]
        return features
    
    def __str__(self):
        """
        Returns a human-readable string summarizing the key statistics of the XenData object.
        """
        def _boundary_status(boundary_obj, kind):
            if boundary_obj is None:
                return f"{kind}: unavailable"
            if isinstance(boundary_obj, LazyBoundaryGeoDataFrame):
                state = "loaded" if boundary_obj._data is not None else "lazy"
                return f"{kind}: available ({self._boundary_source}, {state})"
            try:
                return f"{kind}: available ({self._boundary_source}, {len(boundary_obj):,})"
            except Exception:
                return f"{kind}: available ({self._boundary_source})"

        def _roi_status():
            summary = self.ROIs.summary()
            parts = [f"{summary['n_selections']} selections"]
            if summary["n_classes"]:
                parts.append(f"{summary['n_classes']} classes")
            if self.active_roi is not None:
                parts.append(f"active={getattr(self.active_roi, 'name', 'roi')}")
            if self.subset_roi is not None:
                parts.append(f"subset={getattr(self.subset_roi, 'name', 'roi')}")
            return ", ".join(parts)

        # Run info
        preserve = self.xenium_metadata.get('preservation_method','Unknown')
        major = self.xenium_metadata['major_version']
        minor = self.xenium_metadata['minor_version']
        patch = self.xenium_metadata['patch_version']
        kit_version = f'{preserve} v{major}.{minor}.{patch}'

        # Cell data statistics
        total_cells = len(self.cell_names)
        #unassigned_cells = 1 if 'UNASSIGNED' in self.trans['cell_id'].unique() else 0
        
        # Feature information
        
        predesigned = self.xenium_metadata.get('panel_predesigned_id',None)
        panel_name = self.xenium_metadata.get('panel_name',None)
        organism = self.xenium_metadata.get('panel_organism', 'Unknown Organism')
        panel_info = ' '.join([organism,predesigned,panel_name])
        total_genes = len(self.features)
        
        # Cluster statistics
        total_clusters = len(self.clusters['Cluster'].unique()) if self.clusters is not None else 0
        transcript_repr = (
            "LazyTranscripts[zarr]"
            if isinstance(self.trans, LazyTranscripts)
            else f"DataFrame[parquet] ({len(self.trans):,} rows)"
        )
        adata_repr = (
            f"{self.adata.n_obs:,} cells x {self.adata.n_vars:,} genes"
            if hasattr(self, 'adata') and self.adata is not None
            else "unavailable"
        )
        cluster_repr = (
            f"{len(self.clusters.columns):,} columns, {total_clusters:,} unique clusters"
            if self.clusters is not None and len(self.clusters.columns)
            else "unavailable"
        )
        
        # Build the summary string
        summary = []
        summary.append(f"XenData Summary:")
        summary.append(f"-" * 50)
        summary.append(f"Slide/Region Name: {self.name}")
        summary.append(f"Folder: {self.xenium_folder}")
        summary.append(f"Xenium Kit Version: {kit_version}")
        summary.append(f"Panel: {panel_info}")
        summary.append(f"Transcripts: {transcript_repr}")
        summary.append(f"AnnData: {adata_repr}")
        summary.append(f"Number of Unique Cells: {total_cells:,}")
        summary.append(f"Number of Genes: {total_genes:,}")
        summary.append(f"Clusters: {cluster_repr}")
        #summary.append(f"Cells Marked as Unassigned: {'Yes' if unassigned_cells > 0 else 'No'}")
        
        if hasattr(self, 'adata') and self.adata is not None:
            # Additional AnnData statistics if available
            summary.append(f"Number of Transcripts: {self.adata.X.nnz:,}")
            #summary.append(f"Number of Nuclei: {self.adata.obs['nuclei'].sum() if 'nuclei' in self.adata.obs.columns else 'N/A'}")

        summary.append(f"Boundaries: {_boundary_status(self.cell_boundaries, 'cell')}; {_boundary_status(self.nucleus_boundaries, 'nucleus')}")
        summary.append(f"ROIs: {_roi_status()}")
        
        if self.images != {}:
            summary.append(f"Image Layers: {', '.join(self.images.keys())}")

        summary.append(f"-" * 50)
        return "\n".join(summary)

    def __repr__(self):
        def _line(label, value):
            return f"{label:<18} {value}"

        lines = [f"XenData object: {self.name}"]
        lines.append(_line("Folder", self.xenium_folder))

        if isinstance(self.trans, LazyTranscripts):
            trans_desc = (
                f"`trans`: LazyTranscripts[zarr] "
                f"({self.trans.n_transcripts:,} transcripts, {len(self.trans._gene_names):,} genes)"
            )
        else:
            trans_desc = f"`trans`: DataFrame[parquet] {self.trans.shape}"
        lines.append(_line("Transcripts", trans_desc))

        if self.adata is not None:
            lines.append(_line("Expression", f"`adata`: AnnData {self.adata.shape}"))

        if self.clusters is not None and len(self.clusters.columns):
            lines.append(_line("Clusters", f"`clusters`: DataFrame {self.clusters.shape}"))

        cell_boundary_desc = "unavailable"
        if self.cell_boundaries is not None:
            if isinstance(self.cell_boundaries, LazyBoundaryGeoDataFrame):
                state = "loaded" if self.cell_boundaries._data is not None else "lazy"
                cell_boundary_desc = f"`cell_boundaries`: GeoDataFrame ({self._boundary_source}, {state})"
            else:
                cell_boundary_desc = f"`cell_boundaries`: GeoDataFrame {self.cell_boundaries.shape}"

        nuc_boundary_desc = "unavailable"
        if self.nucleus_boundaries is not None:
            if isinstance(self.nucleus_boundaries, LazyBoundaryGeoDataFrame):
                state = "loaded" if self.nucleus_boundaries._data is not None else "lazy"
                nuc_boundary_desc = f"`nucleus_boundaries`: GeoDataFrame ({self._boundary_source}, {state})"
            else:
                nuc_boundary_desc = f"`nucleus_boundaries`: GeoDataFrame {self.nucleus_boundaries.shape}"

        lines.append(_line("Boundaries", cell_boundary_desc))
        lines.append(_line("", nuc_boundary_desc))

        if self.images:
            image_parts = []
            if 'DAPI' in self.images:
                image_parts.append("`images['DAPI']`")
            if self.protein_images is not None:
                n_ch = len(self.protein_images.get('channel_names', []))
                image_parts.append(f"`protein_images` ({n_ch} channels)")
            lines.append(_line("Images", ", ".join(image_parts)))

        roi_summary = self.ROIs.summary()
        roi_desc = (
            f"`ROIs`: {roi_summary['n_selections']} selections, "
            f"{roi_summary['n_classes']} classes"
        )
        if self.active_roi is not None:
            roi_desc += f"; active=`{getattr(self.active_roi, 'name', 'roi')}`"
        if self.subset_roi is not None:
            roi_desc += f"; subset=`{getattr(self.subset_roi, 'name', 'roi')}`"
        lines.append(_line("ROIs", roi_desc))

        lines.append("Common accessors:")
        lines.append("  `xdata.trans`, `xdata.adata`, `xdata.clusters`, `xdata.cell_boundaries`,")
        lines.append("  `xdata.nucleus_boundaries`, `xdata.ROIs`, `xdata.images`")
        return "\n".join(lines)
    
    def import_ROI(self,
                   roi_file,
                   roi_name: Union[str, int, list, None]=None,
                   plot_ROIs: bool = False,
                   scale_geojson: bool = True,
                   append: bool = True):
        """
        Import one or more ROIs from either a legacy Xenium Analyzer CSV or a GeoJSON file.
        Imported ROIs are stored in ``self.ROIs`` as an ``ROICollection`` that
        preserves both flat selections and Explorer-style ROI classes.

        Parameters:
        - roi_file: path to a ROI CSV or GeoJSON file
        - roi_name: ROI selector. For CSV, this is the Selection name or list of names.
          For GeoJSON, this can be a feature name, feature index, or list of either.
        - plot_ROIs: If True, plots the imported ROIs over a scatter of cell centroids.
        - scale_geojson: If True, scales GeoJSON coordinates by ``self.pixel_size``.
        - append: If True (default), add these ROIs to the existing collection.
          If False, replace the existing ROI collection before importing.
        """
        roi_names = self.ROIs.import_file(
            roi_file=roi_file,
            roi_name=roi_name,
            pixel_size=self.pixel_size,
            scale_geojson=scale_geojson,
            append=append,
        )

        summary = self.ROIs.summary()
        if summary["n_classes"]:
            class_detail = ", ".join(
                f"{name} ({count})" for name, count in summary["classes"].items()
            )
            print(
                f"Imported {len(roi_names)} ROI selection(s). "
                f"Collection now contains {summary['n_selections']} selection(s) "
                f"across {summary['n_classes']} class(es): {class_detail}"
            )
        else:
            print(
                f"Imported {len(roi_names)} ROI selection(s). "
                f"Collection now contains {summary['n_selections']} selection(s)."
            )

        if self.subset_roi is None and self.ROIs:
            self.active_roi = self.ROIs.union

        ### OPTIONAL: plot the imported ROIs
        # Sample 100k cells for faster plotting
        if plot_ROIs:
        
            n_cells = self.adata.obsm['spatial'].shape[0]
            if n_cells > 100000:
                sample_idx = np.random.choice(n_cells, size=100000, replace=False)
                cell_coords = self.adata.obsm['spatial'][sample_idx,:]
            else:
                cell_coords = self.adata.obsm['spatial']
            
            x, y = cell_coords[:,0], cell_coords[:,1]
            fig_scale = 4
            fig, ax = plt.subplots(figsize=[fig_scale, fig_scale * self.aspect_ratio])
            ax.scatter(x, y, s=1, color='white', alpha=0.1)
            ax.set_facecolor('black')

            for roi_name, roi in self.ROIs.items():
                roi.plot(ax)
                # Annotate with ROI name at centroid
                cx, cy = roi.centroid
                ax.text(cx, cy, 
                        roi_name, 
                        color='red', 
                        fontsize=12, 
                        fontweight='bold',
                        ha='center', 
                        va='center')

            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()

        return roi_names

    def import_ROI_xeniumanalyzer(self,
                                  roi_csv_file,
                                  roi_name: Union[str, list, None]=None,
                                  plot_ROIs: bool = False):
        """
        Backward-compatible wrapper for importing ROI files.
        Legacy Xenium Analyzer CSV files and newer GeoJSON ROI exports are both supported.
        """
        return self.import_ROI(
            roi_file=roi_csv_file,
            roi_name=roi_name,
            plot_ROIs=plot_ROIs,
            scale_geojson=True,
        )


    def subset_to_roi(self, ROI, selection=None, scale_geojson=True, inplace=True):
        """
        Subset the data to a specified region of interest (ROI).

        Parameters:
        - ROI: how to specify the ROI. Can be:
            - path to a CSV file containing ROI coordinates (see read_ROI_from_csv)
            - path to a GeoJSON file containing one or more polygon features
            - a numpy array of x,y coordinates defining the ROI polygon: [[x1, y1], [x2, y2], ...]
            - a pandas DataFrame with two columns defining the ROI polygon
            - a shapely geometry object
            - a key from self.ROIs (string name of a previously imported ROI)
        Options:
        - selection: if ROI is a file, the name or feature selector to choose (if multiple are present)
        - scale_geojson: if True, GeoJSON coordinates are scaled by ``self.pixel_size``.
          This is the default because Xenium GeoJSON annotations are commonly stored in pixels.
        - inplace: if True (default), modify this object in place and return None.
          If False, return a new subsetted XenData object, leaving this one unchanged.
        Returns:
        - None if inplace=True, or a new XenData object if inplace=False.
        """
        if not inplace:
            obj = self.copy()
            obj.subset_to_roi(ROI, selection=selection, scale_geojson=scale_geojson, inplace=True)
            return obj

        if isinstance(ROI, str) and ROI in self.ROIs:
            roi_obj = self.ROIs.resolve(ROI)
        else:
            roi_obj = _coerce_roi(
                ROI,
                selection=selection,
                pixel_size=self.pixel_size,
                scale_geojson=scale_geojson,
            )

        roi_polygon = roi_obj.geometry
        
        # Apply the filter to the DataFrame
        if isinstance(self.trans, LazyTranscripts):
            # Use polygon bbox for efficient tile selection, then apply exact polygon mask
            df = roi_obj.query_lazy_transcripts(self.trans, quality="all")
            self.trans = roi_obj.crop_dataframe(df).copy()
            # Filter adata spatially by cell centroid so update_cell_names
            # (called below) reads the already-subsetted obs_names
            if self.adata is not None and 'x_centroid' in self.adata.obs.columns:
                cx = self.adata.obs['x_centroid'].values
                cy = self.adata.obs['y_centroid'].values
                in_roi = roi_obj.contains_points(cx, cy)
                self.adata = self.adata[in_roi, :].copy()
        else:
            self.trans = roi_obj.crop_dataframe(self.trans).copy()

        # Filter celldata
        self.update_cell_names()

        if self.cell_boundaries is not None:
            # Prefer ID-based filtering; fall back to centroid-in-polygon when
            # ID systems differ (e.g. zarr integer IDs vs parquet hash IDs).
            id_overlap = [n for n in self.cell_names if n in self.cell_boundaries.index]
            if id_overlap:
                self.cell_boundaries = self.cell_boundaries.loc[id_overlap]
            else:
                cx = self.cell_boundaries.geometry.centroid.x
                cy = self.cell_boundaries.geometry.centroid.y
                in_roi = roi_obj.contains_points(cx.values, cy.values)
                self.cell_boundaries = self.cell_boundaries[in_roi]

        if self.nucleus_boundaries is not None:
            id_overlap = [n for n in self.cell_names if n in self.nucleus_boundaries.index]
            if id_overlap:
                self.nucleus_boundaries = self.nucleus_boundaries.loc[id_overlap]
            else:
                cx = self.nucleus_boundaries.geometry.centroid.x
                cy = self.nucleus_boundaries.geometry.centroid.y
                in_roi = roi_obj.contains_points(cx.values, cy.values)
                self.nucleus_boundaries = self.nucleus_boundaries[in_roi]

        if self.clusters is not None and len(self.clusters):
            keep_cells = [name for name in self.cell_names if name in self.clusters.index]
            self.clusters = self.clusters.loc[keep_cells]

        if self.adata is not None and self.cell_names:
            keep_cells = [name for name in self.cell_names if name in self.adata.obs_names]
            self.adata = self.adata[keep_cells, :].copy()

        self.area = roi_polygon.area
        self.subset_roi = roi_obj
        self.active_roi = roi_obj

    def crop_to_ROI(self, ROI, selection=None, scale_geojson=True, inplace=True):
        """
        Backward-compatible alias for ``subset_to_roi``.
        """
        return self.subset_to_roi(
            ROI,
            selection=selection,
            scale_geojson=scale_geojson,
            inplace=inplace,
        )
        
    def copy(self):
        import copy
        return copy.deepcopy(self)

    def set_active_roi(self, selector):
        """
        Set the default plotting ROI without cropping the underlying data.

        ``selector`` may be an ROI object, ROI class, selection/class name, or
        integer selection index.
        """
        self.active_roi = self.ROIs.resolve(selector)
        return self.active_roi

    def clear_active_roi(self):
        """
        Clear the active plotting ROI. If the dataset was actually subsetted,
        fall back to the subset ROI instead of exposing out-of-scope extents.
        """
        self.active_roi = self.subset_roi

    @property
    def subset_roi(self):
        """
        ROI used to materially subset this object, if any.
        """
        return self._subset_roi

    @subset_roi.setter
    def subset_roi(self, value):
        self._subset_roi = value

    @property
    def cropped_roi(self):
        """
        Backward-compatible alias for ``subset_roi``.
        """
        return self._subset_roi

    @cropped_roi.setter
    def cropped_roi(self, value):
        self._subset_roi = value

    def write_xenium_explorer(
        self,
        output_dir,
        include_morphology: bool = True,
        crop_morphology: bool = True,
        rebase_coordinates: bool = True,
        pyramidal_morphology: bool = True,
        pyramid_scale: int = 2,
        morphology_tile: tuple = (1024, 1024),
        overwrite: bool = False,
        verbose: bool = True,
    ):
        """
        Write this XenData object to a Xenium Explorer-compatible bundle.

        The resulting directory can be opened directly in the Xenium Explorer
        desktop application and is re-readable by ``XenData(output_dir)``.
        """
        return _write_xenium_explorer_bundle(
            self,
            output_dir,
            include_morphology=include_morphology,
            crop_morphology=crop_morphology,
            rebase_coordinates=rebase_coordinates,
            pyramidal_morphology=pyramidal_morphology,
            pyramid_scale=pyramid_scale,
            morphology_tile=morphology_tile,
            overwrite=overwrite,
            verbose=verbose,
        )

    def write_geo_submission(self,
                             output_dir,
                             matrix_format: str = "mex",
                             include_morphology: bool = True,
                             crop_morphology: bool = True,
                             include_protein_images: bool = True,
                             crop_protein_images: bool = True,
                             rebase_coordinates: bool = True,
                             pyramidal_morphology: bool = True,
                             pyramid_scale: int = 2,
                             morphology_tile: tuple[int, int] = (1024, 1024),
                             overwrite: bool = False):
        """
        Write the current XenData slice to GEO-friendly Xenium-style outputs.

        Written files:
        - morphology.ome.tif
        - transcripts.parquet
        - barcodes.tsv, features.tsv, matrix.mtx
        - cells.parquet
        - cell_boundaries.parquet
        - nucleus_boundaries.parquet

        Notes:
        - ``matrix_format='mex'`` is currently supported.
        - If this object has been subsetted and ``crop_morphology=True``, the
          morphology OME-TIFF is subset to the ROI bounding box in pixel coordinates.
        - Subsetted morphology exports are written as pyramidal OME-TIFFs by default.
        - Detected linked protein images in ``morphology_focus`` can also be exported.
        - Spatial tables are rebased to the subset image origin by default.
        """
        return _write_geo_submission_bundle(
            self,
            output_dir,
            matrix_format=matrix_format,
            include_morphology=include_morphology,
            crop_morphology=crop_morphology,
            include_protein_images=include_protein_images,
            crop_protein_images=crop_protein_images,
            rebase_coordinates=rebase_coordinates,
            pyramidal_morphology=pyramidal_morphology,
            pyramid_scale=pyramid_scale,
            morphology_tile=morphology_tile,
            overwrite=overwrite,
        )

    def rasterize(self, 
        features=None, 
        bin_size=10,
        colormap='Greys_r',
        vmax=None,
        vmin=None,
        title: str='',
        return_img = False):
        """
        Rasterizes the transcript data for specified features into a binned image.
        Parameters:
        - features: list of feature names to rasterize. If None, all features are used.
        - bin_size: size of the bins for rasterization.
        - colormap: colormap to use for the image.
        - vmax: maximum value for normalization (optional).
        - vmin: minimum value for normalization (optional).
        - title: title for the plot.
        - return_img: if True, return the PIL Image object.
        Returns:
        - img: a PIL Image object representing the rasterized data.
        """
        return _pl_namespace.rasterize(
            self,
            features=features,
            bin_size=bin_size,
            colormap=colormap,
            vmax=vmax,
            vmin=vmin,
            title=title,
            return_img=return_img,
        )

    def show_image(self,
                   channel: str = 'DAPI',
                   bounds=None,
                   figsize: Optional[tuple] = None,
                   dpi: Optional[int] = None,
                   cmap: Optional[str] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   z_index: Optional[int] = None,
                   level: Optional[int] = None,
                   micron_coords: Optional[bool] = None,
                   ax=None,
                   clip_percentile: float = 99.5,
                   verbose: bool = True):
        """
        Display an OME-TIFF image from this XenData object in a Jupyter-friendly way.

        Delegates to the module-level ``show_ome_tiff``, automatically passing the
        correct file path, pixel size, and active ROI crop for this dataset.

        Parameters
        ----------
        channel : str
            Which image channel to display. Use 'DAPI' (default) for the morphology
            image (morphology.ome.tif). For protein channels, pass the channel name
            as listed in ``self.protein_images['channel_names']``.
        bounds : tuple(xmin, xmax, ymin, ymax) or None
            Spatial window to display in µm. Overrides ``active_roi`` when provided.
            Automatically enables micron coordinates so the axes match ``splat()``
            and ``plot_boundaries()``. When None, falls back to ``active_roi`` if
            present, otherwise the full image.
        figsize : tuple, optional
            Figure size in inches (width, height). Defaults to matplotlib rcParams.
        dpi : int, optional
            Display DPI. Defaults to matplotlib rcParams.
        cmap : str, optional
            Colormap. Defaults to 'gray' for DAPI and 'magma' for protein channels.
        vmin, vmax : float, optional
            Intensity range for display. If vmax is None, it is set automatically
            using ``clip_percentile``.
        z_index : int, optional
            Which Z slice to show for multi-plane images. When None (default), a
            max-intensity projection across all Z slices is displayed.
        level : int, optional
            Override the auto-selected pyramid level (0 = full resolution).
        ax : matplotlib Axes, optional
            Axes to plot on. If None, a new figure is created.
        clip_percentile : float
            Percentile used to auto-set vmax when vmax is None. Default 99.5.
        verbose : bool
            If True, print the selected pyramid level and image shape.

        Returns
        -------
        ax : matplotlib Axes
        """
        from shapely.geometry import box as shapely_box

        # ── resolve ROI geometry ──────────────────────────────────────────────
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            roi = shapely_box(xmin, ymin, xmax, ymax)
            if micron_coords is None:
                micron_coords = True
        else:
            roi_obj = getattr(self, 'active_roi', None)
            roi = None if roi_obj is None else roi_obj.geometry
            if micron_coords is None:
                # XenData spatial overlays (boundaries, splats, ROIs) are all in
                # micron coordinates, so default to the same frame even for the
                # full image extent.
                micron_coords = True

        # ── resolve source file and default colormap ──────────────────────────
        if channel.upper() == 'DAPI':
            if 'DAPI' not in self.images:
                raise FileNotFoundError(
                    "No DAPI morphology image registered for this dataset."
                )
            image_path = self.images['DAPI']
            default_cmap = 'gray'
        else:
            if self.protein_images is None:
                raise ValueError(
                    "No protein images found for this dataset. "
                    "Expected a 'morphology_focus' folder with linked OME-TIFF files."
                )
            channel_names = self.protein_images['channel_names']
            if channel not in channel_names:
                raise ValueError(
                    f"Channel '{channel}' not found. "
                    f"Available protein channels: {channel_names}"
                )
            image_path = self.protein_images['files'][channel_names.index(channel)]
            default_cmap = 'magma'

        if cmap is None:
            cmap = default_cmap

        ax = show_ome_tiff(
            image_path,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            clip_percentile=clip_percentile,
            z_index=z_index,
            pixel_size=self.pixel_size,
            roi=roi,
            level=level,
            micron_coords=micron_coords,
            ax=ax,
            verbose=verbose,
        )
        ax.set_title(channel, fontsize=12)
        return ax

    def splat(self,
              genes=None,
              image_channel: Optional[str] = None,
              figsize: Optional[tuple] = None,
              dpi: Optional[int] = None,
              image_cmap: Optional[str] = None,
              image_vmin: Optional[float] = None,
              image_vmax: Optional[float] = None,
              image_clip_percentile: float = 99.5,
              image_level: Optional[int] = None,
              image_z_index: Optional[int] = None,
              image_alpha: float = 1.0,
              splat_alpha: float = 0.8,
              splat_cmap: str = 'hot',
              ax=None,
              **splat_kwargs):
        """
        Rasterize transcript data as a coloured image, optionally composited
        over a morphology or protein background image.

        This is the class-level entry point for transcript visualisation. All
        rasterization work is delegated to the module-level ``splat()`` function;
        this method adds automatic bounds derivation from the active ROI and,
        when ``channel`` is specified, background image rendering and RGBA
        compositing so that zero-signal pixels stay transparent.

        Parameters
        ----------
        genes : None | str | list[str] | dict
            Passed directly to the module-level ``splat()``.
            See its docstring for full options.
        image_channel : str or None
            Background image channel. ``None`` (default) renders transcripts
            only. Pass ``'DAPI'`` for the morphology image or a protein channel
            name (e.g. ``'CD3'``) to composite over a protein image.
        figsize : tuple, optional
            Figure size in inches. Defaults to matplotlib rcParams.
        dpi : int, optional
            Display DPI. Defaults to matplotlib rcParams.
        image_cmap : str, optional
            Colormap for the background image. Defaults to ``'gray'`` for DAPI
            and ``'magma'`` for protein channels.
        image_vmin, image_vmax : float, optional
            Intensity clipping for the background image.
        image_clip_percentile : float
            Auto-vmax percentile for the background image. Default 99.5.
        image_level : int, optional
            Override pyramid level for the background image.
        image_z_index : int, optional
            Z slice for the background image (``None`` = max projection).
        image_alpha : float
            Brightness of the background image (0 = black, 1 = full).
            Uses multiplicative dimming so black pixels stay black. Default 1.0.
        splat_alpha : float
            Maximum opacity of the transcript overlay (0 = invisible, 1 = opaque).
            Scales with signal intensity so zero-count pixels are fully transparent.
            Only applied when ``channel`` is set. Default 0.8.
        splat_cmap : str
            Colormap for single-channel splat output before RGBA compositing.
            Ignored for multi-channel (RGB) output. Default ``'hot'``.
        ax : matplotlib Axes, optional
            Axes to plot into. If ``None``, a new figure is created.
        **splat_kwargs
            Forwarded verbatim to the module-level ``splat()`` (e.g.
            ``pixel_size_um``, ``sigma_um``, ``gains``, ``smooth``).
            ``bounds`` is set automatically from the active ROI or the full
            transcript extent if not provided here.

        Returns
        -------
        rgb : np.ndarray
            Raw per-channel raster data (binned / smoothed counts).
        disp : np.ndarray
            Normalised display image, clipped to [0, 1].
        ax : matplotlib Axes
        """
        def _resolve_image_path(channel_name):
            if channel_name is None:
                return None
            if channel_name.upper() == 'DAPI':
                return self.images.get('DAPI')
            if self.protein_images is None:
                return None
            channel_names = self.protein_images['channel_names']
            if channel_name not in channel_names:
                return None
            return self.protein_images['files'][channel_names.index(channel_name)]

        # ── resolve bounds first so image and transcripts always match ───────
        if 'bounds' in splat_kwargs:
            bounds = splat_kwargs['bounds']
            if image_channel is not None:
                bounds = _pixel_aligned_bounds_um(bounds, self.pixel_size)
            splat_kwargs['bounds'] = bounds
        else:
            if getattr(self, 'active_roi', None) is not None:
                roi_bounds = _roi_bounds_um(self.active_roi)
                bounds = _pixel_aligned_bounds_um(roi_bounds, self.pixel_size)
            elif image_channel is not None:
                image_path = _resolve_image_path(image_channel)
                bounds = (
                    _image_extent_um(image_path, self.pixel_size)
                    if image_path is not None
                    else (self.xmin, self.xmax, self.ymin, self.ymax)
                )
            else:
                bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
            splat_kwargs['bounds'] = bounds

        effective_bounds = splat_kwargs['bounds']   # (xmin, xmax, ymin, ymax)

        # ── optional background image ─────────────────────────────────────────
        if image_channel is not None:
            ax = self.show_image(
                channel=image_channel,
                bounds=effective_bounds,
                figsize=figsize,
                dpi=dpi,
                cmap=image_cmap,
                vmin=image_vmin,
                vmax=image_vmax,
                clip_percentile=image_clip_percentile,
                level=image_level,
                z_index=image_z_index,
                ax=ax,
                verbose=False,
            )
            if image_alpha < 1.0:
                img_artist = ax.images[-1]
                vmin, vmax = img_artist.get_clim()
                img_artist.set_clim(vmin, vmax / image_alpha)

        # ── rasterize via module-level splat() ────────────────────────────────
        rgb, disp, ax = splat(self, genes=genes, ax=ax, **splat_kwargs)

        # ── RGBA composite (only when a background image is present) ─────────
        if image_channel is not None:
            n_ch = disp.shape[-1]
            if n_ch >= 3:
                signal = disp[..., :3].max(axis=-1)
                rgba = np.zeros((*disp.shape[:2], 4), dtype=np.float32)
                rgba[..., :3] = disp[..., :3]
            else:
                rgba = plt.get_cmap(splat_cmap)(disp[..., 0]).astype(np.float32)
                signal = disp[..., 0]
            rgba[..., 3] = np.clip(signal * splat_alpha, 0, 1)
            ax.images[-1].set_data(rgba)

        return rgb, disp, ax

    def plot_boundaries(
        self,
        kind: str = 'cell',
        color_by: Optional[str] = None,
        palette: Optional[dict] = None,
        facecolor='none',
        edgecolor='white',
        face_alpha: float = 0.3,
        edge_alpha: float = 0.8,
        linewidth: float = 0.5,
        bounds=None,
        ax=None,
        figsize: tuple = (8, 8),
        max_cells: Optional[int] = None,
    ):
        """
        Overlay cell or nucleus boundary polygons on an axes.

        Designed to compose with ``show_image`` and ``splat`` — pass the ``ax``
        returned by those methods to layer boundaries on top.

        Parameters
        ----------
        kind : {'cell', 'nucleus'}
            Which boundary set to draw. Default 'cell'.
        color_by : str or None
            Column name in ``adata.obs`` to use for per-cell fill colour (e.g.
            ``'Cluster'``). When None, all cells use ``facecolor``.
        palette : dict or None
            Mapping of category value → colour. Auto-generated when None.
        facecolor : colour spec
            Fill colour when ``color_by`` is None. Default ``'none'`` (transparent).
        edgecolor : colour spec
            Outline colour. Default ``'white'``.
        face_alpha : float
            Opacity of the fill (0–1). Applied independently of edge. Default 0.3.
        edge_alpha : float
            Opacity of the outline (0–1). Default 0.8.
        linewidth : float
            Outline width in points. Default 0.5.
        bounds : tuple(xmin, xmax, ymin, ymax) or None
            Spatial subset. Only cells whose centroid falls within these µm bounds
            are drawn. Derived from ``active_roi`` when None.
        ax : matplotlib Axes or None
            Axes to draw on. A new figure is created when None.
        figsize : tuple
            Figure size for the new figure when ``ax`` is None. Default (8, 8).
        max_cells : int or None
            If set, randomly subsample to at most this many cells (for speed
            when no ROI has been applied to a large dataset).

        Returns
        -------
        ax : matplotlib Axes
        """
        return _pl_namespace.plot_boundaries(
            self,
            kind=kind,
            color_by=color_by,
            palette=palette,
            facecolor=facecolor,
            edgecolor=edgecolor,
            face_alpha=face_alpha,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            bounds=bounds,
            ax=ax,
            figsize=figsize,
            max_cells=max_cells,
        )

    def plot_unassigned_transcripts(
        self,
        bin_size=4):
        """
        Plot unassigned transcripts in green and assigned transcripts in blue.
        """
        from PIL import Image

        df = self.trans
        x_edges, y_edges = create_bins(df, bin_size=bin_size)

        # Bin the data into a 2D histogram:
        # Each bin counts the number of transcripts whose (x,y) fall into that bin.
        # For some reason, I need to make the histogram with the y axis first followed by the x axis

        unass = df[df['cell_id'] == 'UNASSIGNED']
        ass = df[df['cell_id'] != 'UNASSIGNED']

        unass_counts, _, _ = np.histogram2d(unass['y_location'], unass['x_location'], bins=[y_edges, x_edges])
        ass_counts, _, _ = np.histogram2d(ass['y_location'], ass['x_location'], bins=[y_edges, x_edges])


        rgb_combined = np.zeros([ass_counts.shape[0],ass_counts.shape[1],3])
        # Assign images to respective channels:
        rgb_combined[:, :, 1] = unass_counts  # Green channel
        rgb_combined[:, :, 2] = ass_counts  # Blue channel
        img = Image.fromarray(np.uint8(rgb_combined))
        return img

    def create_binned_adata(self, 
        bin_size=5,
        exclude_unassigned=True,
        distance_to_nucleus: float=None,
        include_features: list=None,
        ):
        """
        Create a binned AnnData object from transcript data.
        Parameters:
        - xdata: XenData object containing transcript data.
        - bin_size: size of the bins for rasterization.
        - exclude_unassigned: Exclude transcripts labeled as 'UNASSIGNED' in 'cell_id' column
        - distance_to_nucleus (float): filter transcripts farther than this many microns from a nucleus
        

        Save the binned AnnData object to the xdata object.
        """
        self.binned_adata = _create_binned_adata(
            self,
            bin_size=bin_size,
            exclude_unassigned=exclude_unassigned,
            distance_to_nucleus=distance_to_nucleus,
            include_features=include_features,
        )

    def assign_cells_to_ROIs(self,
                              method: Literal['centroid', 'majority'] = 'centroid',
                              level: Literal['selection', 'class'] = 'selection',
                              key_added: str = 'roi',
                              min_overlap: float = 0.0):
        """
        Assign each cell to a named ROI from ``self.ROIs``, storing the result
        as a categorical column in ``self.adata.obs[key_added]``.

        Cells that fall outside every ROI are left as NaN.

        Parameters
        ----------
        method : 'centroid' | 'majority'
            'centroid' (default, fast) — a cell is assigned to whichever ROI
            contains its centroid. If ROIs overlap and a centroid falls in more
            than one, the last matching ROI in ``self.ROIs`` (insertion order)
            wins.

            'majority' (slower) — uses the full cell boundary polygon.  Each
            cell is assigned to the ROI that covers the largest fraction of its
            area.  Requires ``self.cell_boundaries`` to be loaded.
        level : 'selection' | 'class'
            Whether to assign by individual ROI selections (default) or by the
            union of all selections within each ROI class.
        key_added : str
            Column name written to ``self.adata.obs``. Default ``'roi'``.
        min_overlap : float
            *majority mode only.* Minimum fractional area overlap (0–1) needed
            to count as an assignment.  Cells whose best ROI covers less than
            this fraction are left unassigned.  Default 0.0.

        Returns
        -------
        None — modifies ``self.adata.obs[key_added]`` in place.
        """
        if not self.ROIs:
            raise ValueError("No ROIs defined. Call import_ROI() first.")

        if level == 'selection':
            roi_items = list(self.ROIs.items())
        elif level == 'class':
            roi_items = [(name, roi_class.union) for name, roi_class in self.ROIs.classes.items()]
        else:
            raise ValueError("level must be 'selection' or 'class'.")

        roi_names = [name for name, _ in roi_items]
        labels = pd.Series(np.nan, index=self.adata.obs_names, dtype=object)

        if method == 'centroid':
            coords = self.adata.obsm['spatial']  # (n_cells, 2), microns
            for roi_name, roi_data in roi_items:
                mask = roi_data.contains_points(coords[:, 0], coords[:, 1])
                labels.iloc[mask] = roi_name

        elif method == 'majority':
            if self.cell_boundaries is None:
                raise ValueError(
                    "Cell boundaries are required for method='majority'. "
                    "Ensure cell boundaries were loaded from parquet or cells.zarr.zip."
                )
            from shapely.strtree import STRtree

            roi_geoms  = [roi.geometry for _, roi in roi_items]
            tree = STRtree(roi_geoms)

            for cell_id, row in self.cell_boundaries.iterrows():
                cell_geom = row.geometry
                if cell_geom is None or cell_geom.is_empty:
                    continue
                cell_area = cell_geom.area
                if cell_area == 0:
                    continue

                best_roi, best_frac = None, 0.0
                for idx in tree.query(cell_geom):
                    try:
                        frac = cell_geom.intersection(roi_geoms[idx]).area / cell_area
                    except Exception:
                            continue
                    if frac > best_frac:
                        best_frac, best_roi = frac, roi_names[idx]

                if best_roi is not None and best_frac >= min_overlap:
                    if cell_id in labels.index:
                        labels[cell_id] = best_roi
        else:
            raise ValueError("method must be 'centroid' or 'majority'.")

        self.adata.obs[key_added] = pd.Categorical(labels)

        counts = {n: int((labels == n).sum()) for n in roi_names
                  if n in labels.values}
        total  = int(labels.notna().sum())
        detail = ', '.join(f'{n}: {c:,}' for n, c in counts.items())
        print(f"Assigned {total:,}/{self.adata.n_obs:,} cells  [{detail}]")

    def assign_bins_to_ROIs(self,
                             level: Literal['selection', 'class'] = 'selection',
                             key_added: str = 'roi'):
        """
        Assign each spatial bin in ``self.binned_adata`` to a named ROI from
        ``self.ROIs``, using the bin centroid (in microns).

        Requires ``create_binned_adata()`` to have been run first.
        Bins that fall outside every ROI are left as NaN.

        Parameters
        ----------
        level : 'selection' | 'class'
            Whether to assign by individual ROI selections (default) or by the
            union of all selections within each ROI class.
        key_added : str
            Column name written to ``self.binned_adata.obs``. Default ``'roi'``.

        Returns
        -------
        None — modifies ``self.binned_adata.obs[key_added]`` in place.
        """
        if not hasattr(self, 'binned_adata'):
            raise AttributeError(
                "No binned AnnData found. Run create_binned_adata() first."
            )
        if not self.ROIs:
            raise ValueError("No ROIs defined. Call import_ROI() first.")

        bin_size = self.binned_adata.uns['bin_size']
        x_stored = self.binned_adata.obs['x_bin'].values.astype(float)
        y_stored = self.binned_adata.obs['y_bin'].values.astype(float)

        # y_stored = -(y_orig - ymid) where ymid = (ymax_orig - ymin_orig)/2.
        # Reverse: ymid = (max(y_stored) - min(y_stored)) / 2
        #          y_orig = ymid - y_stored
        ymid = (y_stored.max() - y_stored.min()) / 2.0
        y_orig = ymid - y_stored

        # Bin centroid in microns = (bin_index + 0.5) * bin_size
        x_um = (x_stored + 0.5) * bin_size
        y_um = (y_orig   + 0.5) * bin_size

        if level == 'selection':
            roi_items = list(self.ROIs.items())
        elif level == 'class':
            roi_items = [(name, roi_class.union) for name, roi_class in self.ROIs.classes.items()]
        else:
            raise ValueError("level must be 'selection' or 'class'.")

        roi_names = [name for name, _ in roi_items]
        labels = pd.Series(np.nan, index=self.binned_adata.obs_names, dtype=object)

        for roi_name, roi_data in roi_items:
            mask = roi_data.contains_points(x_um, y_um)
            labels.iloc[mask] = roi_name

        self.binned_adata.obs[key_added] = pd.Categorical(labels)

        counts = {n: int((labels == n).sum()) for n in roi_names
                  if n in labels.values}
        total  = int(labels.notna().sum())
        detail = ', '.join(f'{n}: {c:,}' for n, c in counts.items())
        print(f"Assigned {total:,}/{self.binned_adata.n_obs:,} bins  [{detail}]")

    def write_ome_tiff(self, genes, output_path, flip_y=False,
                       compression='zlib',
                       pyramidal: bool = True,
                       pyramid_scale: int = 2,
                       tile: tuple = (1024, 1024)):
        """
        Write a multi-layer OME-TIFF image for specified genes from the binned AnnData object.

        Parameters
        ----------
        genes : list of str
            Gene names to include as channels.
        output_path : str
            Destination path; must end with '.ome.tiff'.
        flip_y : bool
            Flip the y-axis before writing. Default False.
        compression : str
            Compression codec passed to tifffile. Default 'zlib'.
        pyramidal : bool
            Write a tiled pyramidal OME-TIFF (strongly recommended for large images
            and QuPath compatibility). Default True.
        pyramid_scale : int
            Downsampling factor between pyramid levels. Default 2.
        tile : tuple
            Tile size (height, width) in pixels. Default (1024, 1024).

        Note: The binned AnnData object must be created first using create_binned_adata().
        """
        return _write_ome_tiff_bundle(
            self,
            genes=genes,
            output_path=output_path,
            flip_y=flip_y,
            compression=compression,
            pyramidal=pyramidal,
            pyramid_scale=pyramid_scale,
            tile=tile,
        )
    
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
    elif isinstance(data, LazyTranscripts):
        df = data
    else:
        raise ValueError("data must be XenData, LazyTranscripts, or pd.DataFrame")

    if isinstance(df, LazyTranscripts):
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
