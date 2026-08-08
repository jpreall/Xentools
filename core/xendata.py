"""Core XenData object model."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_local_module(module_name, relative_path):
    """Load a module by path for standalone ``xentools.py`` imports."""
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


def _is_lazy_transcripts(obj) -> bool:
    """
    Detect LazyTranscripts-like objects without relying on class identity.
    """
    return (
        hasattr(obj, "query")
        and callable(getattr(obj, "query"))
        and hasattr(obj, "_gene_names")
        and hasattr(obj, "_tile_meta")
    )


try:
    from .coordinates import TransformRegistry
    from .boundaries import LazyBoundaryGeoDataFrame
    from .rois import ROICollection, _coerce_roi, _resolve_roi_selection_selector, _roi_bounds_um
except ImportError:
    from core.coordinates import TransformRegistry
    from core.boundaries import LazyBoundaryGeoDataFrame
    from core.rois import ROICollection, _coerce_roi, _resolve_roi_selection_selector, _roi_bounds_um

try:
    from ..analysis.binning import create_binned_adata as _create_binned_adata
    from ..analysis.neighborhoods import neighborhood_composition as _neighborhood_composition
    from ..analysis.housekeeping import find_housekeeping_genes as _find_housekeeping_genes
    from ..io.read.images import _image_extent_um, find_alignment_file as _find_alignment_file
    from ..io.read.images import load_aligned_image as _load_aligned_image
    from ..io.read.loader import load_xenium_folder
    from ..io.write.images import _pixel_aligned_bounds_um, write_ome_tiff as _write_ome_tiff_bundle
    from ..io.write.xenium import (
        write_geo_submission as _write_geo_submission_bundle,
        write_xenium_explorer as _write_xenium_explorer_bundle,
    )
    from ..utils.geometry import frame
    from ..pl._save import save_figure as _save_figure
    from .. import pl as _pl_namespace
    from .. import gene_sets as _gene_sets_namespace
except ImportError:
    _binning_mod = _load_local_module(
        "_xentools_analysis_binning_for_xendata",
        os.path.join("..", "analysis", "binning.py"),
    )
    _neighborhoods_mod = _load_local_module(
        "_xentools_analysis_neighborhoods_for_xendata",
        os.path.join("..", "analysis", "neighborhoods.py"),
    )
    _housekeeping_mod = _load_local_module(
        "_xentools_analysis_housekeeping_for_xendata",
        os.path.join("..", "analysis", "housekeeping.py"),
    )
    _images_read_mod = _load_local_module(
        "_xentools_io_read_images_for_xendata",
        os.path.join("..", "io", "read", "images.py"),
    )
    _loader_read_mod = _load_local_module(
        "_xentools_io_read_loader_for_xendata",
        os.path.join("..", "io", "read", "loader.py"),
    )
    _images_write_mod = _load_local_module(
        "_xentools_io_write_images_for_xendata",
        os.path.join("..", "io", "write", "images.py"),
    )
    _xenium_write_mod = _load_local_module(
        "_xentools_io_write_xenium_for_xendata",
        os.path.join("..", "io", "write", "xenium.py"),
    )
    _geometry_mod = _load_local_module(
        "_xentools_utils_geometry_for_xendata",
        os.path.join("..", "utils", "geometry.py"),
    )
    _pl_namespace = _load_local_module(
        "_xentools_pl_for_xendata",
        os.path.join("..", "pl", "__init__.py"),
    )
    _gene_sets_namespace = _load_local_module(
        "_xentools_gene_sets_for_xendata",
        os.path.join("..", "gene_sets", "__init__.py"),
    )
    _save_mod = _load_local_module(
        "_xentools_pl_save_for_xendata",
        os.path.join("..", "pl", "_save.py"),
    )

    _create_binned_adata = _binning_mod.create_binned_adata
    _neighborhood_composition = _neighborhoods_mod.neighborhood_composition
    _find_housekeeping_genes = _housekeeping_mod.find_housekeeping_genes
    _image_extent_um = _images_read_mod._image_extent_um
    _find_alignment_file = _images_read_mod.find_alignment_file
    _load_aligned_image = _images_read_mod.load_aligned_image
    load_xenium_folder = _loader_read_mod.load_xenium_folder
    _pixel_aligned_bounds_um = _images_write_mod._pixel_aligned_bounds_um
    _write_ome_tiff_bundle = _images_write_mod.write_ome_tiff
    _write_geo_submission_bundle = _xenium_write_mod.write_geo_submission
    _write_xenium_explorer_bundle = _xenium_write_mod.write_xenium_explorer
    frame = _geometry_mod.frame
    _save_figure = _save_mod.save_figure


create_bins = _pl_namespace.create_bins
show_ome_tiff = _pl_namespace.show_ome_tiff
show_aligned_image = _pl_namespace.show_aligned_image
splat = _pl_namespace.splat
points = _pl_namespace.points
plot_splat = _pl_namespace.plot_splat
plot_points = _pl_namespace.plot_points

__all__ = ["XenData"]


class XenData:
    def __init__(self, xenium_folder, verbose=True, roi_file=None, crop_to_selection=None,
                 cache_threshold=5_000_000,
                 cache_max_bytes: Optional[int]=512_000_000,
                 transcript_source: Literal['auto', 'zarr', 'parquet']='auto',
                 eager_transcript_threshold: int=20_000_000,
                 boundary_source: Literal['auto', 'parquet', 'zarr']='auto',
                 lazy_boundaries: bool=True,
                 include_non_gene_features: bool=False):
        self.xenium_folder = xenium_folder
        self.cache_threshold = cache_threshold
        self.cache_max_bytes = cache_max_bytes
        self.eager_transcript_threshold = int(eager_transcript_threshold)
        loaded = load_xenium_folder(
            xenium_folder,
            transcript_source=transcript_source,
            cache_threshold=cache_threshold,
            cache_max_bytes=cache_max_bytes,
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
        self.feature_metadata = cells.feature_metadata

        boundaries = loaded.boundaries
        self.cell_boundaries = boundaries.cell_boundaries
        self.nucleus_boundaries = boundaries.nucleus_boundaries
        self._boundary_source = boundaries.boundary_source
        self._lazy_boundaries = boundaries.lazy_boundaries

        metadata = loaded.metadata
        self.xenium_metadata = metadata.metadata
        self.pixel_size = metadata.pixel_size
        self.name = metadata.name

        self.rois = ROICollection()
        self.active_roi = None
        self.subset_roi = None
        self.update_cell_names()

        self.images = metadata.images
        self.protein_images = metadata.protein_images
        self.transforms = TransformRegistry()
        identity = np.eye(3, dtype=float)
        for element in ("points:transcripts", "points:cells", "shapes:cells", "shapes:nuclei", "shapes:rois"):
            self.transforms.register(element, identity, source="global")
        if "DAPI" in self.images:
            pixel_to_global = np.array(
                [[self.pixel_size, 0, 0], [0, self.pixel_size, 0], [0, 0, 1]],
                dtype=float,
            )
            self.transforms.register("image:DAPI", pixel_to_global, source="pixels:DAPI")

        if roi_file is not None:
            if verbose:
                print(f'Importing ROIs from file: {roi_file}')
            imported_roi_names = self.import_rois(
                roi_file,
                scale_geojson=True,
                append=False,
            )
            crop_target = _resolve_roi_selection_selector(self.rois, imported_roi_names, crop_to_selection)
            if crop_target is not None:
                if verbose:
                    print(f'Subsetting to ROI selection/class: {crop_target}')
                self.subset_to_roi(crop_target)

    def update_cell_names(self):
        if _is_lazy_transcripts(self.trans):
            # Transcripts are lazy — cell list lives in adata
            if self.adata is not None:
                self._cell_names = self.adata.obs_names.tolist()
            else:
                self._cell_names = []
        elif hasattr(self.trans, "columns") and 'cell_id' in self.trans.columns:
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
        if _is_lazy_transcripts(self.trans):
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
            if hasattr(boundary_obj, "loaded"):
                state = "loaded" if boundary_obj.loaded else "lazy"
                return f"{kind}: available ({self._boundary_source}, {state})"
            try:
                return f"{kind}: available ({self._boundary_source}, {len(boundary_obj):,})"
            except Exception:
                return f"{kind}: available ({self._boundary_source})"

        def _roi_status():
            summary = self.rois.summary()
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
            if _is_lazy_transcripts(self.trans)
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

        if _is_lazy_transcripts(self.trans):
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
            if hasattr(self.cell_boundaries, "loaded"):
                state = "loaded" if self.cell_boundaries.loaded else "lazy"
                cell_boundary_desc = f"`cell_boundaries`: GeoDataFrame ({self._boundary_source}, {state})"
            else:
                cell_boundary_desc = f"`cell_boundaries`: GeoDataFrame {self.cell_boundaries.shape}"

        nuc_boundary_desc = "unavailable"
        if self.nucleus_boundaries is not None:
            if hasattr(self.nucleus_boundaries, "loaded"):
                state = "loaded" if self.nucleus_boundaries.loaded else "lazy"
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
            aligned_names = [
                name for name, image in self.images.items()
                if hasattr(image, "source_to_fixed")
            ]
            if aligned_names:
                image_parts.append(f"aligned: {', '.join(map(repr, aligned_names))}")
            lines.append(_line("Images", ", ".join(image_parts)))

        roi_summary = self.rois.summary()
        roi_desc = (
            f"`rois`: {roi_summary['n_selections']} selections, "
            f"{roi_summary['n_classes']} classes"
        )
        if self.active_roi is not None:
            roi_desc += f"; active=`{getattr(self.active_roi, 'name', 'roi')}`"
        if self.subset_roi is not None:
            roi_desc += f"; subset=`{getattr(self.subset_roi, 'name', 'roi')}`"
        lines.append(_line("ROIs", roi_desc))

        lines.append("Common accessors:")
        lines.append("  `xdata.trans`, `xdata.adata`, `xdata.clusters`, `xdata.cell_boundaries`,")
        lines.append("  `xdata.nucleus_boundaries`, `xdata.rois`, `xdata.images`")
        return "\n".join(lines)
    
    def import_rois(self,
                   roi_file,
                   roi_name: Union[str, int, list, None]=None,
                   plot_rois: bool = False,
                   scale_geojson: bool = True,
                   append: bool = True):
        """
        Import one or more ROIs from either a legacy Xenium Analyzer CSV or a GeoJSON file.
        Imported ROIs are stored in ``self.rois`` as an ``ROICollection`` that
        preserves both flat selections and Explorer-style ROI classes.

        Parameters:
        - roi_file: path to a ROI CSV or GeoJSON file
        - roi_name: ROI selector. For CSV, this is the Selection name or list of names.
          For GeoJSON, this can be a feature name, feature index, or list of either.
        - plot_rois: If True, plots the imported ROIs over a scatter of cell centroids.
        - scale_geojson: If True, scales GeoJSON coordinates by ``self.pixel_size``.
        - append: If True (default), add these ROIs to the existing collection.
          If False, replace the existing ROI collection before importing.
        """
        roi_names = self.rois.import_file(
            roi_file=roi_file,
            roi_name=roi_name,
            pixel_size=self.pixel_size,
            scale_geojson=scale_geojson,
            append=append,
        )

        summary = self.rois.summary()
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

        if self.subset_roi is None and self.rois:
            self.active_roi = self.rois.union

        ### OPTIONAL: plot the imported ROIs
        # Sample 100k cells for faster plotting
        if plot_rois:
        
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

            for roi_name, roi in self.rois.items():
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

    def subset_to_roi(self, roi, selection=None, scale_geojson=True, inplace=True):
        """
        Subset the data to a specified region of interest (ROI).

        Parameters:
        - roi: how to specify the ROI. Can be:
            - path to a CSV file containing ROI coordinates (see read_roi_from_csv)
            - path to a GeoJSON file containing one or more polygon features
            - a numpy array of x,y coordinates defining the ROI polygon: [[x1, y1], [x2, y2], ...]
            - a pandas DataFrame with two columns defining the ROI polygon
            - a shapely geometry object
            - a key from self.rois (string name of a previously imported ROI)
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
            obj.subset_to_roi(roi, selection=selection, scale_geojson=scale_geojson, inplace=True)
            return obj

        if isinstance(roi, str) and roi in self.rois:
            roi_obj = self.rois.resolve(roi)
        else:
            roi_obj = _coerce_roi(
                roi,
                selection=selection,
                pixel_size=self.pixel_size,
                scale_geojson=scale_geojson,
            )

        roi_polygon = roi_obj.geometry
        
        # Apply the filter to the DataFrame
        if _is_lazy_transcripts(self.trans):
            # Use polygon bbox for efficient tile selection, then apply exact polygon mask
            self.trans = roi_obj.query_lazy_transcripts(self.trans, quality="all").copy()
        else:
            self.trans = roi_obj.crop_dataframe(self.trans).copy()

        # Always subset cells from their centroids. Inferring cell membership
        # from retained transcript IDs fails for cells with no transcripts in
        # the ROI and for disjoint ROI groups backed by eager transcript data.
        if self.adata is not None:
            if {'x_centroid', 'y_centroid'}.issubset(self.adata.obs.columns):
                cell_xy = self.adata.obs[['x_centroid', 'y_centroid']].to_numpy(dtype=float)
            elif 'spatial' in self.adata.obsm:
                cell_xy = np.asarray(self.adata.obsm['spatial'], dtype=float)[:, :2]
            else:
                raise KeyError(
                    "Cannot crop cells: expected obs centroid columns or adata.obsm['spatial']."
                )
            in_roi = roi_obj.contains_points(cell_xy[:, 0], cell_xy[:, 1])
            self.adata = self.adata[np.asarray(in_roi, dtype=bool), :].copy()

        # Filter celldata
        self.update_cell_names()

        if self.cell_boundaries is not None:
            cell_boundaries = self.cell_boundaries
            if hasattr(cell_boundaries, "query_bounds"):
                cell_boundaries = cell_boundaries.query_bounds(_roi_bounds_um(roi_obj))
            # Prefer ID-based filtering; fall back to centroid-in-polygon when
            # ID systems differ (e.g. zarr integer IDs vs parquet hash IDs).
            id_overlap = [n for n in self.cell_names if n in cell_boundaries.index]
            if id_overlap:
                self.cell_boundaries = cell_boundaries.loc[id_overlap]
            else:
                cx = cell_boundaries.geometry.centroid.x
                cy = cell_boundaries.geometry.centroid.y
                in_roi = roi_obj.contains_points(cx.values, cy.values)
                self.cell_boundaries = cell_boundaries[in_roi]

        if self.nucleus_boundaries is not None:
            nucleus_boundaries = self.nucleus_boundaries
            if hasattr(nucleus_boundaries, "query_bounds"):
                nucleus_boundaries = nucleus_boundaries.query_bounds(_roi_bounds_um(roi_obj))
            id_overlap = [n for n in self.cell_names if n in nucleus_boundaries.index]
            if id_overlap:
                self.nucleus_boundaries = nucleus_boundaries.loc[id_overlap]
            else:
                cx = nucleus_boundaries.geometry.centroid.x
                cy = nucleus_boundaries.geometry.centroid.y
                in_roi = roi_obj.contains_points(cx.values, cy.values)
                self.nucleus_boundaries = nucleus_boundaries[in_roi]

        if self.clusters is not None and len(self.clusters):
            keep_cells = [name for name in self.cell_names if name in self.clusters.index]
            self.clusters = self.clusters.loc[keep_cells]

        if self.adata is not None and self.cell_names:
            keep_cells = [name for name in self.cell_names if name in self.adata.obs_names]
            self.adata = self.adata[keep_cells, :].copy()

        self.area = roi_polygon.area
        self.subset_roi = roi_obj
        self.active_roi = roi_obj

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def set_active_roi(self, selector):
        """
        Set the default plotting ROI without cropping the underlying data.

        ``selector`` may be an ROI object, ROI class, selection/class name, or
        integer selection index.
        """
        self.active_roi = self.rois.resolve(selector)
        return self.active_roi

    def create_roi_group(self, name, selectors, *, metadata=None, overwrite=False):
        """
        Group existing ROIs under one name without modifying the source ROIs.

        The resulting :class:`~xentools.ROIGroup` retains its members and
        exposes their geometric union, which may be a disjoint ``MultiPolygon``.
        The group name can be passed anywhere a named ROI is accepted, including
        :meth:`set_active_roi`, :meth:`subset_to_roi`, and
        :meth:`find_housekeeping_genes`.

        Parameters
        ----------
        name : str
            Unique name for the group, such as ``"all_liver"``.
        selectors : str, int, ROI-like, or sequence
            Existing selection, class, or group names; integer selection
            indices; or ROI-like objects to combine. A sequence may mix these
            selector types.
        metadata : dict, optional
            Group-level annotations, for example ``{"tissue": "liver"}``.
        overwrite : bool, default False
            Replace an existing group with the same name. Selection and class
            name conflicts are always rejected.

        Returns
        -------
        xentools.ROIGroup
            Logical group containing the resolved member ROIs.

        Examples
        --------
        >>> liver = xdata.create_roi_group(
        ...     "all_liver",
        ...     ["liver_piece_1", "liver_piece_2"],
        ...     metadata={"tissue": "liver"},
        ... )
        >>> liver.member_names
        ['liver_piece_1', 'liver_piece_2']
        >>> liver_only = xdata.subset_to_roi("all_liver", inplace=False)
        >>> ranking = xdata.find_housekeeping_genes(
        ...     rois=["all_liver", "all_kidney"]
        ... )

        Notes
        -----
        Overlapping members are unioned and therefore do not double-count
        cells. Cropping disjoint members uses exact polygon containment, though
        image bounds may include slide space between the pieces.
        """
        return self.rois.create_group(
            name,
            selectors,
            metadata=metadata,
            overwrite=overwrite,
        )

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
        return_img=False,
        save=None,
        save_kwargs: Optional[dict] = None):
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
            save=save,
            save_kwargs=save_kwargs,
        )

    def import_aligned_image(
        self,
        image_path,
        *,
        alignment_file="auto",
        name=None,
        metadata=None,
        overwrite=False,
        poor_alignment_rmse_px=25.0,
    ):
        """
        Register an external OME-TIFF using a Xenium Explorer alignment.

        Explorer alignment bundles (a ZIP or extracted folder) contain
        ``matrix.csv`` and usually ``keypoints.csv``. The matrix maps external-image pixels into Xenium
        morphology pixels; keypoints are used to verify direction and report
        residual error. The original pyramidal image is retained and transformed
        lazily during display rather than resampled into a large duplicate.

        Parameters
        ----------
        image_path : path-like
            External OME-TIFF, such as an H&E or immunofluorescence image.
        alignment_file : path-like or "auto"
            Explorer alignment ZIP, extracted folder, standalone 3×3
            ``matrix.csv``, or ``keypoints.csv`` beside its corresponding
            ``matrix.csv``. ``"auto"``
            searches beside the image and Xenium output and matches numeric sample
            identifiers in their filenames.
        name : str, optional
            Image registry name used by :meth:`show_image`. Defaults to ``"H&E"``
            for filenames ending in ``HE`` and otherwise to the image stem.
        metadata : dict, optional
            User annotations stored with the registration.
        overwrite : bool, default False
            Replace an existing image with the same registry name.
        poor_alignment_rmse_px : float, default 25
            Emit a diagnostic warning when keypoint RMSE exceeds this many fixed-
            image pixels.

        Returns
        -------
        xentools.AlignedImage
            Transform, source metadata, bounds, channels, and validation metrics.

        Examples
        --------
        >>> he = xdata.import_aligned_image(
        ...     "20260803_117451_HE.ome.tif",
        ...     alignment_file="117451_HE_alignment_files.zip",
        ... )
        >>> he.keypoint_rmse_px
        0.0
        >>> ax = xdata.show_image("H&E")
        >>> xdata.plot_boundaries(ax=ax)
        """
        from pathlib import Path
        import re
        import tifffile

        image_path = Path(image_path).expanduser().resolve()
        if alignment_file == "auto":
            alignment_file = _find_alignment_file(image_path, self.xenium_folder)
        if name is None:
            stem = image_path.name
            stem = re.sub(r"\.ome\.tiff?$", "", stem, flags=re.IGNORECASE)
            name = "H&E" if re.search(r"(?:^|[_-])HE$", stem, re.IGNORECASE) else stem
        if name in self.images and not overwrite:
            raise ValueError(
                f"Image name {name!r} is already registered. Pass overwrite=True "
                "or choose a different name."
            )
        fixed_path = self.images.get("DAPI")
        if fixed_path is None or not isinstance(fixed_path, (str, os.PathLike)):
            raise FileNotFoundError(
                "A native DAPI morphology image is required as the fixed alignment frame."
            )
        with tifffile.TiffFile(fixed_path) as tif:
            fixed_series = tif.series[0]
            fixed_shape = (
                int(fixed_series.shape[fixed_series.axes.index("Y")]),
                int(fixed_series.shape[fixed_series.axes.index("X")]),
            )
        registration = _load_aligned_image(
            image_path,
            alignment_file,
            name=name,
            fixed_shape=fixed_shape,
            fixed_pixel_size=self.pixel_size,
            poor_alignment_rmse_px=poor_alignment_rmse_px,
            metadata=metadata,
        )
        self.images[name] = registration
        self.transforms.register(
            f"image:{name}",
            registration.source_to_global,
            source=f"pixels:{name}",
            metadata={"alignment_file": registration.alignment_file},
            overwrite=overwrite,
        )
        return registration

    def show_image(self,
                   channel: str = 'DAPI',
                   bounds=None,
                   figsize: Optional[tuple] = None,
                   dpi: Optional[int] = None,
                   cmap: Optional[str] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   source_channel=None,
                   z_index: Optional[int] = None,
                   level: Optional[int] = None,
                   micron_coords: Optional[bool] = None,
                   ax=None,
                   clip_percentile: float = 99.5,
                   verbose: bool = True,
                   save=None,
                   save_kwargs: Optional[dict] = None):
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
        source_channel : str or int, optional
            Channel within a registered multi-channel external image. Ignored
            for native DAPI and linked morphology-focus channels.
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
        aligned_registration = None
        if channel.upper() == 'DAPI':
            if 'DAPI' not in self.images:
                raise FileNotFoundError(
                    "No DAPI morphology image registered for this dataset."
                )
            image_path = self.images['DAPI']
            default_cmap = 'gray'
        elif channel in self.images and hasattr(self.images[channel], "source_to_fixed"):
            aligned_registration = self.images[channel]
            default_cmap = None
            image_path = aligned_registration.path
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

        if aligned_registration is not None:
            ax = show_aligned_image(
                aligned_registration,
                bounds=None if roi is None else _roi_bounds_um(roi),
                channel=source_channel,
                z_index=z_index,
                level=level,
                figsize=figsize,
                dpi=dpi,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                clip_percentile=clip_percentile,
                ax=ax,
                verbose=verbose,
            )
        else:
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
        _save_figure(ax, save=save, save_kwargs=save_kwargs)
        return ax

    plot_image = show_image

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
              return_array=False,
              ax=None,
              save=None,
              save_kwargs: Optional[dict] = None,
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
        return_array : bool | {'display', 'raw'}
            Return behavior. ``False`` (default) returns the matplotlib axes.
            ``True`` or ``'display'`` returns the normalized display array used
            for plotting. ``'raw'`` returns the raw binned/smoothed raster before
            display normalization and gain clipping.
        ax : matplotlib Axes, optional
            Axes to plot into. If ``None``, a new figure is created.
        **splat_kwargs
            Forwarded verbatim to the module-level ``splat()`` (e.g.
            ``pixel_size_um``, ``target_pixels``, ``sigma_um``, ``gains``,
            ``smooth``). By default, ``pixel_size_um='auto'`` selects a
            display-oriented resolution from ``target_pixels``; pass a numeric
            ``pixel_size_um`` to force exact output resolution.
            ``bounds`` is set automatically from the active ROI or the full
            transcript extent if not provided here.

        Returns
        -------
        matplotlib.axes.Axes | numpy.ndarray
            Axes by default. If ``return_array`` is requested, returns either
            the normalized display array or the raw raster array.
        """
        def _resolve_image_source(channel_name):
            if channel_name is None:
                return None
            if channel_name.upper() == 'DAPI':
                return self.images.get('DAPI')
            if channel_name in self.images and hasattr(self.images[channel_name], "source_to_fixed"):
                return self.images[channel_name]
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
                image_source = _resolve_image_source(image_channel)
                if hasattr(image_source, "micron_bounds"):
                    bounds = image_source.micron_bounds
                elif image_source is not None:
                    bounds = _image_extent_um(image_source, self.pixel_size)
                else:
                    bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
            else:
                bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
            splat_kwargs['bounds'] = bounds

        effective_bounds = splat_kwargs['bounds']   # (xmin, xmax, ymin, ymax)
        existing_image_count = len(ax.images) if ax is not None else 0

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
        rgb, disp, ax = splat(self, genes=genes, ax=ax, return_array="_all", **splat_kwargs)

        # ── RGBA composite when plotting over an image ────────────────────────
        # Module-level splat() draws an opaque RGB image. When XenData.splat()
        # is used as an overlay, convert that newest image to RGBA so black /
        # zero-signal pixels are transparent and the existing image remains
        # visible underneath.
        if image_channel is not None or existing_image_count > 0:
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

        _save_figure(ax, save=save, save_kwargs=save_kwargs)
        if return_array in (False, None):
            return ax
        if return_array is True or return_array == "display":
            return disp
        if return_array == "raw":
            return rgb
        raise ValueError("return_array must be False, True, 'display', or 'raw'.")

    plot_splat = splat

    def plot_binned_splat(
        self,
        genes=None,
        *,
        bounds=None,
        gains=1.0,
        sigma_um: float = 0.0,
        smooth: bool = False,
        global_norm: bool = False,
        ax=None,
        figsize: Optional[tuple] = None,
        dpi: Optional[int] = None,
        show_ticks: bool = False,
        show_legend: bool = True,
        legend_loc: str = "outside right",
        splat_alpha: float = 0.8,
        splat_cmap: str = "hot",
        return_array=False,
        save=None,
        save_kwargs: Optional[dict] = None,
    ):
        """
        Plot gene or gene-set density from ``self.binned_adata``.

        This mirrors ``plot_splat`` but uses a precomputed binned AnnData matrix
        instead of rasterizing transcript coordinates. Run
        ``create_binned_adata()`` first.
        """
        if not hasattr(self, "binned_adata") or self.binned_adata is None:
            raise ValueError("No binned_adata found. Run create_binned_adata() first.")

        if bounds is None:
            if getattr(self, "active_roi", None) is not None:
                bounds = _roi_bounds_um(self.active_roi)
            else:
                bounds = (self.xmin, self.xmax, self.ymin, self.ymax)

        existing_image_count = len(ax.images) if ax is not None else 0
        rgb, disp, ax = _pl_namespace.plot_binned_splat(
            self,
            genes=genes,
            bounds=bounds,
            gains=gains,
            sigma_um=sigma_um,
            smooth=smooth,
            global_norm=global_norm,
            ax=ax,
            show_ticks=show_ticks,
            show_legend=show_legend,
            legend_loc=legend_loc,
            return_array="_all",
        )

        if existing_image_count > 0:
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

        if figsize is not None:
            ax.figure.set_size_inches(*figsize)
        if dpi is not None:
            ax.figure.set_dpi(dpi)

        _save_figure(ax, save=save, save_kwargs=save_kwargs)
        if return_array in (False, None):
            return ax
        if return_array is True or return_array == "display":
            return disp
        if return_array == "raw":
            return rgb
        raise ValueError("return_array must be False, True, 'display', or 'raw'.")

    def plot_boundaries(
        self,
        kind: str = 'cell',
        color_by: Optional[str] = None,
        genes=None,
        layer: Optional[str] = None,
        cmap: str = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        palette: Optional[dict] = None,
        facecolor='none',
        edgecolor='white',
        face_alpha: float = 0.3,
        edge_alpha: float = 0.2,
        linewidth: float = 0.5,
        bounds=None,
        ax=None,
        figsize: tuple = (8, 8),
        max_cells: Optional[int] = None,
        show_legend: bool = True,
        legend_loc: str = 'outside right',
        legend_title: Optional[str] = None,
        background: str = 'black',
        show_axis: bool = False,
        save=None,
        save_kwargs: Optional[dict] = None,
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
        genes : str, sequence, or None
            Gene or gene list to sum per cell and use as continuous fill colour.
            Mutually exclusive with ``color_by``.
        layer : str or None
            AnnData layer to use for expression values. Defaults to ``adata.X``.
        palette : dict or None
            Mapping of category value → colour. Auto-generated when None.
        facecolor : colour spec
            Fill colour when ``color_by`` is None. Default ``'none'`` (transparent).
        edgecolor : colour spec
            Outline colour. Default ``'white'``.
        face_alpha : float
            Opacity of the fill (0–1). Applied independently of edge. Default 0.3.
        edge_alpha : float
            Opacity of the outline (0–1). Default 0.2.
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
        show_legend : bool
            Whether to draw a categorical legend when ``color_by`` is used.
            Default True.
        legend_loc : str
            Legend placement. Defaults to ``'outside right'`` so labels do not
            cover the plotted boundaries. Also accepts ``'outside left'``,
            ``'outside bottom'``, ``'outside top'``, or any matplotlib legend
            location string.
        legend_title : str or None
            Override the legend title. Defaults to ``color_by``.
        background : colour spec
            Axes/figure background used only when creating a standalone axes.
            Ignored when plotting onto an existing ``ax``. Default ``'black'``.
        show_axis : bool
            Whether to show ticks and axis decorations for standalone boundary
            plots. Ignored for existing axes. Default False.

        Returns
        -------
        ax : matplotlib Axes
        """
        return _pl_namespace.plot_boundaries(
            self,
            kind=kind,
            color_by=color_by,
            genes=genes,
            layer=layer,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
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
            show_legend=show_legend,
            legend_loc=legend_loc,
            legend_title=legend_title,
            background=background,
            show_axis=show_axis,
            save=save,
            save_kwargs=save_kwargs,
        )

    def plot_cells(
        self,
        genes=None,
        color_by: Optional[str] = None,
        layer: Optional[str] = None,
        cmap: str = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        palette: Optional[dict] = None,
        facecolor='none',
        edgecolor='white',
        face_alpha: float = 0.8,
        edge_alpha: float = 0.2,
        linewidth: float = 0.5,
        bounds=None,
        ax=None,
        figsize: tuple = (8, 8),
        max_cells: Optional[int] = None,
        show_legend: bool = True,
        legend_loc: str = 'outside right',
        legend_title: Optional[str] = None,
        background: str = 'black',
        show_axis: bool = False,
        save=None,
        save_kwargs: Optional[dict] = None,
    ):
        """
        Plot cell boundary polygons.

        When ``genes`` is supplied, cells are filled by expression of that gene
        or by summed expression of the supplied gene list. When ``genes`` is
        omitted, ``color_by`` can be used for categorical ``adata.obs`` labels.
        """
        return _pl_namespace.plot_cells(
            self,
            genes=genes,
            color_by=color_by,
            layer=layer,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
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
            show_legend=show_legend,
            legend_loc=legend_loc,
            legend_title=legend_title,
            background=background,
            show_axis=show_axis,
            save=save,
            save_kwargs=save_kwargs,
        )

    def points(
        self,
        genes=None,
        *,
        bounds=None,
        quality: str = "all",
        max_points: Optional[int] = 100_000,
        random_state: Optional[int] = 0,
        color: Optional[str] = None,
        palette: Optional[dict] = None,
        cmap: str = "tab20",
        marker: str = "o",
        markers: Optional[dict] = None,
        s: float = 8,
        alpha: float = 0.75,
        linewidths: float = 0,
        edgecolors="none",
        ax=None,
        figsize=(8, 8),
        background: str = "black",
        show_axis: bool = False,
        show_legend: bool = True,
        legend_loc: str = "outside right",
        legend_title: Optional[str] = None,
        max_legend_items: int = 20,
        preserve_limits: bool = True,
        rasterized: bool = True,
        warn_on_sample: bool = True,
        assigned_only: Optional[bool] = None,
        save=None,
        save_kwargs: Optional[dict] = None,
        return_data: bool = False,
    ):
        """
        Plot individual transcripts as colored point markers.

        Defaults to the active ROI when available and caps rendering at
        ``max_points`` transcripts by random sampling. Pass an existing ``ax``
        to composite points over images, splats, or boundary plots.
        """
        return _pl_namespace.points(
            self,
            genes=genes,
            bounds=bounds,
            quality=quality,
            max_points=max_points,
            random_state=random_state,
            color=color,
            palette=palette,
            cmap=cmap,
            marker=marker,
            markers=markers,
            s=s,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
            ax=ax,
            figsize=figsize,
            background=background,
            show_axis=show_axis,
            show_legend=show_legend,
            legend_loc=legend_loc,
            legend_title=legend_title,
            max_legend_items=max_legend_items,
            preserve_limits=preserve_limits,
            rasterized=rasterized,
            warn_on_sample=warn_on_sample,
            assigned_only=assigned_only,
            save=save,
            save_kwargs=save_kwargs,
            return_data=return_data,
        )

    plot_points = points

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

    def neighborhood_composition(self, **kwargs):
        """
        Compute cell-neighborhood label composition for this object.

        By default this uses ``self.active_roi`` when present. Pass
        ``roi=None`` to compute over all cells, or ``roi=<name>`` to use a
        named ROI from ``self.rois``.
        """
        return _neighborhood_composition(self, **kwargs)

    def find_housekeeping_genes(self, **kwargs):
        """
        Find well-expressed genes with minimal variation across this object's ROIs.

        Raw counts are summarized within each ROI, filtered by expression and
        detection, and ranked by within- and between-ROI stability without
        library-size normalization.
        """
        return _find_housekeeping_genes(self, **kwargs)

    def plot_housekeeping_diagnostics(self, ranking, **kwargs):
        """Visualize a housekeeping-gene ranking produced for this object."""
        return _pl_namespace.plot_housekeeping_diagnostics(ranking, **kwargs)

    def gene_set_picker(self, **kwargs):
        """
        Create a gene-set picker tied to this XenData object.

        This is the recommended method-style spelling. ``GeneSetPicker`` is the
        class name; Python method names are conventionally snake_case.

        The picker is returned for users who want direct control, and the
        currently selected, filtered dictionary is mirrored to
        ``self.picked_gene_sets`` whenever selections change.
        """
        picker = _gene_sets_namespace.pick(self, **kwargs)
        self._gene_set_picker = picker
        self.picked_gene_sets = picker.gene_sets
        return picker

    def GeneSetPicker(self, **kwargs):
        """
        Alias for ``gene_set_picker()``.

        Kept for discoverability when users think in terms of the picker class
        name, but ``gene_set_picker`` is the preferred style.
        """
        return self.gene_set_picker(**kwargs)

    def niche_heatmap(self, **kwargs):
        """
        Plot mean neighborhood-composition features by niche label.
        """
        return _pl_namespace.niche_heatmap(self, **kwargs)

    def niche_map(self, **kwargs):
        """
        Plot cells spatially, colored by niche labels or niche-composition values.
        """
        return _pl_namespace.niche_map(self, **kwargs)

    def render(
        self,
        *,
        image=None,
        images=None,
        splat=None,
        binned_splat=None,
        points=None,
        cells=None,
        bounds=None,
        ax=None,
        figsize=(8, 8),
        dpi: Optional[int] = None,
        level: Optional[int] = None,
        background: str = "black",
        show_axis: bool = False,
        title: Optional[str] = None,
        legend=True,
        legend_loc: str = "outside right",
        legend_title: Optional[str] = None,
        legend_max_items: int = 30,
        save=None,
        save_kwargs: Optional[dict] = None,
    ):
        """
        Render a composite spatial view from image, transcript, and cell layers.

        ``render`` is the high-level compositor for exploratory plots. It
        controls layer order automatically: image layers are drawn first,
        transcript splats are drawn over images with transparent zero-signal
        pixels, transcript points are drawn next, and cell boundaries are drawn
        last.

        Parameters
        ----------
        image : str, dict, list, or None
            Single image layer specification. A string is interpreted as an
            image channel, e.g. ``image="DAPI"``. A dictionary with a
            ``"channel"`` key can include image options, e.g.
            ``image={"channel": "DAPI", "level": 2, "alpha": 0.5}``.
        images : list, tuple, dict, or None
            One or more image layer specifications. A dictionary whose keys are
            channel names is convenient for multiple channels, e.g.
            ``images={"DAPI": {"level": 2}, "18S": {"color": "green"}}``.
            The first image is treated as the background. Later images are
            converted to transparent signal overlays.
        splat : str, list, dict, or None
            Transcript-density layer. A gene string, gene list, or gene-set
            dictionary is treated as the ``genes`` argument to
            :meth:`plot_splat`. A dictionary containing plotting option keys can
            also be used, e.g. ``splat={"genes": ["EPCAM"], "gains": [2]}``.
        binned_splat : str, list, dict, or None
            Binned transcript-density layer from ``self.binned_adata``. Run
            ``create_binned_adata()`` first. Accepts the same gene forms as
            ``splat`` or a dictionary of :meth:`plot_binned_splat` options.
        points : str, list, dict, or None
            Transcript point layer. Accepts the same gene forms as ``splat`` or
            a dictionary of :meth:`plot_points` options, e.g.
            ``points={"genes": ["CD3D"], "s": 0.5, "max_points": 20000}``.
        cells : bool or dict or None
            If ``True``, draw default cell boundaries last. If a dictionary,
            pass those options to :meth:`plot_cells`, e.g.
            ``cells={"edge_alpha": 0.1, "linewidth": 0.3}``.
        bounds : tuple or None
            Spatial window as ``(xmin, xmax, ymin, ymax)`` in microns. Defaults
            to the active ROI when present, otherwise the full transcript frame.
        ax : matplotlib.axes.Axes or None
            Existing axes to draw into. If omitted, a new figure and axes are
            created.
        figsize : tuple
            New figure size when ``ax`` is omitted. Default ``(8, 8)``.
        dpi : int or None
            New figure DPI when ``ax`` is omitted.
        level : int or None
            Default image pyramid level applied to image layers that do not
            specify their own ``level``.
        background : str
            Figure and axes background color when creating a new axes.
        show_axis : bool
            Whether to show axis ticks and labels. Default ``False``.
        title : str or None
            Optional axes title.
        legend : bool or str
            Whether to draw one combined legend for the rendered layers.
            Default ``True``. Use ``False`` or ``"off"`` to disable it.
        legend_loc : str
            Combined legend placement. Supports ``"outside right"``,
            ``"outside left"``, ``"outside bottom"``, ``"outside top"``, or
            any Matplotlib legend location.
        legend_title : str or None
            Optional title for the combined legend.
        legend_max_items : int
            Maximum number of entries shown per categorical legend section.
        save : str, path-like, or None
            Optional path to save the rendered figure.
        save_kwargs : dict or None
            Optional keyword arguments forwarded to ``Figure.savefig``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the composite.

        Examples
        --------
        Render DAPI, a three-gene splat, one point gene, and cell boundaries:

        >>> ax = xdata.render(
        ...     image={"channel": "DAPI", "level": 2, "alpha": 0.5},
        ...     splat={"genes": ["C7", "Epcam", "Tagln"], "gains": [3, 3, 3]},
        ...     points={"genes": ["Prss3"], "s": 0.4},
        ...     cells=True,
        ...     bounds=(500, 1500, 900, 1900),
        ... )
        """
        return _pl_namespace.render(
            self,
            image=image,
            images=images,
            splat=splat,
            binned_splat=binned_splat,
            points=points,
            cells=cells,
            bounds=bounds,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            level=level,
            background=background,
            show_axis=show_axis,
            title=title,
            legend=legend,
            legend_loc=legend_loc,
            legend_title=legend_title,
            legend_max_items=legend_max_items,
            save=save,
            save_kwargs=save_kwargs,
        )

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

    def assign_cells_to_rois(self,
                              method: Literal['centroid', 'majority'] = 'centroid',
                              level: Literal['selection', 'class'] = 'selection',
                              key_added: str = 'roi',
                              min_overlap: float = 0.0):
        """
        Assign each cell to a named ROI from ``self.rois``, storing the result
        as a categorical column in ``self.adata.obs[key_added]``.

        Cells that fall outside every ROI are left as NaN.

        Parameters
        ----------
        method : 'centroid' | 'majority'
            'centroid' (default, fast) — a cell is assigned to whichever ROI
            contains its centroid. If ROIs overlap and a centroid falls in more
            than one, the last matching ROI in ``self.rois`` (insertion order)
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
        if not self.rois:
            raise ValueError("No ROIs defined. Call import_rois() first.")

        if level == 'selection':
            roi_items = list(self.rois.items())
        elif level == 'class':
            roi_items = [(name, roi_class.union) for name, roi_class in self.rois.classes.items()]
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

    def assign_bins_to_rois(self,
                             level: Literal['selection', 'class'] = 'selection',
                             key_added: str = 'roi'):
        """
        Assign each spatial bin in ``self.binned_adata`` to a named ROI from
        ``self.rois``, using the bin centroid (in microns).

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
        if not self.rois:
            raise ValueError("No ROIs defined. Call import_rois() first.")

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
            roi_items = list(self.rois.items())
        elif level == 'class':
            roi_items = [(name, roi_class.union) for name, roi_class in self.rois.classes.items()]
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
