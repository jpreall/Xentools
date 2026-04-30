#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import glob, os, sys
import scanpy as sc
import json
import geopandas as gpd
import re
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Literal
from scipy import sparse
from matplotlib.patches import Patch


def _open_zarr_group_compat(zarr_module, store, mode="r", *, force_v2=False):
    """
    Open a Zarr group across Zarr 2 and 3.

    Zarr 3 accepts ``zarr_format=2`` so we can emit/read v2 metadata explicitly.
    Zarr 2 rejects that keyword and already uses v2 metadata.
    """
    kwargs = {"store": store, "mode": mode}
    if force_v2:
        try:
            return zarr_module.open_group(**kwargs, zarr_format=2)
        except TypeError as exc:
            if "zarr_format" not in str(exc):
                raise
    return zarr_module.open_group(**kwargs)



def read_xen_panel(gene_panel_file):
    """
    Reads a Xenium gene panel file and returns the contents as a dictionary.
    """
    with open(gene_panel_file) as f:
        gene_panel = json.load(f)
    return gene_panel

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
    out = {}
    for target in gene_panel_dict['payload']['targets']:
        ID = None
        if 'id' in target['type']['data'].keys():
            ID = target['type']['data']['id']
        GENE_NAME = target['type']['data']['name']
        DESC = target['type']['descriptor']
        COVERAGE = target['info']['gene_coverage']
        PAN_ID = target['source']['identity']['design_id']
        PAN_NAME = target['source']['identity']['name']
        
    
        out[GENE_NAME] = {}
        out[GENE_NAME]['Gene_ID'] = ID
        out[GENE_NAME]['Description'] = DESC
        out[GENE_NAME]['Coverage'] = COVERAGE
        out[GENE_NAME]['Panel_ID'] = PAN_ID
        out[GENE_NAME]['Panel_Name'] = PAN_NAME

        # Add the panel version if it exists
        if 'version' in target['source']['identity'].keys():
            PAN_VERS = target['source']['identity']['version']
            out[GENE_NAME]['Panel_Version'] = PAN_VERS
    
    paneldf = pd.DataFrame.from_dict(out, orient='index')
    return paneldf

def to_TP10k(counts):
    """
    Convert a count matrix (dense or sparse) to TP10K (transcripts-per-10k)
    and return a CSR sparse matrix.
    """
    # ensure sparse
    from scipy import sparse
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)

    # library size per cell
    libsize = np.asarray(counts.sum(axis=1)).flatten()

    # avoid division by zero
    libsize[libsize == 0] = 1

    # scale to 10k
    scaled = counts.multiply(1e4 / libsize[:, None])

    # log1p transform
    scaled = scaled.log1p()

    return sparse.csr_matrix(scaled)

def TP10K(adata):
    """
    Normalize the counts in adata to transcripts per 10,000 (TP10K).
    This function assumes that the counts are stored in the 'counts' layer of adata.
    """
    # Check if 'counts' layer exists
    if 'counts' not in adata.layers:
        raise ValueError("The 'counts' layer is not found in the AnnData object.")
    from scipy import sparse
    counts = adata.layers['counts']
    adata.layers['TP10K'] = sparse.csr_matrix(10000*(counts / np.sum(counts, axis=1).A1[:,None]))


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

def create_bins(df, bin_size=5):
    """
    Create bin edges transcript-level data for rasterization.

    Parameters:
    - df: DataFrame containing 'x_location' and 'y_location' columns. Usually read from transcripts.parquet.
    - bin_size: size of the bins.
    Returns:
    - x_edges: array of bin edges for x coordinates.
    - y_edges: array of bin edges for y coordinates.
    """

    # Determine the range of x and y values
    x_min, x_max = df['x_location'].min(), df['x_location'].max()
    y_min, y_max = df['y_location'].min(), df['y_location'].max()

    # Create bin edges (ensure the max value is included)
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    return x_edges, y_edges

def bin_expression(df, bin_size=5, normalize=False):
    """
    Bin the transcript data into a 2D histogram.
    Parameters:
    - df: DataFrame containing 'x_location' and 'y_location' columns. Usually read from transcripts.parquet.
    - bin_size: size of the bins for rasterization.
    - normalize: if True, normalize the counts to [0, 1].
    Returns:
    - counts: 2D numpy array representing the binned counts.
    """
    # Create bins
    x_edges, y_edges = create_bins(df, bin_size=bin_size)

    # Bin the data into a 2D histogram:
    # Each bin counts the number of transcripts whose (x,y) fall into that bin.
    # For some reason, I need to make the histogram with the y axis first followed by the x axis
    counts, _, _ = np.histogram2d(df['y_location'], df['x_location'], bins=[y_edges, x_edges])
    if normalize:
        counts = counts / counts.max() if counts.max() > 0 else counts

    return counts

def create_binned_image(df, 
    bin_size=5, 
    colormap='Grays', 
    return_array=False,
    vmax=None,
    vmin=None):
    """
    Create a binned image from transcript data.
    Parameters:
    - df: DataFrame containing transcript data with 'x_location' and 'y_location' columns.
    - bin_size: size of the bins for rasterization.
    - colormap: colormap to use for the image.
    - return_array: if True, return the image array as well.
    - vmax: maximum value for normalization (optional).
    - vmin: minimum value for normalization (optional).
    Returns:
    - img: a PIL Image object representing the rasterized data.
    - img_array: the image array if return_array is True.
    """

    from PIL import Image
    x_edges, y_edges = create_bins(df, bin_size=bin_size)

    # Bin the data into a 2D histogram:
    # Each bin counts the number of transcripts whose (x,y) fall into that bin.
    # For some reason, I need to make the histogram with the y axis first followed by the x axis
    counts, _, _ = np.histogram2d(df['y_location'], df['x_location'], bins=[y_edges, x_edges])

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = counts.max()
        
    if (vmax is not None) or (vmin is not None):
        # Normalize counts to [vmin, vmax]
        counts = np.clip(counts, vmin, vmax)
        norm_counts = (counts - vmin) / (vmax - vmin)
    else:
        # Normalize counts to [0, 1]
        norm_counts = counts / counts.max() if counts.max() > 0 else counts
    

    # Normalize the counts to [0,1]. If counts.max() is 0, leave as is.
    #norm_counts = counts / counts.max() if counts.max() > 0 else counts

    # apply colormap
    #colormap = 'viridis'
    colormap = pl.get_cmap(colormap)
    colored_img = colormap(norm_counts)

    # Convert to unsigned 8-bit for PIL compatibility
    img_array = (colored_img[:, :, :3] * 255).astype(np.uint8)

    # Scale normalized counts to [0,255] and convert to unsigned 8-bit for PIL compatibility.
    #img_array = (norm_counts * 255).astype(np.uint8)

    img = Image.fromarray(np.uint8(img_array))
    
    if return_array:
        return img, img_array
    else:
        return img

def create_polygon(df):
    from shapely.geometry import Polygon
    """
    Create a polygon from the cell boundary data.
    The DataFrame should contain 'vertex_x' and 'vertex_y' columns.
    """
    return Polygon(zip(df.vertex_x, df.vertex_y))

def import_segmentation_xenium_parquet(boundaries_file):
    """
    Import cell boundaries from a Xenium cell_boundaries.parquet file
    Returns a GeoDataFrame with cell polygons.
    """
    import geopandas as gpd
    boundaries_df = pd.read_parquet(boundaries_file)
    boundaries_df.set_index('cell_id', inplace=True)
    boundaries_df.index = boundaries_df.index.astype('str')

    cell_boundaries = gpd.GeoDataFrame(
                boundaries_df\
                    .groupby('cell_id')\
                    .apply(create_polygon), columns=['geometry'])
    return cell_boundaries


def import_segmentation_xenium_zarr(cells_zarr_file, kind: Literal['cell', 'nucleus']='cell'):
    """
    Import cell or nucleus boundary polygons from `cells.zarr.zip`.

    Polygons are stored in physical space under `/polygon_sets`, with:
    - set `0` = nucleus
    - set `1` = cell
    """
    import geopandas as gpd
    import zarr
    from shapely.geometry import Polygon

    set_idx = '1' if kind == 'cell' else '0'
    store = zarr.storage.ZipStore(cells_zarr_file, mode='r')
    try:
        root = _open_zarr_group_compat(zarr, store, mode='r', force_v2=True)
        cell_ids_raw = root['cell_id'][:]
        cell_ids = _encode_xenium_cell_ids(cell_ids_raw[:, 0], cell_ids_raw[:, 1]).astype(str)

        grp = root[f'polygon_sets/{set_idx}']
        cell_index = grp['cell_index'][:].astype(np.int64)
        num_vertices = grp['num_vertices'][:].astype(np.int64)
        vertices = grp['vertices'][:]
    finally:
        store.close()

    geometries = {}
    for i, (idx, nverts) in enumerate(zip(cell_index, num_vertices)):
        if nverts <= 0:
            continue
        flat = vertices[i, : 2 * int(nverts)]
        coords = flat.reshape(-1, 2)
        if len(coords) < 4:
            continue
        geometries[str(cell_ids[idx])] = Polygon(coords)

    gdf = gpd.GeoDataFrame(
        {"geometry": pd.Series(geometries, dtype=object)},
        geometry="geometry",
    )
    gdf.index.name = "cell_id"
    return gdf


class LazyBoundaryGeoDataFrame:
    """
    Deferred boundary loader that materializes a GeoDataFrame on first access.
    """

    def __init__(self, loader, label: str = "boundaries"):
        self._loader = loader
        self._label = label
        self._data = None

    def _materialize(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    def __getattr__(self, name):
        return getattr(self._materialize(), name)

    def __getitem__(self, key):
        return self._materialize()[key]

    def __len__(self):
        return len(self._materialize())

    def __iter__(self):
        return iter(self._materialize())

    def __repr__(self):
        if self._data is None:
            return f"LazyBoundaryGeoDataFrame({self._label}, unloaded)"
        return repr(self._data)

"""
if os.path.exists(cell_boundaries_file):
            print('Reading in cell boundaries')
            # Read in cell boundaries
            self.cell_boundaries = pd.read_parquet(cell_boundaries_file)
            self.cell_boundaries.set_index('cell_id', inplace=True)
            self.cell_boundaries.index = self.cell_boundaries.index.astype('str')
            # Flip the y-coordinates to match the Xenium Ranger orientation
            #self.cell_boundaries['vertex_y'] = -(self.cell_boundaries['vertex_y'] - self.cell_boundaries['vertex_y'].max())
            self.cell_boundaries = gpd.GeoDataFrame(
                self.cell_boundaries\
                    .groupby('cell_id')\
                    .apply(create_polygon), columns=['geometry'])
"""    

def read_xenium_to_anndata(xenium_output_folder):
    xdir = xenium_output_folder
    #print(xdir)
    # Choose which file to load the cell feature matrix from
    # check for cell_feature_matrix.h5
    try:
        adata = sc.read_10x_h5(f'{xdir}/cell_feature_matrix.h5')
    except FileNotFoundError:
        if os.path.exists(f'{xdir}/cell_feature_matrix/'):
            adata = sc.read_10x_mtx(f'{xdir}/cell_feature_matrix/', gex_only=False)

    # Read in cell-level metadata
    cells = pd.read_parquet(f'{xdir}/cells.parquet')
    cells.index = cells['cell_id'].astype('str')
    adata.obsm['spatial'] = cells.loc[:,['x_centroid','y_centroid']].values

    # Read in UMAP coords
    umap_file = xdir + '/analysis/umap/gene_expression_2_components/projection.csv'
    try:
        umap_coords = pd.read_csv(umap_file, index_col=0)
        umap_coords.index = umap_coords.index.astype(str)

        # Align to all cells in adata; missing cells get NaNs
        coords_all = umap_coords.reindex(adata.obs_names)

        # Store as "X_umap" (shape: n_cells × 2, with NaNs where UMAP is missing)
        adata.obsm['X_umap'] = coords_all.to_numpy()

    except FileNotFoundError:
        print('No UMAP coordinates found.')
        

    # Flip spatial coords to match Xenium Ranger
    #rot_matrix = [[1,0],[0,-1]]
    #adata.obsm['spatial'] = adata.obsm['spatial'] @ rot_matrix

    # Match Xenium Ranger aspect ratio
    xrange = adata.obsm['spatial'][:,0].max() - adata.obsm['spatial'][:,0].min()
    yrange = adata.obsm['spatial'][:,1].max() - adata.obsm['spatial'][:,1].min()
    aspect_ratio = xrange/yrange
    adata.uns['aspect_ratio'] = aspect_ratio
    #pl.rcParams['figure.figsize'] = [4*aspect_ratio,4]

    # Read in gene panel
    gene_panel = read_xen_panel(xdir + '/gene_panel.json')
    try:
        species = gene_panel['payload']['panel']['species']
    except KeyError:
        species = 'Unknown'
    adata.uns['genome'] = species

    ### Start Scanpy Pipeline
    # Stash matrix layers
    adata.layers['counts'] = adata.X.astype('int').copy()
    adata.layers['TP10K'] = to_TP10k(adata.layers['counts'] ).astype('float32')
    adata.X = adata.layers['TP10K'].copy()

    # Compute QC metrics
    adata.obs['n_counts'] = adata.X.sum(1).A1.astype('int')
    adata.var['total_counts'] = adata.X.sum(0).A1.astype('int')
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1.astype('int')
    adata.obs['logUMIs'] = np.log(adata.obs['n_counts'] + 1)

    # Should I store a raw copy?
    #adata.raw = adata.copy()
    
    # Add Xenium Ranger annotations
    print('Importing Xenium Ranger cluster annotations')
    cfiles = glob.glob(xdir + '/analysis/clustering/*/*csv')
    res = pd.DataFrame()
    for f in cfiles:
        cname = f.split('/')[-2].replace('gene_expression_','')
        #print(cname)
        clusterdf = pd.read_csv(f, index_col=0)
        clusterdf.index = clusterdf.index.astype('str')

        
        #clusters = clusterdf['Cluster'].rename(cname).loc[adata.obs_names].astype('str').astype('category')
        clusters = clusterdf['Cluster'].rename(cname).astype('str').astype('category')
        res = pd.concat([res,clusters], axis=1)
    
    # Safely import cluster labels even if some cells are missing
    res = res.reindex(adata.obs_names)
    adata.obs = adata.obs.merge(res, left_index=True, right_index=True, how='left')
    

    print(adata)
    print("Ready!")
    return adata

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


def _detect_transcripts_format(folder, transcript_source: Literal['auto', 'zarr', 'parquet']='auto'):
    """
    Resolve which transcript backing store to use.

    `auto` preserves the historical preference for `transcripts.zarr.zip` when
    present, otherwise falls back to `transcripts.parquet`.
    """
    has_zarr = os.path.exists(os.path.join(folder, 'transcripts.zarr.zip'))
    has_parquet = os.path.exists(os.path.join(folder, 'transcripts.parquet'))

    if transcript_source == 'zarr':
        if not has_zarr:
            raise FileNotFoundError(f"No transcripts.zarr.zip found in {folder}")
        return 'zarr'

    if transcript_source == 'parquet':
        if not has_parquet:
            raise FileNotFoundError(f"No transcripts.parquet found in {folder}")
        return 'parquet'

    if has_zarr:
        return 'zarr'
    if has_parquet:
        return 'parquet'
    raise FileNotFoundError(
        f"Could not find transcripts.zarr.zip or transcripts.parquet in {folder}"
    )


def _count_transcripts_in_bundle(folder):
    """
    Return transcript count using file metadata without materializing the table.
    """
    zarr_path = os.path.join(folder, 'transcripts.zarr.zip')
    parquet_path = os.path.join(folder, 'transcripts.parquet')

    if os.path.exists(zarr_path):
        import zipfile
        with zipfile.ZipFile(zarr_path) as zf:
            if '.zattrs' in zf.namelist():
                attrs = json.loads(zf.read('.zattrs').decode())
                for key in ('number_rnas', 'num_transcripts'):
                    if key in attrs:
                        return int(attrs[key])

    if os.path.exists(parquet_path):
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)

    raise FileNotFoundError(
        f"Could not determine transcript count: no transcripts.zarr.zip or transcripts.parquet in {folder}"
    )


def _encode_xenium_cell_ids(prefix_arr, suffix_arr):
    """Convert raw cell_id columns to Xenium hash strings (e.g. 'aaaabbbb-12345')."""
    prefix_arr = prefix_arr.astype(np.uint32)
    suffix_arr = suffix_arr.astype(np.uint32)
    hex_strs = np.char.zfill(np.vectorize(lambda x: format(x, 'x'))(prefix_arr), 8)
    trans = str.maketrans('0123456789abcdef', 'abcdefghijklmnop')
    shifted = np.char.translate(hex_strs, trans)
    return np.char.add(np.char.add(shifted, '-'), suffix_arr.astype(str))


def _read_zarr_adata(folder, verbose=True):
    """
    Build an AnnData object from the zarr-format cell feature matrix and cell summaries.

    Sources
    -------
    cell_feature_matrix.zarr.zip
        CSC expression matrix (features × cells), gene names, cell IDs.
    cells.zarr.zip
        cell_summary array with centroid XY, area, and nucleus metrics.

    Returns
    -------
    anndata.AnnData of shape (n_cells, n_features) with:
        obs  : x_centroid, y_centroid, cell_area, nucleus_centroid_x/y, nucleus_area
        var  : gene_ids, feature_types
        obsm : 'spatial' = [[x_centroid, y_centroid], ...]
    """
    import anndata as ad, zarr
    from scipy import sparse

    cfm_path   = os.path.join(folder, 'cell_feature_matrix.zarr.zip')
    cells_path = os.path.join(folder, 'cells.zarr.zip')

    # ── Expression matrix (CSC: features × cells) ────────────────────
    if verbose:
        print('  Loading expression matrix from cell_feature_matrix.zarr.zip...',
              end=' ', flush=True)
    store_cfm = zarr.storage.ZipStore(cfm_path, mode='r')
    try:
        grp_cfm = _open_zarr_group_compat(zarr, store_cfm, mode='r', force_v2=True)
        cf = grp_cfm['cell_features']

        csc_data    = cf['csc/data'][:]            # uint32
        csc_indices = cf['csc/indices'][:].astype(np.int32)  # uint16 feature ids → int32
        csc_indptr  = cf['csc/indptr'][:]          # uint32 cell pointers

        n_features  = int(cf['indptr'].shape[0]) - 1   # from CSR indptr shape
        n_cells     = int(csc_indptr.shape[0]) - 1

        # CSC (features × cells) → transpose to CSR (cells × features)
        X = sparse.csc_matrix(
            (csc_data, csc_indices, csc_indptr),
            shape=(n_features, n_cells),
        ).T.tocsr()

        cfm_cell_ids_raw = cf['cell_id'][:]
        cfm_cell_ids = cfm_cell_ids_raw[:, 0]  # integer prefix used for alignment
    finally:
        store_cfm.close()

    if verbose:
        print(f'done. ({n_cells:,} cells × {n_features:,} features)')

    # ── Gene metadata ─────────────────────────────────────────────────
    import zipfile, json
    with zipfile.ZipFile(cfm_path) as zf:
        attrs = json.loads(zf.read('cell_features/.zattrs').decode())
    var = pd.DataFrame({
        'gene_ids':      attrs['feature_ids'],
        'feature_types': attrs['feature_types'],
    }, index=pd.Index(attrs['feature_keys'], name=''))

    # ── Cell spatial metadata ─────────────────────────────────────────
    if verbose:
        print('  Loading cell summaries from cells.zarr.zip...', end=' ', flush=True)
    store_cells = zarr.storage.ZipStore(cells_path, mode='r')
    try:
        grp_cells = _open_zarr_group_compat(zarr, store_cells, mode='r', force_v2=True)
        cell_summary = grp_cells['cell_summary'][:]   # (n_cells, 8)
        cells_ids    = grp_cells['cell_id'][:, 0]     # integer prefix used for alignment
    finally:
        store_cells.close()

    if verbose:
        print('done.')

    # Align cell_summary rows to CFM cell ordering by cell_id
    id_to_row = {int(cid): i for i, cid in enumerate(cells_ids)}
    summary_order = np.array([id_to_row.get(int(cid), -1) for cid in cfm_cell_ids])
    missing = (summary_order == -1).sum()
    if missing:
        # Fill missing rows with NaN-equivalent zeros
        summary_aligned = np.where(
            (summary_order[:, None] >= 0),
            cell_summary[np.clip(summary_order, 0, len(cell_summary) - 1)],
            np.nan,
        )
    else:
        summary_aligned = cell_summary[summary_order]

    # ── Build AnnData ─────────────────────────────────────────────────
    obs_index = pd.Index(
        _encode_xenium_cell_ids(cfm_cell_ids_raw[:, 0], cfm_cell_ids_raw[:, 1]),
        name='cell_id',
    )
    # cell_summary columns: cell_centroid_x, cell_centroid_y, cell_area,
    #                        nucleus_centroid_x, nucleus_centroid_y, nucleus_area,
    #                        z_level, nucleus_count
    obs = pd.DataFrame({
        'x_centroid':         summary_aligned[:, 0],
        'y_centroid':         summary_aligned[:, 1],
        'cell_area':          summary_aligned[:, 2],
        'nucleus_centroid_x': summary_aligned[:, 3],
        'nucleus_centroid_y': summary_aligned[:, 4],
        'nucleus_area':       summary_aligned[:, 5],
    }, index=obs_index)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm['spatial'] = summary_aligned[:, :2].astype(np.float32)

    return adata


def _read_analysis_zarr(folder, verbose=True):
    """
    Build a clusters DataFrame from analysis.zarr.zip.

    The file contains one or more cell groupings (graph clustering, k-means at
    various resolutions).  Each grouping stores its membership in a sparse
    indptr/indices format where indices are *positional* row numbers into the
    cells.zarr.zip cell_id array.

    Parameters
    ----------
    folder : str
        Xenium / Atera output folder containing analysis.zarr.zip and
        cells.zarr.zip.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Index  : cell_id as str.
        Columns: one per grouping.  The graph-clustering column is named
                 ``'Cluster'`` (matching the parquet-path convention).  k-means
                 columns are named ``'kmeans_N_clusters'``.
        Values : cluster label strings (e.g. ``'Cluster 1'``), NaN for any
                 cell that does not appear in the grouping.
    """
    import zarr

    analysis_path = os.path.join(folder, 'analysis.zarr.zip')
    cells_path    = os.path.join(folder, 'cells.zarr.zip')

    if not os.path.exists(analysis_path):
        return pd.DataFrame()

    if verbose:
        print('  Loading cluster assignments from analysis.zarr.zip...', end=' ', flush=True)

    # ── 1. Read positional cell_id lookup from cells.zarr.zip ─────────────
    store_cells = zarr.storage.ZipStore(cells_path, mode='r')
    try:
        grp_cells = _open_zarr_group_compat(zarr, store_cells, mode='r', force_v2=True)
        cell_ids_raw  = grp_cells['cell_id'][:]
        cell_ids      = cell_ids_raw[:, 0].astype(np.int64)   # integer prefix for positional lookup
    finally:
        store_cells.close()

    # ── 2. Read groupings ─────────────────────────────────────────────────
    store_an = zarr.storage.ZipStore(analysis_path, mode='r')
    try:
        grp_an = _open_zarr_group_compat(zarr, store_an, mode='r', force_v2=True)
        attrs  = dict(grp_an['cell_groups'].attrs)
        grouping_names = attrs['grouping_names']   # list of str
        group_names    = attrs['group_names']      # list of list of str

        columns = {}
        for k, (grp_name, clust_labels) in enumerate(zip(grouping_names, group_names)):
            indptr  = grp_an[f'cell_groups/{k}/indptr'][:]   # (n_clusters+1,)
            indices = grp_an[f'cell_groups/{k}/indices'][:]  # positional into cell_ids

            # Build cell_id_int → cluster_label mapping
            assignment = np.empty(len(cell_ids), dtype=object)
            assignment[:] = np.nan
            for ci, label in enumerate(clust_labels):
                pos = indices[indptr[ci]:indptr[ci + 1]]
                assignment[pos] = label

            # Column name: 'Cluster' for graphclust, 'kmeans_N_clusters' for others
            if 'graphclust' in grp_name:
                col_name = 'Cluster'
            else:
                col_name = grp_name.replace('gene_expression_', '')

            columns[col_name] = assignment
    finally:
        store_an.close()

    # ── 3. Build DataFrame indexed by encoded Xenium cell_id ─────────────
    df = pd.DataFrame(columns, index=pd.Index(
        _encode_xenium_cell_ids(cell_ids_raw[:, 0], cell_ids_raw[:, 1]), name='cell_id'
    ))
    if 'Cluster' in df.columns:
        df['Cluster'] = df['Cluster'].astype('str').astype('category')

    if verbose:
        n_clust = df['Cluster'].nunique() if 'Cluster' in df.columns else 0
        print(f'done. ({n_clust} graph clusters, {len(df):,} cells, '
              f'{len(df.columns)} groupings)')

    return df


def _load_zarr_gene_names(folder):
    """
    Return the ordered gene name list whose index matches gene_identity values in
    transcripts.zarr.zip tiles.

    Xenium 5K Prime (zarr format v5.x): names are stored in transcripts.zarr.zip
    root .zattrs under the key 'gene_names'.

    Atera WTA (zarr format v6.x): no 'gene_names' in transcripts attrs; fall back
    to cell_feature_matrix.zarr.zip → cell_features/.zattrs → 'feature_keys'.
    """
    import zipfile, json
    trans_path = os.path.join(folder, 'transcripts.zarr.zip')
    with zipfile.ZipFile(trans_path) as zf:
        if '.zattrs' in zf.namelist():
            attrs = json.loads(zf.read('.zattrs').decode())
            if 'gene_names' in attrs:
                return attrs['gene_names']          # 5K Prime path

    cfm = os.path.join(folder, 'cell_feature_matrix.zarr.zip')
    with zipfile.ZipFile(cfm) as zf:
        attrs = json.loads(zf.read('cell_features/.zattrs').decode())
    return attrs['feature_keys']                    # Atera WTA path


def frame(transcripts_df):
    xmin,xmax = transcripts_df['x_location'].min(),transcripts_df['x_location'].max()
    ymin,ymax = transcripts_df['y_location'].min(),transcripts_df['y_location'].max()
    return np.array([[xmin,xmax],[ymin,ymax]])


def _normalize_feature_selection(features, available_features, arg_name="features"):
    """Return an ordered feature list after validating membership."""
    available = list(available_features)
    if features is None:
        return available
    if isinstance(features, str):
        features = [features]
    selected = list(features)
    missing = [f for f in selected if f not in set(available)]
    if missing:
        raise ValueError(
            f"Some values in {arg_name} are not present in the dataset: "
            + ", ".join(map(str, missing))
        )
    return selected


def _bin_transcript_dataframe(df, bin_size, feature_names):
    """
    Aggregate a transcript table into a shared binning spec used by xarray and AnnData views.
    """
    feature_names = list(feature_names)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

    if df.empty:
        return {
            "feature_names": np.asarray(feature_names, dtype=object),
            "x_bin_values": np.array([], dtype=int),
            "y_bin_values": np.array([], dtype=int),
            "x_centers_um": np.array([], dtype=float),
            "y_centers_um": np.array([], dtype=float),
            "feature_indices": np.array([], dtype=int),
            "x_indices": np.array([], dtype=int),
            "y_indices": np.array([], dtype=int),
            "counts": np.array([], dtype=np.uint32),
            "shape": (len(feature_names), 0, 0),
            "occupied_x_bins": np.array([], dtype=int),
            "occupied_y_bins": np.array([], dtype=int),
            "bin_rows": np.array([], dtype=int),
        }

    work = df.loc[:, ["x_location", "y_location", "feature_name"]].copy()
    work["x_bin"] = np.floor(work["x_location"] / bin_size).astype(int)
    work["y_bin"] = np.floor(work["y_location"] / bin_size).astype(int)

    grouped = (
        work.groupby(["feature_name", "y_bin", "x_bin"])
        .size()
        .reset_index(name="count")
    )

    feature_index = {gene: i for i, gene in enumerate(feature_names)}
    grouped = grouped[grouped["feature_name"].isin(feature_index)].copy()

    if grouped.empty:
        return {
            "feature_names": np.asarray(feature_names, dtype=object),
            "x_bin_values": np.array([], dtype=int),
            "y_bin_values": np.array([], dtype=int),
            "x_centers_um": np.array([], dtype=float),
            "y_centers_um": np.array([], dtype=float),
            "feature_indices": np.array([], dtype=int),
            "x_indices": np.array([], dtype=int),
            "y_indices": np.array([], dtype=int),
            "counts": np.array([], dtype=np.uint32),
            "shape": (len(feature_names), 0, 0),
            "occupied_x_bins": np.array([], dtype=int),
            "occupied_y_bins": np.array([], dtype=int),
            "bin_rows": np.array([], dtype=int),
        }

    x_min_bin = int(grouped["x_bin"].min())
    x_max_bin = int(grouped["x_bin"].max())
    y_min_bin = int(grouped["y_bin"].min())
    y_max_bin = int(grouped["y_bin"].max())

    x_bin_values = np.arange(x_min_bin, x_max_bin + 1, dtype=int)
    y_bin_values = np.arange(y_min_bin, y_max_bin + 1, dtype=int)
    x_centers_um = (x_bin_values + 0.5) * float(bin_size)
    y_centers_um = (y_bin_values + 0.5) * float(bin_size)

    grouped["feature_index"] = grouped["feature_name"].map(feature_index).astype(int)
    grouped["x_index"] = grouped["x_bin"] - x_min_bin
    grouped["y_index"] = grouped["y_bin"] - y_min_bin

    occupied_bins = (
        grouped.loc[:, ["x_bin", "y_bin"]]
        .drop_duplicates()
        .sort_values(["y_bin", "x_bin"])
        .reset_index(drop=True)
    )
    occupied_keys = list(zip(occupied_bins["x_bin"], occupied_bins["y_bin"]))
    occupied_lookup = {key: i for i, key in enumerate(occupied_keys)}
    grouped["bin_row"] = [
        occupied_lookup[(x_bin, y_bin)]
        for x_bin, y_bin in zip(grouped["x_bin"], grouped["y_bin"])
    ]

    return {
        "feature_names": np.asarray(feature_names, dtype=object),
        "x_bin_values": x_bin_values,
        "y_bin_values": y_bin_values,
        "x_centers_um": x_centers_um,
        "y_centers_um": y_centers_um,
        "feature_indices": grouped["feature_index"].to_numpy(dtype=int),
        "x_indices": grouped["x_index"].to_numpy(dtype=int),
        "y_indices": grouped["y_index"].to_numpy(dtype=int),
        "counts": grouped["count"].to_numpy(dtype=np.uint32),
        "shape": (len(feature_names), len(y_bin_values), len(x_bin_values)),
        "occupied_x_bins": occupied_bins["x_bin"].to_numpy(dtype=int),
        "occupied_y_bins": occupied_bins["y_bin"].to_numpy(dtype=int),
        "bin_rows": grouped["bin_row"].to_numpy(dtype=int),
    }

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


@dataclass
class ROI:
    """
    Lightweight ROI wrapper with a canonical plotting-bounds convention.

    Public ``bounds`` always use ``(xmin, xmax, ymin, ymax)`` so plotting,
    rasterization, and image cropping code do not need to remember shapely's
    native ``(minx, miny, maxx, maxy)`` ordering.
    """
    geometry: Any
    name: str | None = None
    selection_name: str | None = None
    class_name: str | None = None
    poly_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(closed=True, fill=False, edgecolor='yellow', linewidth=1)
    )
    source: str | None = None
    units: str = "micron"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shapely_bounds(self):
        return tuple(map(float, self.geometry.bounds))

    @property
    def bounds(self):
        minx, miny, maxx, maxy = self.shapely_bounds
        return (minx, maxx, miny, maxy)

    @property
    def centroid(self):
        c = self.geometry.centroid
        return (float(c.x), float(c.y))

    @property
    def area(self):
        return float(self.geometry.area)

    @property
    def area_um2(self):
        return self.area

    @property
    def points(self):
        return _geometry_to_roi_points(self.geometry)

    def contains_points(self, x, y):
        from shapely import contains_xy
        return contains_xy(self.geometry, x, y)

    def crop_dataframe(self, df, x_col="x_location", y_col="y_location"):
        mask = self.contains_points(df[x_col].to_numpy(), df[y_col].to_numpy())
        return df.loc[mask].copy()

    def query_lazy_transcripts(self, lazy_transcripts, genes=None, quality="all"):
        xmin, xmax, ymin, ymax = self.bounds
        return lazy_transcripts.query(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            genes=genes,
            quality=quality,
        )

    def plot(self, ax, **kwargs):
        """
        Trace this ROI's outline over an existing matplotlib axis.
        """
        poly_kwargs = dict(self.poly_kwargs)
        poly_kwargs.update(kwargs)
        polygon = pl.Polygon(self.points, **poly_kwargs)
        ax.add_patch(polygon)
        return ax

    def to_pixels(self, pixel_size, name=None):
        from shapely.affinity import scale
        factor = 1.0 / float(pixel_size)
        return ROI.from_geometry(
            scale(self.geometry, xfact=factor, yfact=factor, origin=(0, 0)),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=dict(self.poly_kwargs),
            source=self.source,
            units="pixel",
            metadata=dict(self.metadata),
        )

    def to_microns(self, pixel_size, name=None):
        from shapely.affinity import scale
        factor = float(pixel_size)
        return ROI.from_geometry(
            scale(self.geometry, xfact=factor, yfact=factor, origin=(0, 0)),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=dict(self.poly_kwargs),
            source=self.source,
            units="micron",
            metadata=dict(self.metadata),
        )

    def to_geojson_feature(self, include_name=True):
        from shapely.geometry import mapping
        props = dict(self.metadata)
        if include_name and self.name is not None:
            props.setdefault("name", self.name)
        if self.selection_name is not None:
            props.setdefault("selection_name", self.selection_name)
        if self.class_name is not None:
            props.setdefault("class_name", self.class_name)
        if self.source is not None:
            props.setdefault("source", self.source)
        if self.units:
            props.setdefault("units", self.units)
        return {
            "type": "Feature",
            "properties": props,
            "geometry": mapping(self.geometry),
        }

    def bbox(self, name=None, poly_kwargs=None):
        from shapely.geometry import box as shapely_box
        xmin, xmax, ymin, ymax = self.bounds
        return ROI(
            geometry=shapely_box(xmin, ymin, xmax, ymax),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=self.poly_kwargs if poly_kwargs is None else poly_kwargs,
            source=self.source,
            units=self.units,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_geometry(cls, geometry, name=None, selection_name=None, class_name=None, poly_kwargs=None, source=None, units="micron", metadata=None):
        return cls(
            geometry=geometry,
            name=name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=dict(closed=True, fill=False, edgecolor='yellow', linewidth=1)
            if poly_kwargs is None else poly_kwargs,
            source=source,
            units=units,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_bounds(cls, xmin, xmax, ymin, ymax, name=None, selection_name=None, class_name=None, poly_kwargs=None, source=None, units="micron", metadata=None):
        from shapely.geometry import box as shapely_box
        return cls.from_geometry(
            shapely_box(float(xmin), float(ymin), float(xmax), float(ymax)),
            name=name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=poly_kwargs,
            source=source,
            units=units,
            metadata=metadata,
        )

    @classmethod
    def from_geojson(cls, geojson_path, feature=None, scale_factor=1.0, name=None, selection_name=None, class_name=None, poly_kwargs=None, source=None, units="micron", metadata=None):
        geometry, area_um2 = read_ROI_from_geojson(
            geojson_path,
            feature=feature,
            scale_factor=scale_factor,
        )
        roi_name = name
        if roi_name is None and isinstance(feature, str):
            roi_name = feature
        roi = cls.from_geometry(
            geometry,
            name=roi_name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=poly_kwargs,
            source=os.fspath(geojson_path) if source is None else source,
            units=units,
            metadata=metadata,
        )
        roi.metadata.setdefault("area_um2", area_um2)
        return roi

    def __getitem__(self, key):
        legacy = {
            "geometry": self.geometry,
            "centroid": self.centroid,
            "points": self.points,
            "poly_kwargs": self.poly_kwargs,
            "source": self.source,
            "area_um2": self.area_um2,
            "selection_name": self.selection_name,
            "class_name": self.class_name,
        }
        if key in legacy:
            return legacy[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)


@dataclass
class ROIClass:
    name: str
    rois: list[ROI] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, roi: ROI):
        self.rois.append(roi)

    @property
    def bounds(self):
        if not self.rois:
            return (0.0, 0.0, 0.0, 0.0)
        xmin = min(r.bounds[0] for r in self.rois)
        xmax = max(r.bounds[1] for r in self.rois)
        ymin = min(r.bounds[2] for r in self.rois)
        ymax = max(r.bounds[3] for r in self.rois)
        return (xmin, xmax, ymin, ymax)

    @property
    def union(self):
        from shapely.ops import unary_union

        geom = unary_union([roi.geometry for roi in self.rois]) if self.rois else None
        if geom is None:
            raise ValueError(f"ROI class '{self.name}' has no geometries.")
        return ROI.from_geometry(
            geom,
            name=self.name,
            class_name=self.name,
            poly_kwargs=self.rois[0].poly_kwargs if self.rois else None,
            source=self.rois[0].source if self.rois else None,
            metadata=dict(self.metadata),
        )

    def plot(self, ax, **kwargs):
        for roi in self.rois:
            roi.plot(ax, **kwargs)
        return ax

    def __len__(self):
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rois[key]
        for roi in self.rois:
            if roi.name == key or roi.selection_name == key:
                return roi
        raise KeyError(key)


@dataclass
class ROICollection:
    classes: dict[str, ROIClass] = field(default_factory=dict)
    flat_named: dict[str, ROI] = field(default_factory=dict)

    def add(self, roi: ROI):
        key = str(roi.name) if roi.name is not None else f"roi_{len(self.flat_named)}"
        roi.name = key
        self.flat_named[key] = roi
        class_name = roi.class_name
        if class_name is not None:
            cls = self.classes.get(class_name)
            if cls is None:
                cls = ROIClass(name=class_name)
                self.classes[class_name] = cls
            cls.add(roi)

    def class_names(self):
        return list(self.classes.keys())

    def selection_names(self):
        return list(self.flat_named.keys())

    def all_rois(self):
        return list(self.flat_named.values())

    @property
    def union(self):
        from shapely.ops import unary_union

        rois = self.all_rois()
        if not rois:
            raise ValueError("ROI collection is empty.")
        geom = unary_union([roi.geometry for roi in rois])
        return ROI.from_geometry(
            geom,
            name="all_rois",
            poly_kwargs=rois[0].poly_kwargs if rois else None,
            source=rois[0].source if rois else None,
            metadata=self.summary(),
        )

    def get_union(self, class_name):
        return self.classes[class_name].union

    def plot(self, ax, level: Literal['selection', 'class'] = 'selection', **kwargs):
        if level == 'selection':
            for roi in self.flat_named.values():
                roi.plot(ax, **kwargs)
        elif level == 'class':
            for roi_class in self.classes.values():
                roi_class.plot(ax, **kwargs)
        else:
            raise ValueError("level must be 'selection' or 'class'.")
        return ax

    def items(self):
        return self.flat_named.items()

    def keys(self):
        return self.flat_named.keys()

    def values(self):
        return self.flat_named.values()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        self.classes.clear()
        self.flat_named.clear()

    def summary(self):
        return {
            "n_selections": len(self.flat_named),
            "n_classes": len(self.classes),
            "classes": {name: len(cls.rois) for name, cls in self.classes.items()},
        }

    def import_file(self, roi_file, roi_name=None, pixel_size=1.0, scale_geojson=True, append=True):
        if not append:
            self.clear()
        imported_names = _import_roi_records(
            roi_store=self,
            roi_file=roi_file,
            roi_name=roi_name,
            pixel_size=pixel_size,
            scale_geojson=scale_geojson,
        )
        return imported_names

    def to_geojson_feature_collection(self, level: Literal['selection', 'class'] = 'selection'):
        if level == 'selection':
            features = [roi.to_geojson_feature() for roi in self.flat_named.values()]
        elif level == 'class':
            features = [roi_class.union.to_geojson_feature() for roi_class in self.classes.values()]
        else:
            raise ValueError("level must be 'selection' or 'class'.")
        return {"type": "FeatureCollection", "features": features}

    def resolve(self, selector):
        if isinstance(selector, ROI):
            return selector
        if isinstance(selector, ROIClass):
            return selector.union
        if isinstance(selector, int):
            names = list(self.flat_named.keys())
            if selector < 0 or selector >= len(names):
                raise KeyError(selector)
            return self.flat_named[names[selector]]
        if isinstance(selector, str):
            if selector in self.flat_named:
                return self.flat_named[selector]
            if selector in self.classes:
                return self.classes[selector].union
        raise KeyError(selector)

    def __getitem__(self, key):
        if key in self.flat_named:
            return self.flat_named[key]
        if key in self.classes:
            return self.classes[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.flat_named or key in self.classes

    def __len__(self):
        return len(self.flat_named)

    def __iter__(self):
        return iter(self.flat_named)

    def __bool__(self):
        return bool(self.flat_named or self.classes)


class LazyTranscripts:
    """
    Memory-efficient lazy accessor for Xenium/Atera transcript data in transcripts.zarr.zip.

    Transcripts are stored in a spatial tile grid (grids/0/{col},{row}/).  Within each
    tile they are sorted by gene, so per-gene queries use the gene_offset index array for
    O(1) slice access rather than a full scan.

    Parameters
    ----------
    zarr_path : str
        Path to transcripts.zarr.zip.
    gene_names : list[str]
        Ordered list of gene names; index == gene_call value stored in gene_identity arrays.
    cache_threshold : int
        Queries returning fewer transcripts than this are cached in memory.  Default 5 000 000.
    verbose : bool
        Print progress during spatial index construction.  Default True.
    """

    def __init__(self, zarr_path, gene_names, cache_threshold=5_000_000, verbose=True):
        self._path = str(zarr_path)
        self._gene_names = list(gene_names)
        self._gene_index = {g: i for i, g in enumerate(gene_names)}
        self.cache_threshold = cache_threshold
        self._query_cache = {}          # cache_key -> DataFrame
        self._tile_meta = {}            # 'col,row' -> {'n': int, 'x': float, 'y': float}
        self._tile_size = (500.0, 500.0)  # estimated (dx, dy) in microns; refined below
        self._load_tile_metadata(verbose)

    # ------------------------------------------------------------------
    # Spatial index construction
    # ------------------------------------------------------------------

    def _load_tile_metadata(self, verbose=True):
        """
        Phase 1: read .zarray JSON for every tile (no pixel data, instant).
        Phase 2: sample element [0] from each tile's location array to get one
                 representative coordinate per tile.  One zarr chunk (~1.2 MB) is
                 decompressed per tile; this is the minimum granularity for zarr.zip.
        """
        import zipfile, json, zarr

        # Phase 1 — shape metadata only
        tile_keys = []
        with zipfile.ZipFile(self._path) as zf:
            names = set(zf.namelist())
            for name in names:
                if name.startswith('grids/0/') and name.endswith('location/.zarray'):
                    parts = name.split('/')
                    key = parts[2]
                    if ',' not in key:
                        continue
                    meta = json.loads(zf.read(name).decode())
                    n = meta['shape'][0]
                    if n > 0:
                        self._tile_meta[key] = {'n': n, 'x': None, 'y': None}
                        tile_keys.append(key)

        if not tile_keys:
            return

        if verbose:
            print(f'  Building spatial index for {len(tile_keys)} tiles...', end=' ', flush=True)

        # Phase 2 — one representative coordinate per tile
        store = zarr.storage.ZipStore(self._path, mode='r')
        try:
            grp = _open_zarr_group_compat(zarr, store, mode='r', force_v2=True)
            for key in tile_keys:
                try:
                    first = grp[f'grids/0/{key}/location'][0]  # loads first chunk; keep row 0
                    self._tile_meta[key]['x'] = float(first[0])
                    self._tile_meta[key]['y'] = float(first[1])
                except Exception:
                    del self._tile_meta[key]
        finally:
            store.close()

        if verbose:
            print('done.')

        self._tile_size = self._estimate_tile_size()

    def _estimate_tile_size(self):
        """Estimate per-tile spatial extent from sample coordinates and grid layout."""
        if not self._tile_meta:
            return (500.0, 500.0)

        xs_by_col = {}
        ys_by_row = {}
        for key, meta in self._tile_meta.items():
            if meta['x'] is None:
                continue
            col, row = map(int, key.split(','))
            xs_by_col.setdefault(col, []).append(meta['x'])
            ys_by_row.setdefault(row, []).append(meta['y'])

        def _median_step(d):
            vals = sorted(np.mean(v) for v in d.values())
            if len(vals) < 2:
                return 500.0
            return float(np.median(np.diff(vals)))

        dx = _median_step(xs_by_col)
        dy = _median_step(ys_by_row)
        return (max(dx, 50.0), max(dy, 50.0))

    # ------------------------------------------------------------------
    # Tile selection
    # ------------------------------------------------------------------

    def _candidate_tiles(self, xmin, xmax, ymin, ymax):
        """
        Return tile keys whose sample coordinate falls within the query bbox
        expanded by one tile_size in every direction.  This guarantees that any
        tile whose spatial extent overlaps the bbox is included, at the cost of a
        few false-positive tiles that are filtered out after loading.
        """
        dx, dy = self._tile_size
        x0, x1 = xmin - dx, xmax + dx
        y0, y1 = ymin - dy, ymax + dy
        return [
            key for key, m in self._tile_meta.items()
            if m['x'] is not None
            and x0 <= m['x'] <= x1
            and y0 <= m['y'] <= y1
        ]

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def query(self,
              xmin=None, xmax=None, ymin=None, ymax=None,
              genes=None,
              quality: str = 'high') -> pd.DataFrame:
        """
        Return a DataFrame of transcripts within a bounding box.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float | None
            Bounding box in microns.  None = no limit on that edge.
        genes : str | list[str] | None
            Restrict to these gene names.  None = all genes.
        quality : 'high' | 'low' | 'all'
            Transcript quality tier from gene_offset.  Default 'high'.

        Returns
        -------
        pd.DataFrame with columns x_location, y_location, feature_name.
        """
        import zarr

        xmin = -np.inf if xmin is None else float(xmin)
        xmax =  np.inf if xmax is None else float(xmax)
        ymin = -np.inf if ymin is None else float(ymin)
        ymax =  np.inf if ymax is None else float(ymax)

        # Resolve gene ids
        if genes is None:
            gene_ids = None
        else:
            if isinstance(genes, str):
                genes = [genes]
            gene_ids = [self._gene_index[g] for g in genes if g in self._gene_index]
            if not gene_ids:
                return pd.DataFrame(columns=['x_location', 'y_location', 'feature_name'])

        # Cache lookup
        cache_key = (
            round(xmin, 1), round(xmax, 1), round(ymin, 1), round(ymax, 1),
            tuple(sorted(gene_ids)) if gene_ids is not None else None,
            quality,
        )
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        tile_keys = self._candidate_tiles(xmin, xmax, ymin, ymax)

        store = zarr.storage.ZipStore(self._path, mode='r')
        frames = []
        try:
            grp = _open_zarr_group_compat(zarr, store, mode='r', force_v2=True)
            for key in tile_keys:
                df = self._load_tile(grp, key, xmin, xmax, ymin, ymax, gene_ids, quality)
                if df is not None and len(df):
                    frames.append(df)
        finally:
            store.close()

        result = (pd.concat(frames, ignore_index=True)
                  if frames
                  else pd.DataFrame(columns=['x_location', 'y_location', 'feature_name']))

        if len(result) < self.cache_threshold:
            self._query_cache[cache_key] = result

        return result

    def _load_tile(self, grp, tile_key, xmin, xmax, ymin, ymax, gene_ids, quality):
        """Load and spatially filter one tile, using gene_offset for gene-specific slicing."""
        prefix = f'grids/0/{tile_key}'
        try:
            loc_arr = grp[f'{prefix}/location']      # (N, 3) float32
            gid_arr = grp[f'{prefix}/gene_identity'] # (N, 1) uint16
            off_arr = grp[f'{prefix}/gene_offset']   # (n_genes, 4) uint32
        except Exception:
            return None

        # Quality column indices: [lo_start, lo_end, hi_start, hi_end]
        if quality == 'high':
            q_slices = [(2, 3)]
        elif quality == 'low':
            q_slices = [(0, 1)]
        else:  # 'all'
            q_slices = [(0, 1), (2, 3)]

        if gene_ids is not None:
            # Efficient path: collect contiguous ranges from gene_offset then slice
            offsets = off_arr[:]  # (n_genes, 4) — small, always load fully

            raw_ranges = []
            for gid in gene_ids:
                if gid >= len(offsets):
                    continue
                for sc, ec in q_slices:
                    s, e = int(offsets[gid, sc]), int(offsets[gid, ec])
                    if e > s:
                        raw_ranges.append((s, e))

            if not raw_ranges:
                return None

            # Merge overlapping/adjacent ranges to minimize zarr reads
            raw_ranges.sort()
            merged = [list(raw_ranges[0])]
            for s, e in raw_ranges[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])

            xs, ys, gs = [], [], []
            for s, e in merged:
                locs = loc_arr[s:e]        # (m, 3)
                gids = gid_arr[s:e, 0]    # (m,)
                mask = ((locs[:, 0] >= xmin) & (locs[:, 0] <= xmax) &
                        (locs[:, 1] >= ymin) & (locs[:, 1] <= ymax))
                if not mask.any():
                    continue
                xs.append(locs[mask, 0])
                ys.append(locs[mask, 1])
                gs.append(gids[mask])

            if not xs:
                return None
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            g_idx = np.concatenate(gs)

        else:
            # No gene filter: scan full tile
            locs = loc_arr[:]
            mask = ((locs[:, 0] >= xmin) & (locs[:, 0] <= xmax) &
                    (locs[:, 1] >= ymin) & (locs[:, 1] <= ymax))
            if not mask.any():
                return None
            x = locs[mask, 0]
            y = locs[mask, 1]
            g_idx = gid_arr[:, 0][mask]

        n = len(self._gene_names)
        names = np.array(
            [self._gene_names[i] if i < n else 'Unknown' for i in g_idx],
            dtype=object,
        )
        return pd.DataFrame({'x_location': x, 'y_location': y, 'feature_name': names})

    # ------------------------------------------------------------------
    # Convenience / compatibility
    # ------------------------------------------------------------------

    @property
    def n_transcripts(self):
        """Total transcript count across all tiles (from metadata, no data IO)."""
        return sum(m['n'] for m in self._tile_meta.values())

    def __len__(self):
        return self.n_transcripts

    @property
    def frame(self):
        """
        Approximate [[xmin, xmax], [ymin, ymax]] derived from tile sample coordinates.
        Exact to within one tile_size at each edge.
        """
        xs = [m['x'] for m in self._tile_meta.values() if m['x'] is not None]
        ys = [m['y'] for m in self._tile_meta.values() if m['y'] is not None]
        if xs and ys:
            dx, dy = self._tile_size
            return np.array([[min(xs) - dx/2, max(xs) + dx/2],
                             [min(ys) - dy/2, max(ys) + dy/2]])
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def to_dataframe(self, quality='high'):
        """Materialise all transcripts into memory.  Use sparingly on large datasets."""
        return self.query(quality=quality)

    def to_xarray_bins(self,
                       bin_size=5,
                       genes=None,
                       bounds=None,
                       quality='all',
                       name='counts'):
        """
        Bin transcripts into a labeled xarray.DataArray with dims (feature_name, y, x).

        This is the spatial-array view of transcript counts. It preserves the
        physical micron coordinates of bin centers and is intended for image-like
        operations such as smoothing, reductions, and per-gene raster access.
        """
        import xarray as xr

        feature_names = _normalize_feature_selection(genes, self._gene_names, arg_name="genes")

        if bounds is None:
            df = self.query(genes=feature_names, quality=quality)
        else:
            xmin, xmax, ymin, ymax = map(float, bounds)
            df = self.query(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                genes=feature_names,
                quality=quality,
            )

        binned = _bin_transcript_dataframe(df, bin_size=bin_size, feature_names=feature_names)
        cube = np.zeros(binned["shape"], dtype=np.uint32)
        if len(binned["counts"]):
            cube[
                binned["feature_indices"],
                binned["y_indices"],
                binned["x_indices"],
            ] = binned["counts"]

        attrs = {
            "bin_size": float(bin_size),
            "quality": quality,
            "units": "micron",
            "n_transcripts": int(binned["counts"].sum()),
        }
        if bounds is not None:
            attrs["bounds"] = tuple(map(float, bounds))

        return xr.DataArray(
            cube,
            dims=("feature_name", "y", "x"),
            coords={
                "feature_name": binned["feature_names"],
                "y": binned["y_centers_um"],
                "x": binned["x_centers_um"],
            },
            name=name,
            attrs=attrs,
        )

    def clear_cache(self):
        """Discard all cached query results."""
        self._query_cache.clear()

    def __repr__(self):
        n = self.n_transcripts
        t = len(self._tile_meta)
        g = len(self._gene_names)
        cached = len(self._query_cache)
        return (f"LazyTranscripts({n:,} transcripts | {t} tiles | "
                f"{g:,} genes | cache_threshold={self.cache_threshold:,} | "
                f"{cached} cached queries)")


class XenData:
    def __init__(self, xenium_folder, verbose=True, roi_file=None, crop_to_selection=None,
                 cache_threshold=5_000_000,
                 transcript_source: Literal['auto', 'zarr', 'parquet']='auto',
                 eager_transcript_threshold: int=20_000_000,
                 boundary_source: Literal['auto', 'parquet', 'zarr']='auto',
                 lazy_boundaries: bool=False):
        self.xenium_folder = xenium_folder
        self.cache_threshold = cache_threshold
        self.eager_transcript_threshold = int(eager_transcript_threshold)
        bundle_transcripts_fmt = _detect_transcripts_format(
            xenium_folder,
            transcript_source='auto',
        )
        resolved_transcript_source = transcript_source
        self.n_transcripts = _count_transcripts_in_bundle(xenium_folder)

        if (
            transcript_source == 'auto'
            and bundle_transcripts_fmt == 'zarr'
            and os.path.exists(os.path.join(xenium_folder, 'transcripts.parquet'))
            and self.n_transcripts < self.eager_transcript_threshold
        ):
            resolved_transcript_source = 'parquet'
            if verbose:
                print(
                    f"Dataset has {self.n_transcripts:,} transcripts "
                    f"(< {self.eager_transcript_threshold:,}); using transcripts.parquet."
                )

        self._transcripts_fmt = _detect_transcripts_format(
            xenium_folder,
            transcript_source=resolved_transcript_source,
        )

        if bundle_transcripts_fmt == 'zarr':
            # ── Zarr-backed bundle; transcript source can still be overridden ──
            gene_names = _load_zarr_gene_names(xenium_folder)
            if self._transcripts_fmt == 'zarr':
                zarr_path = os.path.join(xenium_folder, 'transcripts.zarr.zip')
                if verbose:
                    print(f'Zarr format detected.  Indexing {len(gene_names):,} genes across '
                          f'transcripts.zarr.zip...')
                self.trans = LazyTranscripts(zarr_path, gene_names,
                                             cache_threshold=cache_threshold,
                                             verbose=verbose)
                if verbose:
                    print(f'  {self.trans.n_transcripts:,} transcripts in '
                          f'{len(self.trans._tile_meta)} spatial tiles.')
            else:
                if verbose:
                    print('Using transcripts.parquet for rich transcript metadata...')
                self.trans = pd.read_parquet(os.path.join(xenium_folder, 'transcripts.parquet'))
                sample = self.trans["feature_name"].iloc[:100]
                if sample.map(lambda x: isinstance(x, (bytes, bytearray))).any():
                    self.trans["feature_name"] = self.trans["feature_name"].str.decode("utf-8")

            panel_file = os.path.join(xenium_folder, 'gene_panel.json')
            self.gene_panel = read_xen_panel(panel_file) if os.path.exists(panel_file) else None
            self.adata      = _read_zarr_adata(xenium_folder, verbose=verbose)
            self.clusters   = _read_analysis_zarr(xenium_folder, verbose=verbose)
            if len(self.clusters):
                self.adata.obs = self.adata.obs.merge(
                    self.clusters, left_index=True, right_index=True, how='left')
        elif bundle_transcripts_fmt == 'parquet':
            # ── Classic parquet-format dataset ────────────────────────────
            self.trans, self.clusters, self.gene_panel = read_xen_essentials(xenium_folder, verbose)

            if verbose:
                print('Reading in AnnData object')
            self.adata = read_xenium_to_anndata(xenium_folder)

            self.adata.obs = self.adata.obs.merge(
                self.clusters, left_index=True, right_index=True, how='left')
            self.adata.var.merge(
                _make_gene_panel_df(self.gene_panel),
                left_index=True, right_index=True, how='left')

        # ── Common to both formats ─────────────────────────────────────────
        cell_boundaries_file = os.path.join(xenium_folder, 'cell_boundaries.parquet')
        nuc_boundaries_file  = os.path.join(xenium_folder, 'nucleus_boundaries.parquet')
        cells_zarr_file = os.path.join(xenium_folder, 'cells.zarr.zip')

        has_boundary_parquet = os.path.exists(cell_boundaries_file) and os.path.exists(nuc_boundaries_file)
        has_cells_zarr = os.path.exists(cells_zarr_file)

        if boundary_source == 'parquet':
            if not has_boundary_parquet:
                raise FileNotFoundError(
                    f"Requested boundary_source='parquet' but boundary parquet files were not found in {xenium_folder}"
                )
            resolved_boundary_source = 'parquet'
        elif boundary_source == 'zarr':
            if not has_cells_zarr:
                raise FileNotFoundError(
                    f"Requested boundary_source='zarr' but cells.zarr.zip was not found in {xenium_folder}"
                )
            resolved_boundary_source = 'zarr'
        else:
            if has_boundary_parquet:
                resolved_boundary_source = 'parquet'
            elif has_cells_zarr:
                resolved_boundary_source = 'zarr'
            else:
                resolved_boundary_source = None

        if resolved_boundary_source == 'parquet':
            if verbose:
                print('Reading in cell boundaries')
            self.cell_boundaries = pd.read_parquet(cell_boundaries_file)
            self.cell_boundaries.set_index('cell_id', inplace=True)
            self.cell_boundaries.index = self.cell_boundaries.index.astype('str')
            self.cell_boundaries = gpd.GeoDataFrame(
                self.cell_boundaries
                    .groupby('cell_id')
                    .apply(create_polygon), columns=['geometry'])

            if verbose:
                print('Reading in nucleus boundaries')
            self.nucleus_boundaries = pd.read_parquet(nuc_boundaries_file)
            self.nucleus_boundaries.set_index('cell_id', inplace=True)
            self.nucleus_boundaries.index = self.nucleus_boundaries.index.astype('str')
            self.nucleus_boundaries = gpd.GeoDataFrame(
                self.nucleus_boundaries
                    .groupby('cell_id')
                    .apply(create_polygon), columns=['geometry'])
        elif resolved_boundary_source == 'zarr':
            if lazy_boundaries:
                if verbose:
                    print('Registering lazy Zarr-backed cell boundaries')
                self.cell_boundaries = LazyBoundaryGeoDataFrame(
                    lambda: import_segmentation_xenium_zarr(cells_zarr_file, kind='cell'),
                    label='cell boundaries from cells.zarr.zip',
                )
                self.nucleus_boundaries = LazyBoundaryGeoDataFrame(
                    lambda: import_segmentation_xenium_zarr(cells_zarr_file, kind='nucleus'),
                    label='nucleus boundaries from cells.zarr.zip',
                )
            else:
                if verbose:
                    print('Reading in cell boundaries from cells.zarr.zip')
                self.cell_boundaries = import_segmentation_xenium_zarr(cells_zarr_file, kind='cell')
                if verbose:
                    print('Reading in nucleus boundaries from cells.zarr.zip')
                self.nucleus_boundaries = import_segmentation_xenium_zarr(cells_zarr_file, kind='nucleus')
        else:
            self.cell_boundaries = None
            self.nucleus_boundaries = None
        self._boundary_source = resolved_boundary_source
        self._lazy_boundaries = bool(lazy_boundaries and resolved_boundary_source == 'zarr')


        xenium_file = os.path.join(xenium_folder, 'experiment.xenium')
        with open(xenium_file) as f:
            self.xenium_metadata = json.load(f)
        self.pixel_size = self.xenium_metadata['pixel_size']

        keys = ['run_name', 'slide_id', 'region_name']
        self.name = '_'.join([self.xenium_metadata[k] for k in keys])

        self.ROIs = ROICollection()
        self.rois = self.ROIs
        self.active_roi = None
        self.subset_roi = None
        self.update_cell_names()

        self.images = {}
        if os.path.exists(os.path.join(xenium_folder, 'morphology.ome.tif')):
            self.images['DAPI'] = os.path.join(xenium_folder, 'morphology.ome.tif')
        self.protein_images = _detect_linked_protein_images(xenium_folder)
        if self.protein_images is not None:
            self.images['Protein'] = self.protein_images['folder']

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
            fig, ax = pl.subplots(figsize=[fig_scale, fig_scale * self.aspect_ratio])
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
            pl.show()

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

        Parameters
        ----------
        output_dir : str or Path
            Destination directory (created if absent).
        include_morphology : bool
            Include ``morphology.ome.tif``.  Requires ``tifffile``.
        crop_morphology : bool
            Crop to the active ROI bounding box when an ROI is active.
        rebase_coordinates : bool
            Shift spatial coordinates so the ROI crop origin becomes (0, 0).
        pyramidal_morphology : bool
            Write a multi-resolution pyramidal OME-TIFF.
        pyramid_scale : int
            Downsampling factor between pyramid levels.
        morphology_tile : tuple[int, int]
            Tile size (width, height) for the pyramidal OME-TIFF.
        overwrite : bool
            Overwrite an existing output directory.
        verbose : bool
        """
        import importlib.util
        from pathlib import Path as _Path

        # Load helper module by absolute file path — avoids package-resolution
        # issues when xentools is run from within its own directory.
        _helper_path = _Path(__file__).parent / "io_utils" / "_write_explorer.py"
        _spec = importlib.util.spec_from_file_location("_write_explorer", _helper_path)
        _we = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_we)

        outdir = _Path(output_dir)
        if outdir.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {outdir}\n"
                "Pass overwrite=True to replace it."
            )
        outdir.mkdir(parents=True, exist_ok=True)

        crop_origin = _export_crop_origin_um(self) if rebase_coordinates else (0.0, 0.0)

        if verbose:
            print("  Collecting transcripts...", end=" ", flush=True)
        trans_df = _we._materialize_transcripts(self)
        trans_df = _rebase_spatial_dataframe(trans_df, crop_origin)
        n_transcripts = len(trans_df)
        trans_df.to_parquet(outdir / "transcripts.parquet", index=False)
        if verbose:
            print(f"done. ({n_transcripts:,} rows)")

        if verbose:
            print("  Writing cells.parquet...", end=" ", flush=True)
        cells_df = _prepare_cells_dataframe(self)
        cells_df = _rebase_spatial_dataframe(cells_df, crop_origin)
        cells_df.to_parquet(outdir / "cells.parquet", index=False)
        if verbose:
            print(f"done. ({len(cells_df):,} cells)")

        if verbose:
            print("  Writing boundary parquets...", end=" ", flush=True)
        _cell_bdf = _prepare_boundary_dataframe(self, boundary_kind="cell")
        _cell_bdf = _rebase_spatial_dataframe(_cell_bdf, crop_origin)
        _cell_bdf.to_parquet(outdir / "cell_boundaries.parquet", index=False)
        _nuc_bdf = _prepare_boundary_dataframe(self, boundary_kind="nucleus")
        _nuc_bdf = _rebase_spatial_dataframe(_nuc_bdf, crop_origin)
        _nuc_bdf.to_parquet(outdir / "nucleus_boundaries.parquet", index=False)
        if verbose:
            print("done.")

        if verbose:
            print("  Writing cell_feature_matrix/...", end=" ", flush=True)
        _we._write_compressed_mex(self.adata, outdir)
        if verbose:
            print(f"done. ({self.adata.n_obs:,} cells × {self.adata.n_vars:,} features)")

        if verbose:
            print("  Writing analysis/...", end=" ", flush=True)
        _we._write_analysis_directory(self, outdir)
        if verbose:
            print("done.")

        _panel_src = os.path.join(self.xenium_folder, "gene_panel.json")
        if os.path.exists(_panel_src):
            shutil.copy2(_panel_src, outdir / "gene_panel.json")

        if verbose:
            print("  Writing cells.zarr.zip...", end=" ", flush=True)
        _obs_names = self.adata.obs_names.astype(str).tolist()
        _pixel_size = float(self.xenium_metadata.get("pixel_size", 0.2125))
        _we._write_cells_zarr(_obs_names, cells_df, _cell_bdf, _nuc_bdf, outdir,
                               pixel_size=_pixel_size)
        if verbose:
            print("done.")

        if verbose:
            print("  Writing cell_feature_matrix.zarr.zip...", end=" ", flush=True)
        _we._write_cfm_zarr(self.adata, _obs_names, outdir)
        if verbose:
            print("done.")

        if verbose:
            print("  Writing analysis.zarr.zip...", end=" ", flush=True)
        _we._write_analysis_zarr(self.adata, _obs_names, outdir)
        if verbose:
            print("done.")

        if verbose:
            print("  Writing transcripts.zarr.zip...", end=" ", flush=True)
        _source_fov_names = _we._load_source_fov_names(self)
        _we._write_transcripts_zarr(trans_df, outdir, fov_names=_source_fov_names)
        if verbose:
            print(f"done. ({n_transcripts:,} transcripts)")

        _xenium_explorer_files = {
            "cells_zarr_filepath": "cells.zarr.zip",
            "cell_features_zarr_filepath": "cell_feature_matrix.zarr.zip",
            "analysis_zarr_filepath": "analysis.zarr.zip",
            "transcripts_zarr_filepath": "transcripts.zarr.zip",
        }

        if verbose:
            print("  Writing experiment.xenium...", end=" ", flush=True)
        _we._write_experiment_xenium(
            self, outdir, n_transcripts=n_transcripts,
            xenium_explorer_files=_xenium_explorer_files,
        )
        if verbose:
            print("done.")

        if include_morphology:
            _morph_src = os.path.join(self.xenium_folder, "morphology.ome.tif")
            if os.path.exists(_morph_src):
                if verbose:
                    print("  Writing morphology.ome.tif...", end=" ", flush=True)
                _write_morphology_for_slice(
                    self,
                    output_path=outdir / "morphology.ome.tif",
                    crop=crop_morphology,
                    pyramidal=pyramidal_morphology,
                    pyramid_scale=pyramid_scale,
                    tile=morphology_tile,
                )
                if verbose:
                    print("done.")
            elif verbose:
                print("  morphology.ome.tif not found — skipping.")

        if verbose:
            print(f"\nXenium Explorer bundle written to: {outdir}")

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
        from pathlib import Path

        if matrix_format.lower() != "mex":
            raise ValueError("Only matrix_format='mex' is currently supported.")

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        required_outputs = [
            outdir / "transcripts.parquet",
            outdir / "barcodes.tsv",
            outdir / "features.tsv",
            outdir / "matrix.mtx",
            outdir / "cells.parquet",
            outdir / "cell_boundaries.parquet",
            outdir / "nucleus_boundaries.parquet",
        ]
        if include_morphology:
            required_outputs.append(outdir / "morphology.ome.tif")
        if include_protein_images and self.protein_images is not None:
            required_outputs.extend(outdir / "morphology_focus" / name for name in self.protein_images["filenames"])

        existing = [path for path in required_outputs if path.exists()]
        if existing and not overwrite:
            existing_str = ", ".join(path.name for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing output files: {existing_str}")

        crop_origin_um = _export_crop_origin_um(self) if rebase_coordinates else (0.0, 0.0)

        transcripts_df = _rebase_spatial_dataframe(self.trans, crop_origin_um)
        transcripts_df.to_parquet(outdir / "transcripts.parquet", index=False)

        cells_df = _prepare_cells_dataframe(self)
        cells_df = _rebase_spatial_dataframe(cells_df, crop_origin_um)
        cells_df.to_parquet(outdir / "cells.parquet", index=False)

        cell_boundary_df = _prepare_boundary_dataframe(self, boundary_kind="cell")
        cell_boundary_df = _rebase_spatial_dataframe(cell_boundary_df, crop_origin_um)
        cell_boundary_df.to_parquet(outdir / "cell_boundaries.parquet", index=False)

        nucleus_boundary_df = _prepare_boundary_dataframe(self, boundary_kind="nucleus")
        nucleus_boundary_df = _rebase_spatial_dataframe(nucleus_boundary_df, crop_origin_um)
        nucleus_boundary_df.to_parquet(outdir / "nucleus_boundaries.parquet", index=False)

        _write_10x_mex(self.adata, outdir)

        if include_morphology:
            _write_morphology_for_slice(
                self,
                output_path=outdir / "morphology.ome.tif",
                crop=crop_morphology,
                pyramidal=pyramidal_morphology,
                pyramid_scale=pyramid_scale,
                tile=morphology_tile,
            )

        if include_protein_images and self.protein_images is not None:
            _write_protein_images_for_slice(
                self,
                output_dir=outdir / "morphology_focus",
                crop=crop_protein_images,
                pyramid_scale=pyramid_scale,
                tile=morphology_tile,
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
        from PIL import Image
        # Check if features is None or a string, and convert to list if necessary
        if features is None:
            features = self.features
            title = 'All Features'
        elif isinstance(features, str):
            features = [features]

        toplot = self.trans[self.trans['feature_name'].isin(features)]
        
        for feature in features:
            if feature not in self.features:
                features.remove(feature)
                print(f'Feature "{feature}" not found in the dataset. Skipping...')

        if len(features) == 0:
            print('No valid features found in the dataset. Exiting...')
            return None
             
        img = create_binned_image(toplot, 
            bin_size=bin_size, 
            colormap=colormap,
            vmax=vmax,
            vmin=vmin)
        
        w,h = img.size
        ar = w/h
        dpi = pl.rcParams['figure.dpi']

        if return_img:
            return img
        else:
            pl.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
            pl.imshow(img, cmap=colormap)
            pl.axis('off')

            if title == '':
                if len(features) < 3:
                    title = ', '.join(features)
                else:
                    title = ', '.join(features[:3]) + '...'

            pl.title(title, fontsize=12)
            pl.tight_layout()
            pl.show()

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
                    "Expected a 'morphology_focus' folder with ch*.ome.tif files."
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
                rgba = pl.get_cmap(splat_cmap)(disp[..., 0]).astype(np.float32)
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
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.colors as mcolors

        gdf = self.cell_boundaries if kind == 'cell' else self.nucleus_boundaries
        if gdf is None or len(gdf) == 0:
            raise ValueError(
                f"No {kind} boundaries loaded. "
                "Check that boundary parquet files or cells.zarr.zip were available on init."
            )

        # ── spatial subset ────────────────────────────────────────────────────
        if bounds is None and getattr(self, 'active_roi', None) is not None:
            xmin, xmax, ymin, ymax = _roi_bounds_um(self.active_roi)
        elif bounds is not None:
            xmin, xmax, ymin, ymax = bounds
        else:
            xmin = ymin = xmax = ymax = None

        if xmin is not None:
            cx = gdf.geometry.centroid.x
            cy = gdf.geometry.centroid.y
            mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
            gdf = gdf[mask]

        if max_cells is not None and len(gdf) > max_cells:
            gdf = gdf.sample(max_cells, random_state=0)

        if len(gdf) == 0:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            return ax

        # ── build per-cell face colours ───────────────────────────────────────
        if color_by is not None and self.adata is not None and color_by in self.adata.obs.columns:
            obs_col = self.adata.obs[color_by].reindex(gdf.index)

            # If IDs don't match (e.g. zarr int IDs vs parquet hash IDs), fall
            # back to a nearest-centroid spatial join.
            if obs_col.isna().all():
                from scipy.spatial import KDTree
                obs_sub = self.adata.obs[['x_centroid', 'y_centroid', color_by]].dropna(
                    subset=['x_centroid', 'y_centroid'])
                tree = KDTree(obs_sub[['x_centroid', 'y_centroid']].values)
                bnd_cx = gdf.geometry.centroid.x.values
                bnd_cy = gdf.geometry.centroid.y.values
                _, nn_idx = tree.query(np.column_stack([bnd_cx, bnd_cy]))
                obs_col = pd.Series(
                    obs_sub[color_by].iloc[nn_idx].values,
                    index=gdf.index,
                )

            categories = list(obs_col.cat.categories) if hasattr(obs_col, 'cat') \
                         else sorted(obs_col.dropna().unique())
            if palette is None:
                palette = dict(zip(categories, generate_palette(len(categories))))
            raw_colors = obs_col.astype(object).map(palette).fillna('gray').tolist()
            face_rgba = [(*mcolors.to_rgb(c), face_alpha) for c in raw_colors]
        else:
            if facecolor == 'none':
                face_rgba = [(0, 0, 0, 0)] * len(gdf)
            else:
                rgb = mcolors.to_rgb(facecolor)
                face_rgba = [(*rgb, face_alpha)] * len(gdf)

        edge_rgb  = mcolors.to_rgb(edgecolor)
        edge_rgba = (*edge_rgb, edge_alpha)

        # ── y-flip: match splat/show_image convention ─────────────────────────
        # Both splat (y_idx = ymax - y) and show_ome_tiff (img[::-1]) display
        # with physical y inverted: small y at top, large y at bottom.
        # Reflect polygon vertices the same way: y_plot = y_lo + y_hi - y_phys.
        if ax is not None:
            y_lo, y_hi = ax.get_ylim()
        elif xmin is not None:
            y_lo, y_hi = ymin, ymax
        else:
            y_lo = gdf.geometry.bounds['miny'].min()
            y_hi = gdf.geometry.bounds['maxy'].max()

        def _flip_coords(coords):
            arr = np.array(coords)
            arr[:, 1] = y_lo + y_hi - arr[:, 1]
            return arr

        # ── build PatchCollection ─────────────────────────────────────────────
        patches = []
        for geom in gdf.geometry:
            # Handle both Polygon and MultiPolygon
            if geom.geom_type == 'Polygon':
                polys = [geom]
            else:
                polys = list(geom.geoms)
            for poly in polys:
                patches.append(MplPolygon(_flip_coords(poly.exterior.coords), closed=True))

        pc = PatchCollection(
            patches,
            facecolors=face_rgba,
            edgecolors=[edge_rgba] * len(patches),
            linewidths=linewidth,
        )

        # ── draw ──────────────────────────────────────────────────────────────
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')
            ax.set_xlim(gdf.geometry.bounds['minx'].min(),
                        gdf.geometry.bounds['maxx'].max())
            ax.set_ylim(y_lo, y_hi)

        ax.add_collection(pc)
        ax.set_aspect('equal')

        # ── legend when color_by is set ───────────────────────────────────────
        if color_by is not None and palette is not None:
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=(*mcolors.to_rgb(c), face_alpha),
                             edgecolor=edge_rgba,
                             label=str(k))
                       for k, c in palette.items()
                       if k in (obs_col.values if 'obs_col' in dir() else [])]
            if handles:
                ax.legend(handles=handles, fontsize='small',
                          labelcolor='white', facecolor='black',
                          edgecolor='black', loc='upper right')

        return ax

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
        import anndata as ad
        from scipy.sparse import coo_matrix
        print(f'Creating binned AnnData object with bin size of {bin_size}um...')

        include_features = _normalize_feature_selection(
            include_features,
            self.features,
            arg_name="include_features",
        )

        if isinstance(self.trans, LazyTranscripts):
            bounds = None if self.subset_roi is None else self.subset_roi.bounds
            parquet_path = os.path.join(self.xenium_folder, "transcripts.parquet")
            need_metadata = exclude_unassigned or (distance_to_nucleus is not None)

            use_parquet = False
            parquet_columns = ["x_location", "y_location", "feature_name"]
            if exclude_unassigned:
                parquet_columns.append("cell_id")
            if distance_to_nucleus is not None:
                parquet_columns.append("nucleus_distance")

            if need_metadata and os.path.exists(parquet_path):
                try:
                    import pyarrow.parquet as pq
                    schema_names = set(pq.ParquetFile(parquet_path).schema.names)
                    use_parquet = set(parquet_columns).issubset(schema_names)
                except Exception:
                    use_parquet = False

            if use_parquet:
                print("Using transcripts.parquet to preserve transcript metadata filters...")
                df = pd.read_parquet(parquet_path, columns=parquet_columns)
                if bounds is not None:
                    df = df[
                        (df["x_location"] >= bounds[0]) &
                        (df["x_location"] <= bounds[1]) &
                        (df["y_location"] >= bounds[2]) &
                        (df["y_location"] <= bounds[3])
                    ].copy()
            else:
                if need_metadata:
                    print("Transcript metadata are unavailable in the lazy Zarr view; proceeding without them.")
                df = self.trans.query(
                    genes=include_features,
                    quality='all',
                    **(
                        {}
                        if bounds is None
                        else dict(
                            xmin=bounds[0],
                            xmax=bounds[1],
                            ymin=bounds[2],
                            ymax=bounds[3],
                        )
                    ),
                )
        else:
            df = self.trans.copy()

        df = df[df['feature_name'].isin(include_features)].copy()

        if exclude_unassigned:
            if 'cell_id' in df.columns:
                print('Using only transcripts assigned to cells/nuclei...')
                df = df[df['cell_id'] != 'UNASSIGNED'].copy()
            else:
                print("Transcript cell assignments are unavailable; skipping exclude_unassigned filter.")

        if distance_to_nucleus is not None:
            if 'nucleus_distance' in df.columns:
                print(f'Excluding transcripts further than {distance_to_nucleus}um from the nucleus...')
                df = df[df['nucleus_distance'] <= distance_to_nucleus].copy()
            else:
                print("Transcript nucleus distances are unavailable; skipping distance_to_nucleus filter.")

        binned = _bin_transcript_dataframe(df, bin_size=bin_size, feature_names=include_features)

        expression_matrix = coo_matrix(
            (binned["counts"], (binned["bin_rows"], binned["feature_indices"])),
            shape=(len(binned["occupied_x_bins"]), len(binned["feature_names"])),
        )

        adata = ad.AnnData(X=expression_matrix.tocsr())
        adata.obs_names = [f'bin_{i}' for i in range(adata.n_obs)]
        adata.var_names = list(binned["feature_names"])

        if adata.n_obs:
            x_bins = binned["occupied_x_bins"].astype(int)
            y_bins = binned["occupied_y_bins"].astype(int)
            ymid = (y_bins.max() - y_bins.min()) / 2.0
            y_bins_plot = -(y_bins - ymid).astype(int)
            bin_coords = np.column_stack([x_bins, y_bins_plot])
            spatial_um = np.column_stack([
                (x_bins + 0.5) * float(bin_size),
                (y_bins + 0.5) * float(bin_size),
            ])
        else:
            bin_coords = np.zeros((0, 2), dtype=int)
            spatial_um = np.zeros((0, 2), dtype=float)

        adata.obs[['x_bin', 'y_bin']] = bin_coords
        adata.obsm['spatial'] = bin_coords
        adata.obsm['spatial_um'] = spatial_um

        self.binned_adata = adata
        self.binned_adata.uns['bin_size'] = float(bin_size)
        self.binned_adata.uns['pixel_size'] = self.pixel_size
        self.binned_adata.uns['x_bin_edges'] = (
            np.array([], dtype=float)
            if not len(binned["x_bin_values"])
            else np.arange(
                binned["x_bin_values"][0] * bin_size,
                (binned["x_bin_values"][-1] + 1) * bin_size + bin_size,
                bin_size,
                dtype=float,
            )
        )
        self.binned_adata.uns['y_bin_edges'] = (
            np.array([], dtype=float)
            if not len(binned["y_bin_values"])
            else np.arange(
                binned["y_bin_values"][0] * bin_size,
                (binned["y_bin_values"][-1] + 1) * bin_size + bin_size,
                bin_size,
                dtype=float,
            )
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
        bin_size = self.binned_adata.uns['bin_size']

        if genes is None:
            genes = self.binned_adata.var_names.tolist()
        elif isinstance(genes, str):
            genes = [genes]

        if not output_path.endswith('.ome.tiff'):
            raise ValueError("Output path must end with .ome.tiff")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

        imdata = create_multilayer_image(self, genes)
        if flip_y:
            imdata = imdata[::-1, :, :]

        _write_ome_tiff(imdata,
                        channel_names=genes,
                        compression=compression,
                        physical_size_x=bin_size,
                        physical_size_y=bin_size,
                        output_path=output_path,
                        pyramidal=pyramidal,
                        pyramid_scale=pyramid_scale,
                        tile=tile)
    
def read_ROI_from_csv(csv_path, selection=None):
    """
    Read a Xenium Explorer ROI CSV and return (coords, area_um2).

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV exported from Xenium Explorer. Expected header lines:
        #Selection names: name1, name2, ...
        #Areas (µm^2): a1, a2, ...
    selection : str | None
        Region name to extract. If None and only one region exists, that region is returned.
        If None and multiple regions exist, a ValueError is raised listing options.

    Returns
    -------
    coords : (N, 2) float ndarray
        X,Y coordinates for the requested region.
    area_um2 : float | None
        Area for the requested region in µm^2 (from header). None if not found.

    Raises
    ------
    ValueError
        If multiple regions exist and `selection` is not specified or cannot be matched.
    """
    from pathlib import Path
    csv_path = Path(csv_path)

    # --- Parse comment header for names and areas ---
    header_lines = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break  # comments are at the top; stop once data starts

    names_line = next((ln for ln in header_lines if ln.lower().startswith("#selection names")), None)
    areas_line = next((ln for ln in header_lines if "areas" in ln.lower()), None)

    def _parse_list_from_comment(line):
        # e.g., "#Selection names: A, B, C" -> ["A","B","C"]
        return [s.strip() for s in line.split(":", 1)[1].split(",")] if line and ":" in line else []

    def _parse_float_list_from_comment(line):
        # e.g., "#Areas (µm^2): 1.0, 2, 3.5" -> [1.0, 2.0, 3.5]
        vals = _parse_list_from_comment(line)
        out = []
        for v in vals:
            v_clean = re.sub(r"[^\d.+\-eE]", "", v)  # strip units/commas if present
            try:
                out.append(float(v_clean))
            except ValueError:
                out.append(np.nan)
        return out

    header_names = _parse_list_from_comment(names_line)
    header_areas = _parse_float_list_from_comment(areas_line)

    # Map area to name when lengths align; else leave as None per-name
    area_map = {}
    if header_names and header_areas and len(header_names) == len(header_areas):
        area_map = dict(zip(header_names, header_areas))

    # --- Read coordinate table (comments skipped automatically) ---
    df = pd.read_csv(csv_path, comment="#")
    if not {"Selection", "X", "Y"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Selection', 'X', 'Y'.")

    unique_selections = df["Selection"].astype(str).unique().tolist()

    # If user did not specify selection:
    if selection is None:
        if len(unique_selections) == 1:
            selection = unique_selections[0]
        else:
            opts = ", ".join(unique_selections)
            raise ValueError(
                "Multiple regions found. Please specify one of: "
                f"{opts}"
            )

    # Try exact, then case-insensitive, then prefix/substring matches
    sel = str(selection)
    chosen = None
    if sel in unique_selections:
        chosen = sel
    else:
        # case-insensitive exact
        ci_map = {s.lower(): s for s in unique_selections}
        if sel.lower() in ci_map:
            chosen = ci_map[sel.lower()]
        else:
            # prefix/substring (case-insensitive)
            cands = [s for s in unique_selections if s.lower().startswith(sel.lower())]
            if not cands:
                cands = [s for s in unique_selections if sel.lower() in s.lower()]
            if len(cands) == 1:
                chosen = cands[0]
            else:
                opts = ", ".join(unique_selections)
                raise ValueError(
                    f"Selection '{selection}' not uniquely matched. "
                    f"Available options: {opts}"
                )

    sub = df[df["Selection"].astype(str) == chosen]
    if sub.empty:
        raise ValueError(f"No coordinates found for selection '{chosen}'.")

    coords = sub[["X", "Y"]].to_numpy(dtype=float)
    area_um2 = area_map.get(chosen)
    return coords, area_um2

def read_ROI_from_geojson(geojson_path, feature=None, return_gdf=False, scale_factor=1.0):
    """
    Read a GeoJSON ROI and return a shapely geometry suitable for slicing Xenium data.

    Parameters
    ----------
    geojson_path : str | Path
        Path to a GeoJSON file containing one or more polygon features.
    feature : str | int | None
        Optional feature selector. If `str`, matches a feature by its ``name`` property.
        If `int`, selects by zero-based feature index. If None, all features are merged.
    return_gdf : bool, default=False
        If True, also return the loaded GeoDataFrame.
    scale_factor : float, default=1.0
        Uniform scale factor applied to ROI coordinates after loading. Xenium GeoJSON
        ROIs exported in pixel units should typically use the experiment pixel size
        (microns per pixel) here.

    Returns
    -------
    geometry : shapely geometry
        Polygon or MultiPolygon ROI geometry.
    area_um2 : float
        Area of the ROI geometry in the native Xenium coordinate system (um^2).
    gdf : GeoDataFrame, optional
        Returned only when ``return_gdf=True``.
    """
    rois = gpd.read_file(geojson_path)
    if rois.empty:
        raise ValueError(f"No ROI features found in GeoJSON file: {geojson_path}")
    if "geometry" not in rois:
        raise ValueError(f"GeoJSON file does not contain a geometry column: {geojson_path}")

    rois = rois.loc[rois.geometry.notna()].copy()
    if rois.empty:
        raise ValueError(f"GeoJSON file contains no valid geometries: {geojson_path}")

    if scale_factor != 1.0:
        rois["geometry"] = rois.scale(xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

    if feature is None:
        roi_geometry = rois.geometry.union_all()
    elif isinstance(feature, int):
        if feature < 0 or feature >= len(rois):
            raise IndexError(f"Feature index {feature} is out of bounds for {len(rois)} ROI feature(s).")
        roi_geometry = rois.geometry.iloc[feature]
    elif isinstance(feature, str):
        if "name" not in rois.columns:
            raise ValueError("GeoJSON feature selection by name requires a 'name' property.")
        matches = rois.loc[rois["name"].astype(str) == feature]
        if matches.empty:
            detected = ", ".join(rois["name"].dropna().astype(str).unique().tolist())
            raise ValueError(f'ROI feature "{feature}" not found. Detected names: {detected}')
        roi_geometry = matches.geometry.union_all()
    else:
        raise TypeError("feature must be None, a string feature name, or an integer feature index.")

    area_um2 = roi_geometry.area
    if return_gdf:
        return roi_geometry, area_um2, rois
    return roi_geometry, area_um2

def _coerce_roi_geometry(roi_like, selection=None, pixel_size=1.0, scale_geojson=True):
    from shapely.geometry import Polygon

    if isinstance(roi_like, ROI):
        return roi_like.geometry
    if isinstance(roi_like, ROIClass):
        return roi_like.union.geometry

    if hasattr(roi_like, "geom_type"):
        return roi_like

    if isinstance(roi_like, str):
        if not os.path.exists(roi_like):
            raise FileNotFoundError(f"ROI file not found: {roi_like}")
        if roi_like.lower().endswith((".geojson", ".json")):
            scale_factor = pixel_size if scale_geojson else 1.0
            roi_geometry, _ = read_ROI_from_geojson(roi_like, feature=selection, scale_factor=scale_factor)
            return roi_geometry
        roi_like, _ = read_ROI_from_csv(roi_like, selection=selection)

    if isinstance(roi_like, np.ndarray):
        if roi_like.ndim != 2 or roi_like.shape[1] != 2:
            raise ValueError("ROI must be a 2D numpy array with shape (n, 2).")
        return Polygon(roi_like)

    if isinstance(roi_like, pd.DataFrame):
        if roi_like.shape[1] != 2:
            raise ValueError("ROI must be a DataFrame with 2 columns.")
        return Polygon(roi_like.values)

    raise TypeError(
        "ROI must be a shapely geometry, a path to a ROI file, a 2D numpy array, or a 2-column DataFrame."
    )


def _coerce_roi(roi_like, selection=None, pixel_size=1.0, scale_geojson=True, name=None, source=None, poly_kwargs=None):
    if isinstance(roi_like, ROI):
        return roi_like
    if isinstance(roi_like, ROIClass):
        return roi_like.union
    geometry = _coerce_roi_geometry(
        roi_like,
        selection=selection,
        pixel_size=pixel_size,
        scale_geojson=scale_geojson,
    )
    return ROI.from_geometry(
        geometry,
        name=name,
        source=source,
        poly_kwargs=poly_kwargs,
    )

def _geometry_to_roi_points(geometry):
    if geometry.geom_type == "Polygon":
        return np.asarray(geometry.exterior.coords)
    if geometry.geom_type == "MultiPolygon":
        largest = max(geometry.geoms, key=lambda geom: geom.area)
        return np.asarray(largest.exterior.coords)
    raise TypeError(f"Unsupported ROI geometry type: {geometry.geom_type}")

def _normalize_roi_property(value):
    if isinstance(value, dict):
        for key in ("name", "label", "class", "classification", "value", "title"):
            if key in value and value[key] not in (None, ""):
                return str(value[key])
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _normalize_roi_property(value[0])
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _geojson_class_name(row):
    for key in (
        "class_name",
        "class",
        "classification",
        "annotation_class",
        "region_class",
        "group",
        "object_type",
        "type",
    ):
        if key in row.index:
            value = _normalize_roi_property(row.get(key))
            if value is not None:
                return value
    return None


def _geojson_selection_name(row, idx):
    for key in (
        "selection_name",
        "selection",
        "name",
        "label",
        "region_name",
        "annotation_name",
    ):
        if key in row.index:
            value = _normalize_roi_property(row.get(key))
            if value is not None:
                return value
    return f"selection_{idx}"


def _make_roi_name(class_name, selection_name, idx):
    if class_name and selection_name:
        return f"{class_name}:{selection_name}"
    if selection_name:
        return selection_name
    if class_name:
        return f"{class_name}_{idx}"
    return f"selection_{idx}"


def _register_roi_record(roi_store, name, geometry, selection_name=None, class_name=None, poly_kwargs=None, source=None, area_um2=None, metadata=None):
    roi = ROI.from_geometry(
        geometry,
        name=str(name),
        selection_name=selection_name,
        class_name=class_name,
        poly_kwargs=poly_kwargs,
        source=source,
        metadata={
            "area_um2": geometry.area if area_um2 is None else area_um2,
            **({} if metadata is None else metadata),
        },
    )
    if isinstance(roi_store, ROICollection):
        roi_store.add(roi)
    else:
        roi_store[str(name)] = roi

def _select_geojson_features(rois, roi_name=None):
    if roi_name is None:
        return [(str(row.get("name", f"feature_{idx}")), row.geometry) for idx, row in rois.iterrows()]

    selectors = roi_name if isinstance(roi_name, list) else [roi_name]
    selected = []
    used_labels = set()

    for selector in selectors:
        if isinstance(selector, int):
            if selector < 0 or selector >= len(rois):
                raise IndexError(f"Feature index {selector} is out of bounds for {len(rois)} ROI feature(s).")
            row = rois.iloc[selector]
            label = str(row.get("name", f"feature_{selector}"))
            if label in used_labels:
                continue
            selected.append((label, row.geometry))
            used_labels.add(label)
            continue

        selector = str(selector)
        if "name" not in rois.columns:
            raise ValueError("GeoJSON feature selection by name requires a 'name' property.")

        matches = rois.loc[rois["name"].astype(str) == selector]
        if matches.empty:
            detected = ", ".join(rois["name"].dropna().astype(str).unique().tolist())
            raise ValueError(f'ROI feature "{selector}" not found. Detected names: {detected}')

        for _, row in matches.iterrows():
            label = str(row.get("name", selector))
            if label in used_labels:
                continue
            selected.append((label, row.geometry))
            used_labels.add(label)

    return selected


def _match_geojson_row(row, selector):
    if selector is None:
        return True
    if isinstance(selector, int):
        return False
    selector = str(selector)
    candidates = {
        _normalize_roi_property(row.get("name")) if "name" in row.index else None,
        _normalize_roi_property(row.get("selection_name")) if "selection_name" in row.index else None,
        _normalize_roi_property(row.get("selection")) if "selection" in row.index else None,
        _geojson_selection_name(row, 0),
        _geojson_class_name(row),
    }
    candidates = {c for c in candidates if c}
    return selector in candidates

def _import_roi_records(roi_store, roi_file, roi_name=None, pixel_size=1.0, scale_geojson=True):
    roi_path = os.fspath(roi_file)

    if roi_path.lower().endswith((".geojson", ".json")):
        scale_factor = pixel_size if scale_geojson else 1.0
        _, _, rois = read_ROI_from_geojson(roi_path, return_gdf=True, scale_factor=scale_factor)
        imported_names = []
        selectors = roi_name if isinstance(roi_name, list) else ([roi_name] if roi_name is not None else None)
        class_counts = {}
        for idx, row in rois.iterrows():
            if selectors is not None:
                matched = False
                for selector in selectors:
                    if isinstance(selector, int) and selector == idx:
                        matched = True
                        break
                    if _match_geojson_row(row, selector):
                        matched = True
                        break
                if not matched:
                    continue

            class_name = _geojson_class_name(row)
            selection_name = _geojson_selection_name(row, idx)
            if class_name:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if selection_name == f"selection_{idx}":
                    selection_name = f"{class_name}_{class_counts[class_name]:03d}"
            roi_name_full = _make_roi_name(class_name, selection_name, idx)
            metadata = {
                col: row[col]
                for col in rois.columns
                if col != "geometry" and row[col] is not None
            }
            _register_roi_record(
                roi_store,
                name=roi_name_full,
                geometry=row.geometry,
                selection_name=selection_name,
                class_name=class_name,
                source=roi_path,
                metadata=metadata,
            )
            imported_names.append(roi_name_full)
        return imported_names

    polydf = pd.read_csv(roi_path, comment='#')
    if 'Selection' not in polydf.columns:
        raise KeyError('The provided ROI CSV file does not contain a "Selection" column. Please ensure it is a valid selections file produced by Xenium Analyzer.')

    polydf['Selection'] = polydf['Selection'].astype('str')
    detected_roi_names = polydf['Selection'].unique().tolist()

    if isinstance(roi_name, str):
        if roi_name not in detected_roi_names:
            raise ValueError(f'ROI name "{roi_name}" not found in file. Detected ROIs: {detected_roi_names}')
        selected_names = [roi_name]
    elif isinstance(roi_name, list):
        selected_names = [str(name) for name in roi_name]
        for name in selected_names:
            if name not in detected_roi_names:
                raise ValueError(f'ROI name "{name}" not found in file. Detected ROIs: {detected_roi_names}')
    elif roi_name is None:
        selected_names = detected_roi_names
    else:
        raise TypeError("CSV ROI selection must be None, a string ROI name, or a list of ROI names.")

    for name in selected_names:
        pts = polydf.loc[polydf['Selection'] == name, ['X', 'Y']].to_numpy()
        geometry = _coerce_roi_geometry(pts)
        _register_roi_record(
            roi_store,
            name=name,
            geometry=geometry,
            selection_name=name,
            source=roi_path,
        )

    return selected_names

def _read_xenium_table_if_present(xenium_folder, filename):
    path = os.path.join(xenium_folder, filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _resolve_roi_selection_selector(roi_collection, imported_names, selector):
    """
    Resolve an init-time crop selector against an imported ROI collection.

    Accepted selectors
    ------------------
    None / False:
        no cropping
    True:
        crop to the only imported selection; error if the file contains more than one
    int:
        crop to the Nth imported selection (0-based)
    str:
        crop by selection name, combined flat ROI name, or class name
    """
    if selector is None or selector is False:
        return None

    if selector is True:
        if len(imported_names) != 1:
            raise ValueError(
                "crop_to_selection=True requires the ROI file to contain exactly one "
                f"imported selection; found {len(imported_names)}."
            )
        return imported_names[0]

    if isinstance(selector, int):
        if selector < 0 or selector >= len(imported_names):
            raise IndexError(
                f"crop_to_selection index {selector} is out of bounds for "
                f"{len(imported_names)} imported selection(s)."
            )
        return imported_names[selector]

    if isinstance(selector, str):
        # Let downstream ROICollection / subset_to_roi resolve class names,
        # flat combined names, or explicit selection names.
        return selector

    return selector

def _export_crop_origin_um(xdata):
    subset_roi = getattr(xdata, "subset_roi", None)
    if subset_roi is None:
        return 0.0, 0.0
    minx, _, miny, _ = _roi_bounds_um(subset_roi)
    x0_px = max(0, int(np.floor(minx / xdata.pixel_size)))
    y0_px = max(0, int(np.floor(miny / xdata.pixel_size)))
    return x0_px * xdata.pixel_size, y0_px * xdata.pixel_size

def _rebase_spatial_dataframe(df, origin_um):
    if df is None:
        return None
    x0_um, y0_um = origin_um
    out = df.copy()
    x_cols = ["x_location", "x_centroid", "vertex_x"]
    y_cols = ["y_location", "y_centroid", "vertex_y"]
    for col in x_cols:
        if col in out.columns:
            out[col] = out[col] - x0_um
    for col in y_cols:
        if col in out.columns:
            out[col] = out[col] - y0_um
    return out

def _parse_ome_xml(ome_xml):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(ome_xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    return root, ns

def _detect_linked_protein_images(xenium_folder):
    folder = os.path.join(xenium_folder, "morphology_focus")
    if not os.path.isdir(folder):
        return None

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".ome.tif") and f.lower().startswith("ch")
    )
    if not files:
        return None

    first_file = os.path.join(folder, files[0])
    try:
        import tifffile
        with tifffile.TiffFile(first_file) as tif:
            ome_xml = tif.ome_metadata
    except Exception:
        return None

    if not ome_xml:
        return None

    root, ns = _parse_ome_xml(ome_xml)
    pixels = root.find('.//ome:Pixels', ns)
    if pixels is None:
        return None

    channel_names = []
    for channel in root.findall('.//ome:Channel', ns):
        channel_names.append(channel.attrib.get('Name', f'Channel {len(channel_names)}'))

    file_map = {}
    for tiffdata in root.findall('.//ome:TiffData', ns):
        first_c = int(tiffdata.attrib.get('FirstC', 0))
        uuid_node = tiffdata.find('ome:UUID', ns)
        if uuid_node is not None and uuid_node.attrib.get('FileName'):
            file_map[first_c] = uuid_node.attrib['FileName']

    filenames = [file_map.get(i, files[i] if i < len(files) else None) for i in range(len(channel_names))]
    if any(name is None for name in filenames):
        return None

    return {
        "folder": folder,
        "files": [os.path.join(folder, name) for name in filenames],
        "filenames": filenames,
        "channel_names": channel_names,
        "axes": "CYX",
        "shape": (
            int(pixels.attrib.get("SizeC", len(channel_names))),
            int(pixels.attrib.get("SizeY", 0)),
            int(pixels.attrib.get("SizeX", 0)),
        ),
        "pixel_size": float(pixels.attrib.get("PhysicalSizeX", 1.0)),
        "linked": True,
        "source_file": first_file,
        "ome_xml_template": ome_xml,
    }

def _prepare_cells_dataframe(xdata):
    keep_cells = set(xdata.adata.obs_names.astype(str))
    cells_df = _read_xenium_table_if_present(xdata.xenium_folder, "cells.parquet")

    if cells_df is not None and "cell_id" in cells_df.columns:
        cells_df["cell_id"] = cells_df["cell_id"].astype(str)
        cells_df = cells_df.loc[cells_df["cell_id"].isin(keep_cells)].copy()
        return cells_df

    fallback = pd.DataFrame(index=xdata.adata.obs_names.astype(str))
    fallback.index.name = "cell_id"
    if "spatial" in xdata.adata.obsm:
        fallback["x_centroid"] = xdata.adata.obsm["spatial"][:, 0]
        fallback["y_centroid"] = xdata.adata.obsm["spatial"][:, 1]
    counts = xdata.adata.layers["counts"] if "counts" in xdata.adata.layers else xdata.adata.X
    fallback["transcript_counts"] = np.asarray(counts.sum(axis=1)).ravel().astype(int)
    fallback["total_counts"] = fallback["transcript_counts"]
    return fallback.reset_index()

def _geometry_to_boundary_rows(cell_id, geometry, label_id):
    rows = []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for poly in polygons:
        coords = np.asarray(poly.exterior.coords)
        for x, y in coords:
            rows.append({
                "cell_id": str(cell_id),
                "vertex_x": float(x),
                "vertex_y": float(y),
                "label_id": int(label_id),
            })
    return rows

def _prepare_boundary_dataframe(xdata, boundary_kind="cell"):
    keep_cells = set(xdata.adata.obs_names.astype(str))
    if boundary_kind == "cell":
        filename = "cell_boundaries.parquet"
        geometry_df = xdata.cell_boundaries
    elif boundary_kind == "nucleus":
        filename = "nucleus_boundaries.parquet"
        geometry_df = xdata.nucleus_boundaries
    else:
        raise ValueError("boundary_kind must be 'cell' or 'nucleus'.")

    boundary_df = _read_xenium_table_if_present(xdata.xenium_folder, filename)
    if boundary_df is not None and "cell_id" in boundary_df.columns:
        boundary_df["cell_id"] = boundary_df["cell_id"].astype(str)
        return boundary_df.loc[boundary_df["cell_id"].isin(keep_cells)].copy()

    if geometry_df is None:
        return pd.DataFrame(columns=["cell_id", "vertex_x", "vertex_y", "label_id"])

    rows = []
    for label_id, (cell_id, row) in enumerate(geometry_df.iterrows(), start=1):
        rows.extend(_geometry_to_boundary_rows(cell_id, row.geometry, label_id))
    return pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y", "label_id"])

def _write_10x_mex(adata, output_dir):
    from pathlib import Path
    from scipy import io as spio

    output_dir = Path(output_dir)
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.transpose().tocsr()

    barcodes = adata.obs_names.astype(str)
    gene_ids = None
    for candidate in ("gene_ids", "id", "gene_id"):
        if candidate in adata.var.columns:
            gene_ids = adata.var[candidate].astype(str).to_numpy()
            break
    if gene_ids is None:
        gene_ids = adata.var_names.astype(str).to_numpy()

    feature_types = None
    for candidate in ("feature_types", "feature_type"):
        if candidate in adata.var.columns:
            feature_types = adata.var[candidate].astype(str).to_numpy()
            break
    if feature_types is None:
        feature_types = np.repeat("Gene Expression", adata.n_vars)

    features_df = pd.DataFrame({
        0: gene_ids,
        1: adata.var_names.astype(str),
        2: feature_types,
    })

    pd.Series(barcodes).to_csv(output_dir / "barcodes.tsv", sep="\t", header=False, index=False)
    features_df.to_csv(output_dir / "features.tsv", sep="\t", header=False, index=False)
    spio.mmwrite(str(output_dir / "matrix.mtx"), matrix)

def _roi_bounds_in_pixels(roi_geometry, pixel_size, image_shape):
    minx, miny, maxx, maxy = _shapely_bounds(roi_geometry)
    x0 = max(0, int(np.floor(minx / pixel_size)))
    y0 = max(0, int(np.floor(miny / pixel_size)))
    x1 = min(int(image_shape[-1]), int(np.ceil(maxx / pixel_size)))
    y1 = min(int(image_shape[-2]), int(np.ceil(maxy / pixel_size)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI bounding box does not overlap the morphology image.")
    return x0, x1, y0, y1


def _pixel_aligned_bounds_um(bounds, pixel_size):
    """Snap micron bounds to the pixel grid used for image crops."""
    xmin, xmax, ymin, ymax = map(float, bounds)
    x0 = np.floor(xmin / pixel_size) * pixel_size
    y0 = np.floor(ymin / pixel_size) * pixel_size
    x1 = np.ceil(xmax / pixel_size) * pixel_size
    y1 = np.ceil(ymax / pixel_size) * pixel_size
    return (x0, x1, y0, y1)


def _roi_bounds_um(roi_geometry):
    """Return ROI bounds in the plotting convention (xmin, xmax, ymin, ymax)."""
    if isinstance(roi_geometry, ROI):
        return roi_geometry.bounds
    minx, miny, maxx, maxy = _shapely_bounds(roi_geometry)
    return (minx, maxx, miny, maxy)


def _shapely_bounds(roi_geometry):
    """Return shapely-order bounds (minx, miny, maxx, maxy) for ROI-like inputs."""
    if isinstance(roi_geometry, ROI):
        return roi_geometry.shapely_bounds
    return tuple(map(float, roi_geometry.bounds))


def _image_extent_um(image_path, pixel_size):
    """Return the full-resolution image extent in microns."""
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        shape = series.shape
    return (0.0, shape[-1] * pixel_size, 0.0, shape[-2] * pixel_size)

def _scale_bounds_for_level(bounds, level):
    x0, x1, y0, y1 = bounds
    scale = 2 ** level
    return (
        int(np.floor(x0 / scale)),
        int(np.ceil(x1 / scale)),
        int(np.floor(y0 / scale)),
        int(np.ceil(y1 / scale)),
    )

def _cropped_shape_from_bounds(source_array, bounds):
    x0, x1, y0, y1 = bounds
    if source_array.ndim == 2:
        return (y1 - y0, x1 - x0)
    if source_array.ndim == 3:
        return (source_array.shape[0], y1 - y0, x1 - x0)
    raise ValueError(f"Unsupported source array ndim: {source_array.ndim}")

def _iter_cropped_tiles(source_array, bounds, tile=(1024, 1024)):
    x0, x1, y0, y1 = bounds
    tile_y, tile_x = tile
    for y in range(y0, y1, tile_y):
        for x in range(x0, x1, tile_x):
            ys = slice(y, min(y + tile_y, y1))
            xs = slice(x, min(x + tile_x, x1))
            if source_array.ndim == 2:
                yield np.asarray(source_array[ys, xs])
            elif source_array.ndim == 3:
                yield np.asarray(source_array[:, ys, xs])
            else:
                raise ValueError(f"Unsupported source array ndim: {source_array.ndim}")

def _source_series_level_arrays(series, zarr_store=None):
    import zarr
    if zarr_store is None:
        zarr_store = series.aszarr()
    root = zarr.open(zarr_store, mode="r")
    if hasattr(root, "shape"):
        return [root], zarr_store
    keys = sorted((int(k), k) for k in root.keys() if str(k).isdigit())
    arrays = [root[k] for _, k in keys]
    if not arrays:
        raise ValueError("Could not find pyramid level arrays in zarr store.")
    return arrays, zarr_store

def _build_pyramid_levels(image, scale_factor=2):
    if scale_factor < 2:
        raise ValueError("scale_factor must be >= 2 for pyramid generation.")

    levels = []
    current = image
    while min(current.shape[-2:]) > 1:
        next_level = current[:, ::scale_factor, ::scale_factor]
        if next_level.shape[-2:] == current.shape[-2:]:
            break
        levels.append(next_level)
        current = next_level
    return levels

def _write_pyramidal_ome_tiff(image,
                              output_path,
                              pixel_size,
                              tile=(1024, 1024),
                              pyramid_scale=2,
                              compression="jpeg2000"):
    import tifffile

    pyramid_levels = _build_pyramid_levels(image, scale_factor=pyramid_scale)
    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        tif.write(
            image,
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeY": pixel_size,
            },
            tile=tile,
            compression=compression,
            subifds=len(pyramid_levels),
        )
        for level in pyramid_levels:
            tif.write(
                level,
                tile=tile,
                compression=compression,
                subfiletype=1,
            )

def _write_pyramidal_ome_tiff_from_levels(level_arrays,
                                          output_path,
                                          base_bounds,
                                          pixel_size,
                                          tile=(1024, 1024),
                                          compression="jpeg2000"):
    import tifffile

    if not level_arrays:
        raise ValueError("level_arrays must not be empty.")

    cropped_levels = []
    for level, level_array in enumerate(level_arrays):
        level_bounds = _scale_bounds_for_level(base_bounds, level)
        cropped_levels.append(
            np.asarray(level_array[:, level_bounds[2]:level_bounds[3], level_bounds[0]:level_bounds[1]])
        )

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        tif.write(
            cropped_levels[0],
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeY": pixel_size,
            },
            tile=tile,
            compression=compression,
            subifds=max(0, len(cropped_levels) - 1),
        )
        for level in cropped_levels[1:]:
            tif.write(
                level,
                tile=tile,
                compression=compression,
                subfiletype=1,
            )

def _linked_ome_xml(channel_names,
                    filenames,
                    file_uuids,
                    size_x,
                    size_y,
                    pixel_size,
                    root_uuid,
                    template_xml=None,
                    dtype_name="uint16"):
    import re

    if template_xml is not None:
        xml_str = template_xml
        xml_str = re.sub(r'UUID="urn:uuid:[^"]+"', f'UUID="{root_uuid}"', xml_str, count=1)
        xml_str = re.sub(r'Type="[^"]+"', f'Type="{dtype_name}"', xml_str, count=1)
        xml_str = re.sub(r'SizeX="[^"]+"', f'SizeX="{size_x}"', xml_str, count=1)
        xml_str = re.sub(r'SizeY="[^"]+"', f'SizeY="{size_y}"', xml_str, count=1)
        xml_str = re.sub(r'SizeZ="[^"]+"', 'SizeZ="1"', xml_str, count=1)
        xml_str = re.sub(r'SizeC="[^"]+"', f'SizeC="{len(channel_names)}"', xml_str, count=1)
        xml_str = re.sub(r'SizeT="[^"]+"', 'SizeT="1"', xml_str, count=1)
        xml_str = re.sub(r'PhysicalSizeX="[^"]+"', f'PhysicalSizeX="{pixel_size}"', xml_str, count=1)
        xml_str = re.sub(r'PhysicalSizeY="[^"]+"', f'PhysicalSizeY="{pixel_size}"', xml_str, count=1)
        xml_str = xml_str.replace('PhysicalSizeXUnit="µm"', 'PhysicalSizeXUnit="&#181;m"')
        xml_str = xml_str.replace('PhysicalSizeYUnit="µm"', 'PhysicalSizeYUnit="&#181;m"')
        xml_str = xml_str.replace('WellOriginXUnit="µm"', 'WellOriginXUnit="&#181;m"')
        xml_str = xml_str.replace('WellOriginYUnit="µm"', 'WellOriginYUnit="&#181;m"')

        for i, name in enumerate(channel_names):
            xml_str = re.sub(
                rf'(<Channel[^>]*ID="Channel:{i}"[^>]*Name=")[^"]+(")',
                rf'\g<1>{name}\2',
                xml_str,
                count=1,
            )

        uuid_pattern = re.compile(r'(<UUID FileName=")([^"]+)(">)(urn:uuid:[^<]+)(</UUID>)')
        replacements = iter(zip(filenames, file_uuids))
        def _replace_uuid(match):
            try:
                filename, file_uuid = next(replacements)
            except StopIteration:
                return match.group(0)
            return f'{match.group(1)}{filename}{match.group(3)}{file_uuid}{match.group(5)}'
        xml_str = uuid_pattern.sub(_replace_uuid, xml_str)

        return xml_str.encode("ascii", "xmlcharrefreplace").decode("ascii")

    channel_entries = "\n".join(
        f'      <Channel ID="Channel:{i}" Name="{name}" SamplesPerPixel="1"/>'
        for i, name in enumerate(channel_names)
    )
    tiffdata_entries = "\n".join(
        f'      <TiffData FirstZ="0" FirstT="0" FirstC="{i}" PlaneCount="1"><UUID FileName="{filenames[i]}">{file_uuids[i]}</UUID></TiffData>'
        for i in range(len(channel_names))
    )
    xml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" UUID="{root_uuid}">
  <Instrument ID="Instrument:0">
    <Microscope Manufacturer="10x Genomics" Model="Xenium"/>
  </Instrument>
  <Image ID="Image:0">
    <InstrumentRef ID="Instrument:0"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{dtype_name}" SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" SizeC="{len(channel_names)}" SizeT="1" PhysicalSizeX="{pixel_size}" PhysicalSizeXUnit="um" PhysicalSizeY="{pixel_size}" PhysicalSizeYUnit="um">
{channel_entries}
{tiffdata_entries}
    </Pixels>
  </Image>
</OME>'''
    return xml_str.encode("ascii", "xmlcharrefreplace").decode("ascii")

def _write_linked_pyramidal_ome_tiffs(image,
                                      output_dir,
                                      filenames,
                                      channel_names,
                                      pixel_size,
                                      template_xml=None,
                                      tile=(1024, 1024),
                                      pyramid_scale=2,
                                      compression="jpeg2000"):
    import tifffile
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image.ndim != 3:
        raise ValueError("Expected protein image array with shape (C, Y, X).")
    if image.shape[0] != len(channel_names) or image.shape[0] != len(filenames):
        raise ValueError("Protein image channel count does not match filenames/channel names.")

    file_uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in filenames]

    for idx, filename in enumerate(filenames):
        channel_image = image[idx]
        pyramid_levels = _build_pyramid_levels(channel_image[np.newaxis, ...], scale_factor=pyramid_scale)
        ome_xml = _linked_ome_xml(
            channel_names=channel_names,
            filenames=filenames,
            file_uuids=file_uuids,
            size_x=image.shape[2],
            size_y=image.shape[1],
            pixel_size=pixel_size,
            root_uuid=file_uuids[idx],
            template_xml=template_xml,
            dtype_name=image.dtype.name,
        )
        with tifffile.TiffWriter(output_dir / filename, bigtiff=True) as tif:
            tif.write(
                channel_image,
                description=ome_xml,
                tile=tile,
                compression=compression,
                subifds=len(pyramid_levels),
            )
            for level in pyramid_levels:
                tif.write(
                    level[0],
                    tile=tile,
                    compression=compression,
                    subfiletype=1,
                )

def _write_linked_pyramidal_ome_tiffs_from_levels(level_arrays_by_channel,
                                                  output_dir,
                                                  filenames,
                                                  channel_names,
                                                  pixel_size,
                                                  base_bounds,
                                                  template_xml=None,
                                                  tile=(1024, 1024),
                                                  compression="jpeg2000"):
    import tifffile
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(level_arrays_by_channel) != len(channel_names) or len(level_arrays_by_channel) != len(filenames):
        raise ValueError("Protein channel count does not match level arrays.")

    file_uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in filenames]
    n_levels = len(level_arrays_by_channel[0])

    for idx, filename in enumerate(filenames):
        channel_levels = level_arrays_by_channel[idx]
        if len(channel_levels) != n_levels:
            raise ValueError("Protein channel pyramid depth mismatch.")
        ome_xml = _linked_ome_xml(
            channel_names=channel_names,
            filenames=filenames,
            file_uuids=file_uuids,
            size_x=base_bounds[1] - base_bounds[0],
            size_y=base_bounds[3] - base_bounds[2],
            pixel_size=pixel_size,
            root_uuid=file_uuids[idx],
            template_xml=template_xml,
            dtype_name=channel_levels[0].dtype.name,
        )
        with tifffile.TiffWriter(output_dir / filename, bigtiff=True) as tif:
            tif.write(
                data=_iter_cropped_tiles(channel_levels[0], base_bounds, tile=tile),
                shape=_cropped_shape_from_bounds(channel_levels[0], base_bounds),
                dtype=channel_levels[0].dtype,
                description=ome_xml,
                metadata=None,
                tile=tile,
                compression=compression,
                subifds=max(0, n_levels - 1),
            )
            for level, level_array in enumerate(channel_levels[1:], start=1):
                level_bounds = _scale_bounds_for_level(base_bounds, level)
                tif.write(
                    data=_iter_cropped_tiles(level_array, level_bounds, tile=tile),
                    shape=_cropped_shape_from_bounds(level_array, level_bounds),
                    dtype=level_array.dtype,
                    metadata=None,
                    tile=tile,
                    compression=compression,
                    subfiletype=1,
                )

def _write_morphology_for_slice(xdata,
                                output_path,
                                crop=True,
                                pyramidal=True,
                                pyramid_scale=2,
                                tile=(1024, 1024)):
    import tifffile
    import zarr

    morphology_path = os.path.join(xdata.xenium_folder, "morphology.ome.tif")
    if not os.path.exists(morphology_path):
        raise FileNotFoundError(f"Morphology image not found: {morphology_path}")

    subset_roi = getattr(xdata, "subset_roi", None)
    if not crop or subset_roi is None:
        shutil.copy2(morphology_path, output_path)
        return

    # Compressed Xenium OME-TIFFs may require imagecodecs for tiled region reads.
    with tifffile.TiffFile(morphology_path) as tif:
        series = tif.series[0]
        level_arrays, zarr_store = _source_series_level_arrays(series)
        try:
            base_bounds = _roi_bounds_in_pixels(subset_roi, xdata.pixel_size, level_arrays[0].shape)
            if pyramidal:
                _write_pyramidal_ome_tiff_from_levels(
                    level_arrays,
                    output_path=output_path,
                    base_bounds=base_bounds,
                    pixel_size=xdata.pixel_size,
                    tile=tile,
                )
            else:
                cropped = np.asarray(level_arrays[0][:, base_bounds[2]:base_bounds[3], base_bounds[0]:base_bounds[1]])
                tifffile.imwrite(
                    output_path,
                    cropped,
                    ome=True,
                    metadata={
                        "axes": "ZYX",
                        "PhysicalSizeX": xdata.pixel_size,
                        "PhysicalSizeY": xdata.pixel_size,
                    },
                )
        finally:
            zarr_store.close()

def _write_protein_images_for_slice(xdata,
                                    output_dir,
                                    crop=True,
                                    pyramid_scale=2,
                                    tile=(1024, 1024)):
    import tifffile
    import zarr
    from pathlib import Path

    protein_info = xdata.protein_images
    if protein_info is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_roi = getattr(xdata, "subset_roi", None)
    if not crop or subset_roi is None:
        for src, name in zip(protein_info["files"], protein_info["filenames"]):
            shutil.copy2(src, output_dir / name)
        return

    level_arrays_by_channel = []
    base_bounds = None
    for src in protein_info["files"]:
        with tifffile.TiffFile(src) as tif:
            page = tif.pages[0]
            levels = [zarr.open(page.aszarr(), mode="r")]
            if hasattr(page, "pages") and page.pages is not None:
                for subpage in page.pages:
                    levels.append(zarr.open(subpage.aszarr(), mode="r"))
            level_arrays_by_channel.append(levels)
            if base_bounds is None:
                base_bounds = _roi_bounds_in_pixels(subset_roi, xdata.pixel_size, levels[0].shape)

    _write_linked_pyramidal_ome_tiffs_from_levels(
        level_arrays_by_channel,
        output_dir=output_dir,
        filenames=protein_info["filenames"],
        channel_names=protein_info["channel_names"],
        pixel_size=xdata.pixel_size,
        base_bounds=base_bounds,
        template_xml=protein_info.get("ome_xml_template"),
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
    pl.figure(figsize=[fig_scale * aspect_ratio, fig_scale])

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
    fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
    ax.imshow(im)
    ax.axis('off')

    # Define colors for the channels
    colors = ['red', 'green', 'blue'][:len(gene_sets)]
    labels = list(gene_sets.keys())
    add_legend_outside(labels, colors, fig, ax)

    pl.tight_layout()
    pl.show()

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
    pl.figure(figsize=[fig_scale * aspect_ratio, fig_scale])

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
    fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])

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

    pl.tight_layout()
    pl.show()

def _write_ome_tiff(image_array,
                    channel_names,
                    channel_ids=None,
                    channel_colors=None,
                    physical_size_x=5,
                    physical_size_y=5,
                    significant_bits=12,
                    compression='zlib',
                    output_path=None,
                    pyramidal: bool = True,
                    pyramid_scale: int = 2,
                    tile: tuple = (1024, 1024)):
    import tifffile

    # Expect image_array of shape (Y, X, C)
    Y, X, C = image_array.shape
    if len(channel_names) != C:
        raise ValueError("Length of channel_names must equal the number of channels in image_array.")
    if channel_ids is not None and len(channel_ids) != C:
        raise ValueError("Length of channel_ids must equal the number of channels in image_array.")
    if channel_colors is not None and len(channel_colors) != C:
        raise ValueError("Length of channel_colors must equal the number of channels in image_array.")

    # Generate default channel IDs if none are provided.
    if channel_ids is None:
        channel_ids = [f"Channel:{i}" for i in range(C)]

    # --- Rearrange from (Y, X, C) to XYZCT order. ---
    # Insert singleton Z and T dimensions.
    # arr = np.transpose(image_array, (2, 0, 1))  # becomes (C, Y, X) #DEPRECATED
    # arr = arr[np.newaxis, :, np.newaxis, :, :]  # becomes (1, C, 1, Y, X) #DEPRECATED

    def to_tczyx(a):  # (Y, X, C) -> (1, C, 1, Y, X)
        return a.transpose(2, 0, 1)[np.newaxis, :, np.newaxis, :, :]
    arr = to_tczyx(image_array)

    # Build channel metadata entries.
    """
    channel_entries = []
    for i, name in enumerate(channel_names):
        if channel_colors is not None:
            entry = f'      <Channel ID="{channel_ids[i]}" Name="{name}" SamplesPerPixel="1" Color="{channel_colors[i]}" />'
        else:
            entry = f'      <Channel ID="{channel_ids[i]}" Name="{name}" SamplesPerPixel="1" />'
        channel_entries.append(entry)
    channel_entries_str = "\n".join(channel_entries)
    """

    channel_entries = []
    channel_maxes = image_array.reshape(-1, image_array.shape[2]).max(axis=0)
    
    for i, name in enumerate(channel_names):
        max_val = float(channel_maxes[i])  # convert to float to avoid dtype issues in XML
        color_attr = f' Color="{channel_colors[i]}"' if channel_colors is not None else ''
        entry = f'''      <Channel ID="{channel_ids[i]}" Name="{name}" SamplesPerPixel="1"{color_attr}>
            <DisplaySettings>
            <DisplayRangeMin>0</DisplayRangeMin>
            <DisplayRangeMax>{max_val}</DisplayRangeMax>
            </DisplaySettings>
        </Channel>'''
        channel_entries.append(entry)
    channel_entries_str = "\n".join(channel_entries)

    # Build the OME-XML metadata.
    
    # dtype name must be OME-compatible
    dtype_name = {
        np.dtype("uint8"): "uint8",
        np.dtype("uint16"): "uint16",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }.get(image_array.dtype, image_array.dtype.name)
    # in OME spec below, under Pixels:
    # Try substituting in dtype_name.  WORKING: image_array.dtype.name
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" ID="Pixels:0" Type="{image_array.dtype.name}"
            SizeX="{X}"
            SizeY="{Y}"
            SizeC="{C}"
            SizeZ="1"
            SizeT="1"
            PhysicalSizeX="{physical_size_x}"
            PhysicalSizeY="{physical_size_y}"
            SignificantBits="{significant_bits}">
            {channel_entries_str}
    </Pixels>
    </Image>
    </OME>'''

    if output_path is None:
        output_path = f'multilayer_{physical_size_x}um.ome.tiff'

    if pyramidal:
        # Build pyramid from (C, Y, X); _build_pyramid_levels subsamples [:, ::s, ::s]
        arr_cyx = image_array.transpose(2, 0, 1)  # (Y, X, C) → (C, Y, X)
        pyramid_levels = _build_pyramid_levels(arr_cyx, scale_factor=pyramid_scale)

        def _to_tczyx(a_cyx):
            return a_cyx[np.newaxis, :, np.newaxis, :, :]  # → (1, C, 1, Y, X)

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                _to_tczyx(arr_cyx),
                photometric='minisblack',
                compression=compression,
                description=ome_xml,
                metadata=None,
                tile=tile,
                subifds=len(pyramid_levels),
            )
            for level in pyramid_levels:
                tif.write(
                    _to_tczyx(level),
                    photometric='minisblack',
                    compression=compression,
                    metadata=None,
                    tile=tile,
                    subfiletype=1,
                )
    else:
        tifffile.imwrite(output_path,
                         arr,
                         photometric='minisblack',
                         compression=compression,
                         description=ome_xml,
                         metadata=None)

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
        fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
        ax.imshow(imdata, cmap=cmap)
        ax.axis('off')
        pl.show()

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
        dpi = pl.rcParams['figure.dpi']
    if figsize is None:
        figsize = pl.rcParams['figure.figsize']

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
        fig, ax = pl.subplots(figsize=figsize, dpi=dpi)

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
        pl.tight_layout()

    return ax

def _apply_weights(d, scheme: Optional[Literal["binary","inverse","gaussian"]], sigma: Optional[float]):
    if scheme in (None, "binary"):
        return np.ones_like(d, dtype=float)
    if scheme == "inverse":
        # avoid div-by-zero on self or coincident points
        return 1.0 / np.maximum(d, 1e-12)
    if scheme == "gaussian":
        if not sigma:
            raise ValueError("sigma must be provided for gaussian weighting")
        return np.exp(-(d**2) / (2.0 * sigma**2))
    raise ValueError(f"Unknown weight scheme: {scheme}")

def build_spatial_graph(
    coords: np.ndarray,
    *,
    use_radius: Optional[float] = None,   # in same units as coords (e.g., µm)
    n_neighbors: int = 30,                # used if use_radius is None
    include_self: bool = False,
    symmetrize: Literal["none","max","mean"] = "max",
    weight_scheme: Optional[Literal["binary","inverse","gaussian"]] = "binary",
    sigma: Optional[float] = None         # required if weight_scheme="gaussian"
) -> sparse.csr_matrix:
    """
    Return CSR adjacency (cells x cells) using cKDTree. No sklearn/squidpy.
    """
    from scipy.spatial import cKDTree
    N = coords.shape[0]
    tree = cKDTree(coords)

    if use_radius is not None:
        # Efficient radius graph with distances
        D = tree.sparse_distance_matrix(tree, max_distance=use_radius, output_type="coo_matrix")
        if not include_self:
            mask = D.row != D.col
            D = sparse.coo_matrix((D.data[mask], (D.row[mask], D.col[mask])), shape=(N, N))
        w = _apply_weights(D.data, weight_scheme, sigma)
        A = sparse.coo_matrix((w, (D.row, D.col)), shape=(N, N)).tocsr()
    else:
        # kNN graph
        k = n_neighbors + (1 if include_self else 0)
        d, idx = tree.query(coords, k=k)  # shapes: (N,k)
        # build COO
        rows = np.repeat(np.arange(N), k)
        cols = idx.ravel()
        if not include_self:
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]
            d = d.ravel()[mask]
        else:
            d = d.ravel()
        w = _apply_weights(d, weight_scheme, sigma)
        A = sparse.coo_matrix((w, (rows, cols)), shape=(N, N)).tocsr()

    # symmetrize if requested
    if symmetrize == "max":
        A = A.maximum(A.T)
    elif symmetrize == "mean":
        A = 0.5 * (A + A.T)
        A.eliminate_zeros()

    return A

def build_niches(
    adata,
    *,
    xy_key: Optional[str] = "spatial",     # adata.obsm key OR None to use obs[['x','y']]
    label_key: str = "graphclust",
    use_radius: Optional[float] = None,
    n_neighbors: int = 30,
    symmetrize: Literal["none","max","mean"] = "max",
    weight_scheme: Optional[Literal["binary","inverse","gaussian"]] = "binary",
    sigma: Optional[float] = None,
    normalize: Literal["none","prop","zscore"] = "prop",
    k_niches: Optional[int] = None,        # if set, assigns clusters via k-means
    key_added: str = "niche"
):
    """
    Builds a neighbor-composition matrix (cells x celltypes) and stores it in:
      - adata.obsm[f"{key_added}_X"]  (dense np.ndarray)
      - adata.uns[f"{key_added}_celltypes"] (list of column names)
      - adata.obsp["spatial_connectivities"] (sparse CSR adjacency)
      - optionally adata.obs[f"{key_added}_k{k_niches}"] if k_niches provided
    """
    # ---- coordinates ----
    if xy_key is not None and xy_key in adata.obsm:
        coords = np.asarray(adata.obsm[xy_key], dtype=float)
        if coords.shape[1] > 2:  # handle (x,y,*) by taking first two cols
            coords = coords[:, :2]
    else:
        coords = np.asarray(adata.obs[["x","y"]].values, dtype=float)

    # ---- spatial graph ----
    A = build_spatial_graph(
        coords,
        use_radius=use_radius,
        n_neighbors=n_neighbors,
        include_self=False,
        symmetrize=symmetrize,
        weight_scheme=weight_scheme,
        sigma=sigma,
    )
    adata.obsp["spatial_connectivities"] = A

    # ---- one-hot of cell types ----
    ct = adata.obs[label_key].astype("category")
    ct_names = list(ct.cat.categories)
    onehot_df = pd.get_dummies(ct, sparse=True)  # cells x celltypes
    onehot = sparse.csr_matrix(onehot_df.values)

    # ---- neighbor composition ----
    X = (A @ onehot).astype(float)  # counts or weighted sums

    # ---- normalization ----
    if normalize == "prop":
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        X = sparse.diags(1.0 / row_sums) @ X
    elif normalize == "zscore":
        col_means = np.asarray(X.mean(axis=0)).ravel()
        col_sqmeans = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        col_stds = np.sqrt(np.maximum(col_sqmeans - col_means**2, 1e-12))
        # center: X - mean
        X = X - sparse.csr_matrix(np.broadcast_to(col_means, X.shape))
        # scale
        invstd = 1.0 / np.where(col_stds == 0, 1.0, col_stds)
        X = X @ sparse.diags(invstd)

    # ---- stash ----
    adata.obsm[f"{key_added}_X"] = X.toarray() if sparse.issparse(X) else X
    adata.uns[f"{key_added}_celltypes"] = ct_names

    # ---- (optional) cluster in niche space ----
    if k_niches is not None and k_niches > 1:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k_niches, n_init="auto", random_state=0)
        labs = km.fit_predict(adata.obsm[f"{key_added}_X"])
        adata.obs[f"{key_added}_k{k_niches}"] = pd.Categorical(labs.astype(str))

    return adata

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
