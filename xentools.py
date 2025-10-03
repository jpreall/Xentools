#!/usr/bin/env python
# Need to figure out how to import jpplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import glob, os, sys
import scanpy as sc
import json
import geopandas as gpd
import re
from typing import Union


__all__ = ['XenData', 
           'read_xen_panel', 
           'read_json', 
           'prep_for_inference', 
           'create_bins', 
           'bin_expression', 
           'create_binned_image', 
           'create_polygon', 
           'generate_palette', 
           'plot_palette',
            'TP10K',
            'plot_binned_rgb',
            'plot_binned_greyscale',]

possible_jpplot_paths = [
    '/Users/jpreall/CSHL Dropbox Team Dropbox/Jon Preall/Preall_Lab/Preall/scripts/jpplot/master/',
    '/Users/jpreall/Dropbox/Preall_Lab/Preall/scripts/jpplot/master/',
    '/grid/preall/home/jpreall/jpplot/'
]
for jpplot_path in possible_jpplot_paths:
    if os.path.exists(jpplot_path):
        sys.path.append(jpplot_path)
        import jpplot
        break
else:
    raise FileNotFoundError("jpplot path not found. Please check the paths.")



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

def read_xenium_to_anndata(xenium_output_folder):
    xdir = xenium_output_folder
    #print(xdir)
    # Choose which file to load the cell feature matrix from
    # check for cell_feature_matrix.h5
    if os.path.exists(f'{xdir}/cell_feature_matrix.h5'):
        adata = jpplot.read_cellranger_h5(f'{xdir}/cell_feature_matrix.h5')
    elif os.path.exists(f'{xdir}/cell_feature_matrix/'):
        adata = jpplot.read_mtx_fromGEO(f'{xdir}/cell_feature_matrix/')

    # Read in cell-level metadata
    cells = pd.read_parquet(f'{xdir}/cells.parquet')
    cells.index = cells['cell_id'].astype('str')
    adata.obsm['spatial'] = cells.loc[:,['x_centroid','y_centroid']].values

    # Read in UMAP coords
    umap_file = xdir + '/analysis/umap/gene_expression_2_components/projection.csv'
    umap_coords = pd.read_csv(umap_file, index_col=0)
    umap_coords.index = umap_coords.index.astype('str')
    keep_cells = [name for name in adata.obs_names if name in umap_coords.index]

    # Thow out any cells without UMAP coords
    adata = adata[keep_cells,:].copy()
    adata.obsm['X_umap'] = umap_coords.loc[keep_cells,:].values

    # Flip spatial coords to match Xenium Ranger
    rot_matrix = [[1,0],[0,-1]]
    adata.obsm['spatial'] = adata.obsm['spatial'] @ rot_matrix

    # Match Xenium Ranger aspect ratio
    xrange = adata.obsm['spatial'][:,0].max() - adata.obsm['spatial'][:,0].min()
    yrange = adata.obsm['spatial'][:,1].max() - adata.obsm['spatial'][:,1].min()
    aspect_ratio = xrange/yrange
    #pl.rcParams['figure.figsize'] = [4*aspect_ratio,4]

    # Read in gene panel
    gene_panel = read_xen_panel(xdir + '/gene_panel.json')
    try:
        species = gene_panel['payload']['panel']['species']
    except:
        species = 'Unknown'
    adata.uns['genome'] = species

    # Start Scanpy Pipeline
    adata.layers['counts'] = adata.X.astype('int').copy()
    adata.obs['n_counts'] = adata.X.sum(1).A1.astype('int')
    adata.var['total_counts'] = adata.X.sum(0).A1.astype('int')
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1.astype('int')
    adata.obs['logUMIs'] = np.log(adata.obs['n_counts'] + 1)

    TP10K(adata)

    adata.X = np.log1p(adata.layers['TP10K'])
    adata.raw = adata.copy()
    
    # Add Xenium Ranger annotations
    print('Importing Xenium Ranger cluster annotations')
    cfiles = glob.glob(xdir + '/analysis/clustering/*/*csv')
    res = pd.DataFrame()
    for f in cfiles:
        cname = f.split('/')[-2].replace('gene_expression_','')
        #print(cname)
        clusterdf = pd.read_csv(f, index_col=0)
        clusterdf.index = clusterdf.index.astype('str')
        clusters = clusterdf['Cluster'].rename(cname).loc[adata.obs_names].astype('str').astype('category')
        res = pd.concat([res,clusters], axis=1)
    adata.obs[res.columns] = res

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
    clusters = pd.read_csv(clusters_file, index_col=0)
    clusters.index = clusters.index.astype('str')
    clusters['Cluster'] = clusters['Cluster'].astype('str')
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

    gene_panel = read_xen_panel(panel_file)

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
    def __init__(self, xenium_folder, verbose=True):
        import json
        # Read data using your existing function
        #self.celldata, self.trans, self.nucdata, self.clusters, self.gene_panel = read_xen_essentials(xenium_folder, verbose)
        self.trans, self.clusters, self.gene_panel = read_xen_essentials(xenium_folder, verbose)
        
        # You can also include other processing or attributes if needed
        # For example, you could compute additional derived attributes here
        if verbose:
            print('Reading in AnnData object')
        self.adata = read_xenium_to_anndata(xenium_folder)
        #jpplot.ncounts(self.adata)

        # Add cluster information to the AnnData object
        self.adata.obs = self.adata.obs.merge(self.clusters, left_index=True, right_index=True, how='left')

        # Add gene panel information to the AnnData object
        self.adata.var.merge(_make_gene_panel_df(self.gene_panel), 
                             left_index=True, right_index=True, how='left')

        xenium_file = f'{xenium_folder}/experiment.xenium'
        with open(xenium_file) as f:
            self.xenium_metadata = json.load(f)
        self.pixel_size = self.xenium_metadata['pixel_size']
        
        # Name the analysis run
        keys = ['run_name','slide_id','region_name']
        self.name = '_'.join([self.xenium_metadata[k] for k in keys])

        self.update_cell_names()

        # Read in cell and nucleus boundaries
        cell_boundaries_file = f'{xenium_folder}/cell_boundaries.parquet'
        nuc_boundaries_file = f'{xenium_folder}/nucleus_boundaries.parquet'

        self.ROIs = {}
        
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
        else:
            self.cell_boundaries = None

        if os.path.exists(nuc_boundaries_file):
            print('Reading in nucleus boundaries')
            # Read in nucleus boundaries
            self.nucleus_boundaries = pd.read_parquet(nuc_boundaries_file)
            self.nucleus_boundaries.set_index('cell_id', inplace=True)
            self.nucleus_boundaries.index = self.nucleus_boundaries.index.astype('str')
            # Flip the y-coordinates to match the Xenium Ranger orientation
            #self.nucleus_boundaries['vertex_y'] = -(self.nucleus_boundaries['vertex_y'] - self.nucleus_boundaries['vertex_y'].max())
            self.nucleus_boundaries = gpd.GeoDataFrame(
                self.nucleus_boundaries\
                    .groupby('cell_id')\
                    .apply(create_polygon), columns=['geometry'])
        else:
            self.nucleus_boundaries = None
            

        # Clean up the cell and nucleus boundaries to make sure they match the celldata
        # This is important because the cell and nucleus boundaries may contain cells
        # that are not present in the celldata or transcript data

        # Look for a morphology.ome.tiff file, save the path if it exists
        self.images = {}
        if os.path.exists(f'{xenium_folder}/morphology.ome.tif'):
            self.images['DAPI'] = f'{xenium_folder}/morphology.ome.tif'

        """
        if self.cell_boundaries is not None:
            print('Cleaning up cell boundaries...')
            keep_cells = [name for name in self.cell_boundaries.index if name in self.celldata.index]
            self.cell_boundaries = self.cell_boundaries.loc[keep_cells]

        if self.nucleus_boundaries is not None:
            print('Cleaning up nucleus boundaries...')
            keep_cells = [name for name in self.nucleus_boundaries.index if name in self.celldata.index]
            self.nucleus_boundaries = self.nucleus_boundaries.loc[keep_cells]
        """

    def update_cell_names(self):
        self._cell_names = sorted(self.trans['cell_id'].unique())
        if 'UNASSIGNED' in self._cell_names:
            self._cell_names.remove('UNASSIGNED')

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
        
        # Build the summary string
        summary = []
        summary.append(f"XenData Summary:")
        summary.append(f"-" * 50)
        summary.append(f"Slide/Region Name: {self.name}")
        summary.append(f"Xenium Kit Version: {kit_version}")
        summary.append(f"Panel: {panel_info}")
        summary.append(f"Number of Unique Cells: {total_cells:,}")
        summary.append(f"Number of Genes: {total_genes:,}")
        summary.append(f"Number of Clusters: {total_clusters}")
        #summary.append(f"Cells Marked as Unassigned: {'Yes' if unassigned_cells > 0 else 'No'}")
        
        if hasattr(self, 'adata') and self.adata is not None:
            # Additional AnnData statistics if available
            summary.append(f"Number of Transcripts: {self.adata.X.nnz:,}")
            #summary.append(f"Number of Nuclei: {self.adata.obs['nuclei'].sum() if 'nuclei' in self.adata.obs.columns else 'N/A'}")
        
        if self.images != {}:
            summary.append(f"Image Layers: {', '.join(self.images.keys())}")

        summary.append(f"-" * 50)
        return "\n".join(summary)

    def __repr__(self):
        return str(self)
    
    def import_ROI_xeniumanalyzer(self, roi_csv_file, roi_name):
        """
        Imports a region of interest generated with the lasso tool in Xenium Analyzer.
        Stores the ROI as a polygon, centroid, and points in the xdata.ROIs dictionary.

        Parameters:
        - roi_csv_file: XenData object containing transcript data.
        - roi_name: A human-readable name for this region (eg. Mouse ID + phenotype)
        """

        if roi_name is None:
            roi_name = os.path.basename(roi_csv_file).replace('.csv','')
            
        import matplotlib.patches as patches

        polydf = pd.read_csv(roi_csv_file,
            comment='#')
        
        def polygon_centroid(pts):
            x = pts[:,0]
            y = pts[:,1]
            
            # ensure polygon is closed
            if x[0] != x[-1] or y[0] != y[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])

            # Shoelace formula
            A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
            Cx = (1/(6*A)) * np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
            Cy = (1/(6*A)) * np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
            
            return Cx, Cy
        
        # Build and draw the polygon
        pts = polydf[['X','Y']].to_numpy()
        pts = pts/5
        cx, cy = polygon_centroid(pts)
        poly_kwargs = dict(closed=True, fill=False, edgecolor='aqua', linewidth=2)
        poly = patches.Polygon(pts, **poly_kwargs)

        self.ROIs[roi_name] = {
            'polygon': poly,
            'centroid': (cx, cy),
            'points': pts,
            'poly_kwargs': poly_kwargs,
        }

    def crop_to_ROI(self, ROI, selection=None):
        """
        Crop the data to a specified region of interest (ROI).
        Parameters:
        - ROI: a file path to a CSV file containing the ROI coordinates
        - or a numpy array with shape (n,2) where n is the number of points defining the ROI.
        - selection: if ROI is a file, the name of the ROI to select (if multiple are present in the file)
        The ROI should be in the format [[x1, y1], [x2, y2], ...].
        """
        from shapely.geometry import Polygon, Point
        from shapely import contains_xy

        #check if ROI is a file or a numpy array
        if isinstance(ROI, str):
            ROI, roi_area = read_ROI_from_csv(ROI, selection=selection)
        elif isinstance(ROI, np.ndarray):
            if ROI.shape[1] != 2:
                raise ValueError("ROI must be a 2D numpy array with shape (n, 2).")
        elif isinstance(ROI, pd.DataFrame):
            if ROI.shape[1] != 2:
                raise ValueError("ROI must be a DataFrame with 2 columns.")
            ROI = ROI.values

        # Create a polygon from the ROI
        roi_polygon = Polygon(ROI)
        
        # Apply the filter to the DataFrame
        points = self.trans[['x_location', 'y_location']].values
        #ROI_filter = np.array([roi_polygon.contains(Point(p)) for p in points])
        ROI_filter = contains_xy(roi_polygon, points[:,0], points[:,1])

        # Filter transcripts
        #self.trans = self.trans[make_filter(self.trans, ROI)]
        self.trans = self.trans[ROI_filter].copy()

        # Filter celldata
        self.update_cell_names()

        #keep_cells = [name for name in self.cell_names if name in self.celldata.index]
        #self.celldata = self.celldata.loc[keep_cells]
        keep_cells = [name for name in self.cell_names if name in self.cell_boundaries.index]
        self.cell_boundaries = self.cell_boundaries.loc[keep_cells]

        # Filter nucdata
        #self.nucdata = self.nucdata.loc[keep_cells]
        self.nucleus_boundaries = self.nucleus_boundaries.loc[keep_cells]

        # Filter clusters
        keep_cells = [name for name in self.cell_names if name in self.clusters.index]
        self.clusters = self.clusters.loc[keep_cells]

        # Filter AnnData
        self.adata = self.adata[keep_cells,:].copy()

        # Filter cell and nucleus boundaries
        #keep_cells_boundaries = [name for name in self.cell_boundaries.index if name in self.celldata.index]
        #self.cell_boundaries = self.cell_boundaries.loc[keep_cells_boundaries]
        #self.nucleus_boundaries = self.nucleus_boundaries.loc[keep_cells_boundaries]

        self.area = roi_polygon.area
        
    def copy(self):
        import copy
        return copy.deepcopy(self)

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
        distance_to_nucleus: float=None):
        """
        Create a binned AnnData object from transcript data.
        Parameters:
        - xdata: XenData object containing transcript data.
        - bin_size: size of the bins for rasterization.
        - exclude_unassigned: Exclude transcripts labeled as 'UNASSIGNED' in 'cell_id' column
        - distance_to_nucleus: 

        Save the binned AnnData object to the xdata object.
        """
        import anndata as ad
        from scipy.sparse import coo_matrix
        print(f'Creating binned AnnData object with bin size of {bin_size}um...')

        df = self.trans

        if exclude_unassigned:
            print('Using only transcripts assigned to cells/nuclei...')
            df = df[df['cell_id'] != 'UNASSIGNED'].copy()
        
        if distance_to_nucleus is not None:
            print(f'Excluding transcripts further than {distance_to_nucleus}um from the nucleus...')
            # Filter out transcripts that are further than distance_to_nucleus from the nucleus
            df = df[df['nucleus_distance'] <= distance_to_nucleus].copy()

        # Determine the bin indices
        df['x_bin'] = (df['x_location'] // bin_size).astype(int)
        df['y_bin'] = (df['y_location'] // bin_size).astype(int)
        
        # Create unique bin identifiers
        df['bin_id'] = list(zip(df['x_bin'], df['y_bin']))
        
        # Pivot table to create sparse matrix
        bin_groups = df.groupby(['bin_id', 'feature_name']).size().reset_index(name='count')
        
        # Convert bin coordinates to a categorical index
        bin_index = {bid: i for i, bid in enumerate(bin_groups['bin_id'].unique())}
        gene_index = {gene: i for i, gene in enumerate(bin_groups['feature_name'].unique())}
        
        # Map to indices
        row = bin_groups['bin_id'].map(bin_index)
        col = bin_groups['feature_name'].map(gene_index)
        data = bin_groups['count'].values
        
        # Create sparse matrix
        expression_matrix = coo_matrix((data, (row, col)), 
            shape=(len(bin_index), len(gene_index)))
        
        # Convert bin_index back to spatial coordinates
        bin_coords = np.array(list(bin_index.keys()))
        ymid = (bin_coords[:,1].max() - bin_coords[:,1].min())/2
        bin_coords[:,1] = -(bin_coords[:,1] - ymid).astype('int')
        # Create AnnData object
        adata = ad.AnnData(X=expression_matrix.tocsr())
        
        # Store metadata
        adata.obs_names = [f'bin_{i}' for i in range(len(bin_index))]
        adata.var_names = list(gene_index.keys())
        adata.obs[['x_bin', 'y_bin']] = bin_coords
        adata.obsm['spatial'] = bin_coords
        
        # Save the new binned AnnData object to the xdata object
        self.binned_adata = adata
        self.binned_adata.uns['bin_size'] = bin_size
        self.binned_adata.uns['pixel_size'] = self.pixel_size
        self.binned_adata.uns['bin_edges'] = np.arange(df['x_location'].min(), df['x_location'].max() + bin_size, bin_size)
        self.binned_adata.uns['bin_edges'] = np.arange(df['y_location'].min(), df['y_location'].max() + bin_size, bin_size)

    def write_ome_tiff(self, genes, output_path, flip_y=False,
                       compression='zlib'):
        """
        Write a multi-layer OME-TIFF image for specified genes from the binned AnnData object.
        Parameters:
        - genes: list of gene names to include in the image.
        - output_path: path to save the OME-TIFF file.
        - flip_y: if True, flip the y-axis of the image. Default is False.
        - compression: compression method for the OME-TIFF file. Default is 'zlib'.

        Note: The binned AnnData object must be created first using create_binned_adata().
        """
        bin_size = self.binned_adata.uns['bin_size']

        if genes is None:
            genes=self.binned_adata.var_names.tolist()
        elif isinstance(genes, str):
            genes = [genes]

        ## make sure the output path is going to write to a folder that exists and that it ends with ome.tiff:
        if not os.path.exists(os.path.dirname(output_path)):
            raise ValueError("Output path does not exist.")
        if not output_path.endswith('.ome.tiff'):
            raise ValueError("Output path must end with .ome.tiff")

        if output_path is None:
            f'{self.name}_multilayer_{bin_size}um.ome.tiff'

        imdata = create_multilayer_image(self, genes)
        if flip_y:
            # Flip the y-axis
            imdata = imdata[::-1, :, :]

        _write_ome_tiff(imdata, 
                       channel_names=genes,
                       compression=compression,
                       physical_size_x=bin_size,
                       physical_size_y=bin_size,
                       output_path=output_path)
    
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
    ):
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
    im = Image.fromarray(np.uint8(imdata))
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

def _write_ome_tiff(image_array, 
                    channel_names, 
                    channel_ids=None, 
                    channel_colors=None,
                   physical_size_x=5, 
                   physical_size_y=5, 
                   significant_bits=12,
                   compression='zlib',
                   output_path=None):
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
    tifffile.imwrite(output_path, 
                     arr,
                     compression=compression,
                     description=ome_xml)

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
    
    expr = data[:, genes].X.todense().sum(1).A1
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
    
    im = Image.fromarray(imdata)

    if return_img:
        return im
    else:
        fig, ax = pl.subplots(figsize=[fig_scale * aspect_ratio, fig_scale])
        ax.imshow(im)
        ax.axis('off')
        pl.show()