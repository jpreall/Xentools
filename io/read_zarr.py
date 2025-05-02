import zarr
import numpy as np

def read_transcripts_zarr_do_ddf(transcrips_zarr):
    """
    Read transcript density from a Zarr store.

    Parameters
    ----------
    zzfile : str
        Path to the Zarr store (ZIP-compressed).

    Returns
    -------
    ddf : dask.dataframe.DataFrame
        Dask DataFrame containing the transcript density.
    """
    import dask.array as da
    import dask.bag as db
    import dask.dataframe as dd

    # 1. Open the ZIP-compressed Zarr store
    store = zarr.ZipStore(transcrips_zarr, mode='r')
    root = zarr.open_group(store, mode='r')

    # 2. Inspect the hierarchy (optional)
    root.tree()  

    # 3. Point to the CSR group
    csr = root['density']['gene']

    # 4. Wrap the on‑disk arrays as lazy Dask arrays
    data    = da.from_array(csr['data'],    chunks=csr['data'].chunks)
    indices = da.from_array(csr['indices'], chunks=csr['indices'].chunks)
    indptr  = da.from_array(csr['indptr'],  chunks=csr['indptr'].chunks)

    # 5. Grab grid dimensions from the attributes
    rows = csr.attrs['rows']
    cols = csr.attrs['cols']
    # (each row index i encodes (gene = i // rows, grid_row = i % rows))

    # 6. Build a Dask bag that expands each CSR row into a list of records
    def extract_records(i):
        start, end = int(indptr[i]), int(indptr[i+1])
        gene      = i // rows
        grid_row  = i % rows
        # note: slicing a Dask array here will return a NumPy-backed chunk
        cols_i    = indices[start:end].compute()
        counts_i  = data[start:end].compute()
        return [
            {'gene':    int(gene),
            'grid_row':int(grid_row),
            'grid_col':int(c),
            'count':   int(cnt)}
        for c, cnt in zip(cols_i, counts_i)
        ]

    nrows = len(csr['indptr']) - 1
    bag   = db.from_sequence(range(nrows), npartitions=100)
    bag   = bag.map(extract_records).flatten()

    # 7. Convert to a Dask DataFrame
    ddf = bag.to_dataframe()
    return ddf

def retrieve_gene_density_zarr(store_path, gene_name,):
    """
    Retrieve transcript density for a given gene in Xenium transcripts.zarr.zip.
    
    Parameters
    ----------
    store_path : str
        Path to your transcripts.zarr.zip file.
    gene_name : str
        Name of the gene you want to plot (must be one of csr.attrs['gene_names']).

    Returns
    -------
    density : np.ndarray
        2D array of shape (rows, cols) containing the binned counts.
    """
    # 1. Open the ZIP‐compressed Zarr store
    store = zarr.ZipStore(store_path, mode='r')
    root = zarr.open_group(store, mode='r')

    # 2. Access the CSR‐encoded density for genes
    csr = root['density']['gene']
    # list of gene names  [oai_citation_attribution:0‡10x Genomics](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/3.1/advanced/xoa-output-zarr)
    gene_names = csr.attrs['gene_names']            
    try:
        gene_idx = gene_names.index(gene_name)
    except ValueError:
        raise ValueError(f"Gene name '{gene_name}' not found; available genes: {gene_names}")

    # 3. Load CSR arrays into memory
    data    = csr['data'][:]       # nonzero counts
    indices = csr['indices'][:]    # column indices of each count
    indptr  = csr['indptr'][:]     # row pointers into data/indices

    # 4. Grid dimensions from attributes
    rows = csr.attrs['rows']
    cols = csr.attrs['cols']

    # 5. Allocate the dense image
    density = np.zeros((rows, cols), dtype=data.dtype)

    # 6. Fill in counts for each grid row of the selected gene
    #    rows of CSR matrix are ordered as (gene0,row0), (gene0,row1), …, (gene1,row0), …
    base = gene_idx * rows
    for r in range(rows):
        i = base + r
        start, end = indptr[i], indptr[i+1]
        density[r, indices[start:end]] = data[start:end]

    return density