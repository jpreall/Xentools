"""
Pure helper functions for writing Xenium-compatible outputs.

This module is intentionally standalone — it imports nothing from the parent
xentools package so it can be loaded by file path when needed without
triggering circular imports or package-resolution issues.

The orchestration (calling these helpers in the right order) lives in
`xentools.py` methods such as `XenData.write_xenium_explorer()`.
"""

from __future__ import annotations

import gzip
import io as _io
import importlib.util
import json
import os
import shutil
import sys
import uuid as _uuid_mod
import warnings
import zipfile as _zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "write_xenium_gene_groups",
    "_invert_xen_gene_list_dict",
    "extract_ome_channel_names",
    "_prepare_cells_dataframe",
    "_prepare_boundary_dataframe",
    "_write_10x_mex",
    "_write_experiment_xenium",
    "_write_compressed_mex",
    "_write_analysis_directory",
    "_materialize_transcripts",
    "_decode_xenium_cell_ids",
    "_write_cells_zarr",
    "_write_cfm_zarr",
    "_write_analysis_zarr",
    "_write_transcripts_zarr",
    "write_xenium_explorer",
    "write_geo_submission",
]

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def write_xenium_gene_groups(xen_gene_list_dict, output_file):
    """
    Write a gene-group CSV readable by Xenium Explorer.
    """
    inverted_dict = _invert_xen_gene_list_dict(xen_gene_list_dict)

    with open(output_file, "w", newline="") as f:
        f.write("gene,group\n")
        for gene, groups in inverted_dict.items():
            f.write(f"{gene},{','.join(groups)}\n")


def _invert_xen_gene_list_dict(xen_gene_list_dict):
    """
    Invert `{group_name: [genes...]}` to `{gene: [group_names...]}`.
    """
    from collections import defaultdict

    gene_to_sets = defaultdict(list)
    for set_name, genes in xen_gene_list_dict.items():
        for gene in genes:
            gene_to_sets[gene].append(set_name)
    return gene_to_sets


def _load_sibling_module(module_name, filename):
    """Load a sibling writer module when this file is imported outside a package."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _image_writer_functions():
    try:
        from .images import _write_morphology_for_slice, _write_protein_images_for_slice
    except ImportError:
        images = _load_sibling_module("_xentools_io_write_images_for_xenium", "images.py")
        _write_morphology_for_slice = images._write_morphology_for_slice
        _write_protein_images_for_slice = images._write_protein_images_for_slice
    return _write_morphology_for_slice, _write_protein_images_for_slice


def extract_ome_channel_names(path):
    """
    Extract OME channel names from a TIFF file's OME-XML metadata.
    """
    import tifffile
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata

    root = ET.fromstring(xml)
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    channels = [
        ch.attrib.get("Name")
        for ch in root.findall(".//ome:Image[1]/ome:Pixels/ome:Channel", ns)
    ]

    return pd.Series(channels, name="channel")


def _read_xenium_table_if_present(xenium_folder, filename):
    path = os.path.join(xenium_folder, filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _geometry_to_boundary_rows(cell_id, geometry, label_id):
    rows = []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for poly in polygons:
        coords = np.asarray(poly.exterior.coords)
        for x, y in coords:
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "vertex_x": float(x),
                    "vertex_y": float(y),
                    "label_id": int(label_id),
                }
            )
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


def _roi_bounds_um(roi):
    if hasattr(roi, "shapely_bounds") and hasattr(roi, "bounds"):
        return tuple(map(float, roi.bounds))
    if hasattr(roi, "bounds"):
        bounds = tuple(map(float, roi.bounds))
        if len(bounds) != 4:
            raise ValueError("ROI bounds must contain four values.")
        minx, miny, maxx, maxy = bounds
        return minx, maxx, miny, maxy
    raise TypeError("ROI-like object must expose bounds.")


def _export_crop_origin_um(xdata):
    subset_roi = getattr(xdata, "subset_roi", None)
    if subset_roi is None:
        return 0.0, 0.0
    xmin, _, ymin, _ = _roi_bounds_um(subset_roi)
    x0_px = max(0, int(np.floor(xmin / xdata.pixel_size)))
    y0_px = max(0, int(np.floor(ymin / xdata.pixel_size)))
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


def write_xenium_explorer(
    xdata,
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
    Write a XenData-like object to a Xenium Explorer-compatible bundle.
    """
    outdir = Path(output_dir)
    if outdir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {outdir}\n"
            "Pass overwrite=True to replace it."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    crop_origin = _export_crop_origin_um(xdata) if rebase_coordinates else (0.0, 0.0)

    if verbose:
        print("  Collecting transcripts...", end=" ", flush=True)
    trans_df = _materialize_transcripts(xdata)
    trans_df = _rebase_spatial_dataframe(trans_df, crop_origin)
    n_transcripts = len(trans_df)
    trans_df.to_parquet(outdir / "transcripts.parquet", index=False)
    if verbose:
        print(f"done. ({n_transcripts:,} rows)")

    if verbose:
        print("  Writing cells.parquet...", end=" ", flush=True)
    cells_df = _prepare_cells_dataframe(xdata)
    cells_df = _rebase_spatial_dataframe(cells_df, crop_origin)
    cells_df.to_parquet(outdir / "cells.parquet", index=False)
    if verbose:
        print(f"done. ({len(cells_df):,} cells)")

    if verbose:
        print("  Writing boundary parquets...", end=" ", flush=True)
    cell_bdf = _prepare_boundary_dataframe(xdata, boundary_kind="cell")
    cell_bdf = _rebase_spatial_dataframe(cell_bdf, crop_origin)
    cell_bdf.to_parquet(outdir / "cell_boundaries.parquet", index=False)
    nuc_bdf = _prepare_boundary_dataframe(xdata, boundary_kind="nucleus")
    nuc_bdf = _rebase_spatial_dataframe(nuc_bdf, crop_origin)
    nuc_bdf.to_parquet(outdir / "nucleus_boundaries.parquet", index=False)
    if verbose:
        print("done.")

    if verbose:
        print("  Writing cell_feature_matrix/...", end=" ", flush=True)
    _write_compressed_mex(xdata.adata, outdir)
    if verbose:
        print(f"done. ({xdata.adata.n_obs:,} cells x {xdata.adata.n_vars:,} features)")

    if verbose:
        print("  Writing analysis/...", end=" ", flush=True)
    _write_analysis_directory(xdata, outdir)
    if verbose:
        print("done.")

    panel_src = os.path.join(xdata.xenium_folder, "gene_panel.json")
    if os.path.exists(panel_src):
        shutil.copy2(panel_src, outdir / "gene_panel.json")

    if verbose:
        print("  Writing cells.zarr.zip...", end=" ", flush=True)
    obs_names = xdata.adata.obs_names.astype(str).tolist()
    pixel_size = float(xdata.xenium_metadata.get("pixel_size", 0.2125))
    _write_cells_zarr(obs_names, cells_df, cell_bdf, nuc_bdf, outdir, pixel_size=pixel_size)
    if verbose:
        print("done.")

    if verbose:
        print("  Writing cell_feature_matrix.zarr.zip...", end=" ", flush=True)
    _write_cfm_zarr(xdata.adata, obs_names, outdir)
    if verbose:
        print("done.")

    if verbose:
        print("  Writing analysis.zarr.zip...", end=" ", flush=True)
    _write_analysis_zarr(xdata.adata, obs_names, outdir)
    if verbose:
        print("done.")

    if verbose:
        print("  Writing transcripts.zarr.zip...", end=" ", flush=True)
    source_fov_names = _load_source_fov_names(xdata)
    _write_transcripts_zarr(trans_df, outdir, fov_names=source_fov_names)
    if verbose:
        print(f"done. ({n_transcripts:,} transcripts)")

    xenium_explorer_files = {
        "cells_zarr_filepath": "cells.zarr.zip",
        "cell_features_zarr_filepath": "cell_feature_matrix.zarr.zip",
        "analysis_zarr_filepath": "analysis.zarr.zip",
        "transcripts_zarr_filepath": "transcripts.zarr.zip",
    }

    if verbose:
        print("  Writing experiment.xenium...", end=" ", flush=True)
    _write_experiment_xenium(
        xdata,
        outdir,
        n_transcripts=n_transcripts,
        xenium_explorer_files=xenium_explorer_files,
    )
    if verbose:
        print("done.")

    if include_morphology:
        morph_src = os.path.join(xdata.xenium_folder, "morphology.ome.tif")
        if os.path.exists(morph_src):
            if verbose:
                print("  Writing morphology.ome.tif...", end=" ", flush=True)
            write_morphology, _ = _image_writer_functions()
            write_morphology(
                xdata,
                output_path=outdir / "morphology.ome.tif",
                crop=crop_morphology,
                pyramidal=pyramidal_morphology,
                pyramid_scale=pyramid_scale,
                tile=morphology_tile,
            )
            if verbose:
                print("done.")
        elif verbose:
            print("  morphology.ome.tif not found - skipping.")

    if verbose:
        print(f"\nXenium Explorer bundle written to: {outdir}")


def write_geo_submission(
    xdata,
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
    overwrite: bool = False,
):
    """
    Write a XenData-like slice to GEO-friendly Xenium-style outputs.
    """
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
    if include_protein_images and xdata.protein_images is not None:
        required_outputs.extend(
            outdir / "morphology_focus" / name
            for name in xdata.protein_images["filenames"]
        )

    existing = [path for path in required_outputs if path.exists()]
    if existing and not overwrite:
        existing_str = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output files: {existing_str}")

    crop_origin_um = _export_crop_origin_um(xdata) if rebase_coordinates else (0.0, 0.0)

    transcripts_df = _materialize_transcripts(xdata)
    transcripts_df = _rebase_spatial_dataframe(transcripts_df, crop_origin_um)
    transcripts_df.to_parquet(outdir / "transcripts.parquet", index=False)

    cells_df = _prepare_cells_dataframe(xdata)
    cells_df = _rebase_spatial_dataframe(cells_df, crop_origin_um)
    cells_df.to_parquet(outdir / "cells.parquet", index=False)

    cell_boundary_df = _prepare_boundary_dataframe(xdata, boundary_kind="cell")
    cell_boundary_df = _rebase_spatial_dataframe(cell_boundary_df, crop_origin_um)
    cell_boundary_df.to_parquet(outdir / "cell_boundaries.parquet", index=False)

    nucleus_boundary_df = _prepare_boundary_dataframe(xdata, boundary_kind="nucleus")
    nucleus_boundary_df = _rebase_spatial_dataframe(nucleus_boundary_df, crop_origin_um)
    nucleus_boundary_df.to_parquet(outdir / "nucleus_boundaries.parquet", index=False)

    _write_10x_mex(xdata.adata, outdir)

    write_morphology, write_protein_images = _image_writer_functions()
    if include_morphology:
        write_morphology(
            xdata,
            output_path=outdir / "morphology.ome.tif",
            crop=crop_morphology,
            pyramidal=pyramidal_morphology,
            pyramid_scale=pyramid_scale,
            tile=morphology_tile,
        )

    if include_protein_images and xdata.protein_images is not None:
        write_protein_images(
            xdata,
            output_dir=outdir / "morphology_focus",
            crop=crop_protein_images,
            pyramid_scale=pyramid_scale,
            tile=morphology_tile,
        )



def _dedup_zarr_zip(zip_path):
    """
    Post-process a zarr v3-written zip to be pure zarr v2 compatible.

    zarr v3 injects two zarr-v2-incompatible artifacts:
    1. Duplicate entries: .zgroup/.zattrs written multiple times — keep last.
    2. `dimension_separator` field in .zarray JSON — strip it (not in zarr v2 spec;
       causes Xenium Explorer's strict parser to crash).
    3. Empty .zattrs files added for every array/group — skip them (zarr v2 only
       writes .zattrs where attributes actually exist).
    """
    import json as _json

    zip_path = str(zip_path)
    with _zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        last_idx = {}
        for i, info in enumerate(infos):
            last_idx[info.filename] = i
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w", compression=_zipfile.ZIP_STORED, allowZip64=True) as out:
            for i, info in enumerate(infos):
                if last_idx[info.filename] != i:
                    continue
                with zf.open(info) as src:
                    data = src.read()
                # Strip zarr-v3-only field from array metadata
                if info.filename.endswith(".zarray"):
                    try:
                        meta = _json.loads(data)
                        meta.pop("dimension_separator", None)
                        data = _json.dumps(meta, separators=(",", ":")).encode()
                    except Exception:
                        pass
                # Drop empty .zattrs files (zarr v3 adds them everywhere)
                elif info.filename.endswith(".zattrs"):
                    try:
                        attrs = _json.loads(data)
                        if not attrs:
                            continue
                    except Exception:
                        pass
                out.writestr(info, data)
    with open(zip_path, "wb") as f:
        f.write(buf.getvalue())


def _open_zarr_group_compat(zarr_module, store, mode="r", *, force_v2=False):
    """
    Open a Zarr group across Zarr 2 and 3.

    Zarr 3 accepts ``zarr_format=2`` to force v2 metadata. Zarr 2 rejects that
    keyword but already reads/writes v2 metadata natively.
    """
    kwargs = {"store": store, "mode": mode}
    if force_v2:
        try:
            return zarr_module.open_group(**kwargs, zarr_format=2)
        except TypeError as exc:
            if "zarr_format" not in str(exc):
                raise
    return zarr_module.open_group(**kwargs)


def _cluster_col_to_dir_name(col: str) -> str:
    """Map an adata.obs column name to Explorer's analysis subdirectory name."""
    if col == "Cluster":
        return "gene_expression_graphclust"
    return f"gene_expression_{col}"


def _get_var_column(adata, candidates, default=None):
    for c in candidates:
        if c in adata.var.columns:
            return adata.var[c].astype(str).to_numpy()
    return np.asarray(default) if default is not None else adata.var_names.astype(str).to_numpy()


def _is_lazy_transcripts(trans) -> bool:
    """Duck-type check for LazyTranscripts — avoids importing xentools."""
    return hasattr(trans, "_tile_meta") and hasattr(trans, "query")


def _decode_xenium_cell_ids(hash_strings):
    """
    Convert Xenium hash cell ID strings back to (prefix_uint32, suffix_uint32).

    Inverse of _encode_xenium_cell_ids: 'aaaabbbb-1' → (prefix, 1).
    """
    trans = str.maketrans("abcdefghijklmnop", "0123456789abcdef")
    prefix_list, suffix_list = [], []
    for s in hash_strings:
        hash_part, suffix_str = s.rsplit("-", 1)
        prefix_list.append(int(hash_part.translate(trans), 16))
        suffix_list.append(int(suffix_str))
    return np.array(prefix_list, dtype=np.uint32), np.array(suffix_list, dtype=np.uint32)


# ---------------------------------------------------------------------------
# experiment.xenium
# ---------------------------------------------------------------------------

def _write_experiment_xenium(
    xdata,
    output_dir: Path,
    n_transcripts: int | None = None,
    xenium_explorer_files: dict | None = None,
):
    """
    Write experiment.xenium, updating cell count and removing zarr-specific paths.
    """
    meta = dict(xdata.xenium_metadata)
    meta["num_cells"] = int(xdata.adata.n_obs)
    if n_transcripts is not None:
        meta["num_transcripts"] = int(n_transcripts)

    meta.pop("xenium_explorer_files", None)
    if xenium_explorer_files:
        meta["xenium_explorer_files"] = xenium_explorer_files

    images = dict(meta.get("images", {}))
    images["morphology_filepath"] = "morphology.ome.tif"
    images.pop("morphology_focus_filepath", None)
    meta["images"] = images

    (output_dir / "experiment.xenium").write_text(json.dumps(meta, indent=4))


# ---------------------------------------------------------------------------
# Compressed MEX (cell_feature_matrix/)
# ---------------------------------------------------------------------------

def _write_compressed_mex(adata, output_dir: Path):
    """Write gzip-compressed barcodes / features / matrix to cell_feature_matrix/."""
    mex_dir = output_dir / "cell_feature_matrix"
    mex_dir.mkdir(exist_ok=True)

    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    mat_coo = matrix.T.tocoo()   # genes × cells
    n_genes, n_cells = mat_coo.shape

    # Zarr-format feature types use short names; map to 10x MEX canonical strings
    # so scanpy.read_10x_mtx(gex_only=True) keeps them on round-trip reads.
    _TYPE_MAP = {
        "gene":                       "Gene Expression",
        "negative_control_probe":     "Negative Control Probe",
        "negative_control_codeword":  "Negative Control Codeword",
        "unassigned_codeword":        "Unassigned Codeword",
        "aggregate_gene":             "Gene Expression",
    }

    barcodes = adata.obs_names.astype(str)
    gene_ids = _get_var_column(adata, ("gene_ids", "id", "gene_id"), default=adata.var_names)
    raw_types = _get_var_column(
        adata, ("feature_types", "feature_type"),
        default=["Gene Expression"] * adata.n_vars,
    )
    feat_types = np.array([_TYPE_MAP.get(t, t) for t in raw_types])

    with gzip.open(mex_dir / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(barcodes) + "\n")

    with gzip.open(mex_dir / "features.tsv.gz", "wt") as f:
        for gid, gname, ftype in zip(gene_ids, adata.var_names, feat_types):
            f.write(f"{gid}\t{gname}\t{ftype}\n")

    # Header then data via numpy (avoids Python-level loop over millions of entries)
    mtx_data = np.column_stack([mat_coo.row + 1, mat_coo.col + 1, mat_coo.data.astype(int)])
    with gzip.open(mex_dir / "matrix.mtx.gz", "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write('%metadata_json: {"software_version": "xentools", "format_version": 2}\n')
        f.write(f"{n_genes} {n_cells} {mat_coo.nnz}\n")
        np.savetxt(f, mtx_data, fmt="%d", delimiter=" ")


# ---------------------------------------------------------------------------
# Transcript materialization
# ---------------------------------------------------------------------------

_TRANSCRIPT_DEFAULTS = {
    "transcript_id": 0,
    "cell_id": "UNASSIGNED",
    "overlaps_nucleus": 0,
    "z_location": 0.0,
    "qv": 0.0,
}


def _materialize_transcripts(xdata) -> pd.DataFrame:
    """
    Return a transcript DataFrame for the current (possibly subsetted) XenData.

    Priority
    --------
    1. Original ``transcripts.parquet`` in xenium_folder, filtered to current cells
       + spatially filtered UNASSIGNED transcripts (avoids loading off-ROI data).
    2. ``LazyTranscripts.query()`` over the active spatial frame (zarr datasets).
    3. ``xdata.trans`` directly (parquet-format datasets already in memory).
    """
    keep_cells = set(xdata.adata.obs_names.astype(str))
    parquet_path = os.path.join(xdata.xenium_folder, "transcripts.parquet")

    # Spatial ROI bounds from xdata.frame if available (µm coordinates)
    spatial_bounds = None
    try:
        frame = xdata.frame
        if frame is not None:
            spatial_bounds = (
                float(frame[0, 0]), float(frame[0, 1]),
                float(frame[1, 0]), float(frame[1, 1]),
            )
    except Exception:
        pass

    subset_roi = getattr(xdata, "subset_roi", None)
    roi_geometry = None if subset_roi is None else getattr(subset_roi, "geometry", subset_roi)
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        x_col = "x_location" if "x_location" in df.columns else "x_centroid"
        y_col = "y_location" if "y_location" in df.columns else "y_centroid"

        if roi_geometry is not None and x_col in df.columns and y_col in df.columns:
            from shapely import contains_xy

            roi_mask = contains_xy(
                roi_geometry,
                df[x_col].to_numpy(),
                df[y_col].to_numpy(),
            )
            df = df[roi_mask].copy()

        if "cell_id" in df.columns:
            df["cell_id"] = df["cell_id"].astype(str)
            assigned_mask = df["cell_id"].isin(keep_cells)
            unassigned_mask = df["cell_id"] == "UNASSIGNED"

            if roi_geometry is None and spatial_bounds is not None and unassigned_mask.any():
                xmin, xmax, ymin, ymax = spatial_bounds
                if x_col in df.columns and y_col in df.columns:
                    in_roi = (
                        (df[x_col] >= xmin) & (df[x_col] <= xmax) &
                        (df[y_col] >= ymin) & (df[y_col] <= ymax)
                    )
                    unassigned_mask = unassigned_mask & in_roi

            df = df[assigned_mask | unassigned_mask].copy()
        for col, default in _TRANSCRIPT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        return df

    if _is_lazy_transcripts(xdata.trans):
        frame = xdata.frame
        df = xdata.trans.query(
            xmin=float(frame[0, 0]),
            xmax=float(frame[0, 1]),
            ymin=float(frame[1, 0]),
            ymax=float(frame[1, 1]),
            genes=None,
            quality="all",
        )
        for col, default in _TRANSCRIPT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        return df

    df = xdata.trans.copy()
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)
        df = df[df["cell_id"].isin(keep_cells | {"UNASSIGNED"})].copy()
    return df


def _load_source_fov_names(xdata) -> list[str]:
    """Return the original dataset's fov_names table in source order."""
    zarr_path = os.path.join(xdata.xenium_folder, "transcripts.zarr.zip")
    if os.path.exists(zarr_path):
        try:
            with _zipfile.ZipFile(zarr_path) as zf:
                attrs = json.loads(zf.read(".zattrs").decode())
            names = attrs.get("fov_names")
            if names:
                return [str(x) for x in names]
        except Exception:
            pass

    parquet_path = os.path.join(xdata.xenium_folder, "transcripts.parquet")
    if os.path.exists(parquet_path):
        try:
            cols = ["transcript_id", "fov_name"]
            df = pd.read_parquet(parquet_path, columns=cols)
            if "transcript_id" in df.columns:
                tid_u64 = df["transcript_id"].fillna(0).astype(np.uint64).to_numpy()
                fov_idx = np.maximum(0, (tid_u64 >> np.uint64(32)) - np.uint64(65536)).astype(np.uint32)
                if "fov_name" in df.columns:
                    valid = df["fov_name"].notna().to_numpy()
                    lookup = (
                        pd.DataFrame({
                            "fov_idx": pd.Series(fov_idx[valid], dtype=np.uint32),
                            "fov_name": pd.Series(np.asarray(df.loc[valid, "fov_name"], dtype=object)),
                        })
                        .drop_duplicates(subset=["fov_idx"])
                        .sort_values("fov_idx")
                    )
                    if len(lookup):
                        max_fov = int(lookup["fov_idx"].max())
                        names = [str(i) for i in range(max_fov + 1)]
                        for idx, name in lookup.itertuples(index=False):
                            names[int(idx)] = str(name)
                        return names
        except Exception:
            pass

    return []


# ---------------------------------------------------------------------------
# Analysis / cluster CSVs
# ---------------------------------------------------------------------------

_CLUSTER_COLUMNS = {
    "Cluster",
    "kmeans_2_clusters", "kmeans_3_clusters", "kmeans_4_clusters",
    "kmeans_5_clusters", "kmeans_6_clusters", "kmeans_7_clusters",
    "kmeans_8_clusters", "kmeans_9_clusters", "kmeans_10_clusters",
}


def _write_analysis_directory(xdata, output_dir: Path):
    """
    Write ``analysis/clustering/<grouping>/clusters.csv`` for each cluster column.

    Format: ``Barcode,Cluster`` (Explorer's expected CSV layout).
    """
    obs = xdata.adata.obs
    cluster_cols = [c for c in obs.columns if c in _CLUSTER_COLUMNS]
    if not cluster_cols:
        return

    cluster_root = output_dir / "analysis" / "clustering"
    cluster_root.mkdir(parents=True, exist_ok=True)
    barcodes = xdata.adata.obs_names.astype(str)

    for col in cluster_cols:
        grouping_dir = cluster_root / _cluster_col_to_dir_name(col)
        grouping_dir.mkdir(exist_ok=True)
        labels = obs[col].astype(str).values
        with open(grouping_dir / "clusters.csv", "w") as f:
            f.write("Barcode,Cluster\n")
            for barcode, label in zip(barcodes, labels):
                if label not in ("nan", "None", ""):
                    f.write(f"{barcode},{label}\n")


# ---------------------------------------------------------------------------
# Zarr helper: boundary vertex packing
# ---------------------------------------------------------------------------

def _pack_boundary_vertices(boundaries_df: pd.DataFrame, obs_names, max_vertices: int = 25):
    """
    Pack boundary polygon rows into padded (N, max_vertices*2) float32 arrays.

    Returns
    -------
    vertices : (N, max_vertices*2) float32   interleaved [x0, y0, x1, y1, ...]
    num_vertices : (N,) int32
    """
    n_cells = len(obs_names)
    vertices = np.zeros((n_cells, max_vertices * 2), dtype=np.float32)
    num_vertices = np.zeros(n_cells, dtype=np.int32)

    if boundaries_df is None or boundaries_df.empty:
        return vertices, num_vertices

    x_col = "vertex_x" if "vertex_x" in boundaries_df.columns else boundaries_df.columns[1]
    y_col = "vertex_y" if "vertex_y" in boundaries_df.columns else boundaries_df.columns[2]

    cell_id_to_idx = {cid: i for i, cid in enumerate(obs_names)}

    for cell_id, group in boundaries_df.groupby("cell_id", sort=False):
        idx = cell_id_to_idx.get(str(cell_id))
        if idx is None:
            continue
        verts_xy = group[[x_col, y_col]].values  # (M, 2)
        nv = min(len(verts_xy), max_vertices)
        num_vertices[idx] = nv
        vertices[idx, : nv * 2] = verts_xy[:nv].flatten()  # [x0,y0,x1,y1,...]

    return vertices, num_vertices


# ---------------------------------------------------------------------------
# Zarr writer: cells.zarr.zip
# ---------------------------------------------------------------------------

def _write_cells_zarr(obs_names, cells_df, cell_bdf, nucleus_bdf, output_dir: Path,
                       pixel_size: float = 0.2125):
    """
    Write ``cells.zarr.zip`` containing cell IDs, summary stats, and boundary polygons.

    Parameters
    ----------
    obs_names  : sequence of hash-format cell ID strings (order defines cell positions)
    cells_df   : cells.parquet DataFrame (cell_id, x_centroid, y_centroid, ...)
    cell_bdf   : cell_boundaries DataFrame (cell_id, vertex_x, vertex_y)
    nucleus_bdf: nucleus_boundaries DataFrame (cell_id, vertex_x, vertex_y)
    output_dir : bundle root directory
    pixel_size : µm per pixel (for the masks homogeneous_transform)
    """
    import zarr
    from zarr.storage import ZipStore

    obs_names = list(obs_names)
    n_cells = len(obs_names)

    # --- cell_id (N, 2) uint32 ---
    prefix, suffix = _decode_xenium_cell_ids(obs_names)
    cell_id_arr = np.column_stack([prefix, suffix]).astype(np.uint32)

    # --- cell_summary (N, 8) float64 ---
    # Columns: x_centroid, y_centroid, cell_area, nucleus_x, nucleus_y,
    #          nucleus_area, seg_method_idx, nucleus_count
    cs = np.zeros((n_cells, 8), dtype=np.float64)
    if cells_df is not None and not cells_df.empty:
        cdf = cells_df.set_index("cell_id")
        for col_i, col_name in enumerate(["x_centroid", "y_centroid", "cell_area"]):
            if col_name in cdf.columns:
                cs[:, col_i] = cdf.reindex(obs_names)[col_name].fillna(0.0).values
        # nucleus centroid ≈ cell centroid when not separately available
        cs[:, 3] = cs[:, 0]
        cs[:, 4] = cs[:, 1]
        if "nucleus_area" in cdf.columns:
            cs[:, 5] = cdf.reindex(obs_names)["nucleus_area"].fillna(0.0).values
    cs[:, 6] = 3.0   # segmentation method index (nucleus-expansion default)
    cs[:, 7] = 1.0   # nucleus_count

    # --- boundary vertex arrays ---
    cell_verts, cell_nv = _pack_boundary_vertices(cell_bdf, obs_names)
    nuc_verts, nuc_nv = _pack_boundary_vertices(nucleus_bdf, obs_names)
    cell_idx = np.arange(n_cells, dtype=np.uint32)

    # --- write zarr ---
    store_path = str(output_dir / "cells.zarr.zip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = ZipStore(store_path, mode="w")
        root = _open_zarr_group_compat(zarr, store, mode="w", force_v2=True)
        compressor = _native_blosc_compressor()

        root.create_array("cell_id", data=cell_id_arr, chunks=cell_id_arr.shape, compressor=compressor)
        root.create_array("cell_summary", data=cs, chunks=(n_cells, 1), compressor=compressor)

        ps = root.create_group("polygon_sets")

        ps0 = ps.create_group("0")  # nucleus
        ps0.create_array("cell_index", data=cell_idx, chunks=cell_idx.shape, compressor=compressor)
        ps0.create_array("method", data=np.full(n_cells, 3, dtype=np.uint32), chunks=(n_cells,), compressor=compressor)
        ps0.create_array("num_vertices", data=nuc_nv, chunks=nuc_nv.shape, compressor=compressor)
        ps0.create_array("vertices", data=nuc_verts, chunks=(min(5651, n_cells), 25), compressor=compressor)

        ps1 = ps.create_group("1")  # cell
        ps1.create_array("cell_index", data=cell_idx, chunks=cell_idx.shape, compressor=compressor)
        ps1.create_array("method", data=np.full(n_cells, 2, dtype=np.uint32), chunks=(n_cells,), compressor=compressor)
        ps1.create_array("num_vertices", data=cell_nv, chunks=cell_nv.shape, compressor=compressor)
        ps1.create_array("vertices", data=cell_verts, chunks=(min(5651, n_cells), 25), compressor=compressor)

        # masks/ — required by Explorer; write placeholder 1×1 arrays with correct transform
        masks = root.create_group("masks")
        # Transform maps µm → pixels: scale = 1/pixel_size
        um_to_px = 1.0 / max(pixel_size, 1e-6)
        transform = np.array([
            [um_to_px, 0, 0, 0],
            [0, um_to_px, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        masks.create_array("homogeneous_transform", data=transform, chunks=(4, 4), compressor=compressor)
        # Placeholder 1×1 mask arrays (no actual cell fills; avoids Explorer crash)
        placeholder = np.zeros((1, 1), dtype=np.uint32)
        masks.create_array("0", data=placeholder, chunks=(1, 1), compressor=compressor)
        masks.create_array("1", data=placeholder, chunks=(1, 1), compressor=compressor)

        root.attrs.update({
            "major_version": 6,
            "minor_version": 2,
            "name": "CellSegmentationDataset",
            "number_cells": n_cells,
            "polygon_set_descriptions": [
                "DAPI-based nuclei segmentation",
                "Cell Segmentation",
            ],
            "polygon_set_display_names": ["Nucleus boundaries", "Cell boundaries"],
            "polygon_set_names": ["nucleus", "cell"],
            "segmentation_methods": [
                "Segmented by boundary stain (None)",
                "Segmented by interior stain (None)",
                "Segmented by nucleus expansion of 5.0\u00b5m",
                "Segmented by nuclear stain (DAPI)",
            ],
            "spatial_units": "microns",
        })
        store.close()
    _dedup_zarr_zip(output_dir / "cells.zarr.zip")


# ---------------------------------------------------------------------------
# Zarr writer: cell_feature_matrix.zarr.zip
# ---------------------------------------------------------------------------

def _write_cfm_zarr(adata, obs_names, output_dir: Path):
    """
    Write ``cell_feature_matrix.zarr.zip`` in Xenium's CSR/CSC sparse format.

    Layout
    ------
    cell_features/
      cell_id  (N, 2) uint32
      indptr   (n_features+1,) uint32    CSR row pointers  (features × cells)
      indices  (nnz,) uint32             CSR column indices (cell positions)
      data     (nnz,) uint32             CSR values
      csc/
        indptr   (n_cells+1,) uint32     CSC col pointers (cells × features)
        indices  (nnz,) uint16           CSC row indices  (feature positions)
        data     (nnz,) uint32           CSC values
    """
    import zarr
    from zarr.storage import ZipStore

    obs_names = list(obs_names)
    n_cells = len(obs_names)
    n_features = adata.n_vars

    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)

    # CSR (features × cells): features are rows, cells are columns
    csr_feat = matrix.T.tocsr().astype(np.int32)
    csr_feat.eliminate_zeros()

    # CSC of the same (features × cells) matrix: cells are columns
    # indptr has n_cells+1 entries, indices are feature indices (uint16)
    csc_cell = matrix.T.tocsc().astype(np.int32)
    csc_cell.eliminate_zeros()

    # Build feature metadata from adata.var
    feature_ids = _get_var_column(adata, ("gene_ids", "id", "gene_id"),
                                  default=adata.var_names).tolist()
    feature_keys = adata.var_names.astype(str).tolist()

    _MEX_TO_ZARR = {
        "Gene Expression":          "gene",
        "Negative Control Probe":   "negative_control_probe",
        "Negative Control Codeword":"negative_control_codeword",
        "Unassigned Codeword":      "unassigned_codeword",
    }
    raw_types = _get_var_column(
        adata, ("feature_types", "feature_type"),
        default=["gene"] * n_features,
    ).tolist()
    feature_types = [_MEX_TO_ZARR.get(t, t) for t in raw_types]

    # Decode cell IDs
    prefix, suffix = _decode_xenium_cell_ids(obs_names)
    cell_id_arr = np.column_stack([prefix, suffix]).astype(np.uint32)

    store_path = str(output_dir / "cell_feature_matrix.zarr.zip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = ZipStore(store_path, mode="w")
        root = _open_zarr_group_compat(zarr, store, mode="w", force_v2=True)
        compressor = _native_blosc_compressor()

        cf = root.create_group("cell_features")
        cf.create_array("cell_id", data=cell_id_arr, chunks=cell_id_arr.shape, compressor=compressor)

        # CSR arrays
        indptr_csr = csr_feat.indptr.astype(np.uint32)
        indices_csr = csr_feat.indices.astype(np.uint32)
        data_csr = csr_feat.data.astype(np.uint32)
        cf.create_array("indptr", data=indptr_csr, chunks=indptr_csr.shape, compressor=compressor)
        cf.create_array("indices", data=indices_csr, chunks=indices_csr.shape, compressor=compressor)
        cf.create_array("data", data=data_csr, chunks=data_csr.shape, compressor=compressor)

        # CSC sub-group
        csc_grp = cf.create_group("csc")
        indptr_csc = csc_cell.indptr.astype(np.uint32)
        # Feature indices fit in uint16 for ≤65535 features
        indices_csc = csc_cell.indices.astype(np.uint16)
        data_csc = csc_cell.data.astype(np.uint32)
        csc_grp.create_array("indptr", data=indptr_csc, chunks=indptr_csc.shape, compressor=compressor)
        csc_grp.create_array("indices", data=indices_csc, chunks=indices_csc.shape, compressor=compressor)
        csc_grp.create_array("data", data=data_csc, chunks=data_csc.shape, compressor=compressor)

        cf.attrs.update({
            "feature_ids": feature_ids,
            "feature_keys": feature_keys,
            "feature_types": feature_types,
            "major_version": 4,
            "minor_version": 1,
            "number_cells": n_cells,
            "number_features": n_features,
        })
        store.close()
    _dedup_zarr_zip(output_dir / "cell_feature_matrix.zarr.zip")


# ---------------------------------------------------------------------------
# Zarr writer: analysis.zarr.zip
# ---------------------------------------------------------------------------

def _write_analysis_zarr(adata, obs_names, output_dir: Path):
    """
    Write ``analysis.zarr.zip`` with cell-group cluster assignments.

    Each cluster column in adata.obs is written as one grouping.  Unassigned
    cells (NaN / 'None' / '') are excluded from the indices arrays.

    Layout
    ------
    cell_groups/
      {k}/
        indices  (n_assigned,) uint32   cell positions in obs_names order
        indptr   (n_clusters+1,) uint32
    cell_groups attrs: grouping_names, group_names, major_version, ...
    """
    import zarr
    from zarr.storage import ZipStore

    obs_names = list(obs_names)
    obs = adata.obs

    cluster_cols = [c for c in obs.columns if c in _CLUSTER_COLUMNS]
    if not cluster_cols:
        return

    # Canonical Explorer grouping order
    _ORDER = [
        "Cluster",
        "kmeans_2_clusters", "kmeans_3_clusters", "kmeans_4_clusters",
        "kmeans_5_clusters", "kmeans_6_clusters", "kmeans_7_clusters",
        "kmeans_8_clusters", "kmeans_9_clusters", "kmeans_10_clusters",
    ]
    ordered_cols = [c for c in _ORDER if c in cluster_cols]
    ordered_cols += [c for c in cluster_cols if c not in ordered_cols]

    grouping_names = [_cluster_col_to_dir_name(c) for c in ordered_cols]
    all_group_names = []
    all_indices = []
    all_indptrs = []

    positions = np.arange(len(obs_names), dtype=np.uint32)

    for col in ordered_cols:
        labels = obs[col].astype(str).values

        # Find unique non-empty labels; sort numerically by trailing integer if present
        valid_mask = np.array([lbl not in ("nan", "None", "") for lbl in labels])

        def _sort_key(x):
            # "Cluster 3" → (0, 3, "Cluster 3"); "3" → (0, 3, "3"); else alpha
            parts = x.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return (0, int(parts[1]), x)
            if x.isdigit():
                return (0, int(x), x)
            return (1, 0, x)

        unique_labels = sorted(set(labels[valid_mask]), key=_sort_key)

        # Display names: if label is numeric only, prefix "Cluster"; otherwise use as-is
        group_display = []
        for lbl in unique_labels:
            group_display.append(f"Cluster {lbl}" if lbl.isdigit() else lbl)

        indices_list = []
        indptr_list = [0]
        for lbl in unique_labels:
            mask = (labels == lbl) & valid_mask
            cell_positions = np.sort(positions[mask])
            indices_list.append(cell_positions)
            indptr_list.append(indptr_list[-1] + len(cell_positions))

        all_group_names.append(group_display)
        all_indices.append(
            np.concatenate(indices_list) if indices_list else np.array([], dtype=np.uint32)
        )
        all_indptrs.append(np.array(indptr_list, dtype=np.uint32))

    store_path = str(output_dir / "analysis.zarr.zip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = ZipStore(store_path, mode="w")
        root = _open_zarr_group_compat(zarr, store, mode="w", force_v2=True)
        compressor = _native_blosc_compressor()

        cg = root.create_group("cell_groups")
        for k, (idx_arr, ptr_arr) in enumerate(zip(all_indices, all_indptrs)):
            grp = cg.create_group(str(k))
            grp.create_array("indices", data=idx_arr, chunks=idx_arr.shape, compressor=compressor)
            grp.create_array("indptr", data=ptr_arr, chunks=ptr_arr.shape, compressor=compressor)

        cg.attrs.update({
            "group_names": all_group_names,
            "grouping_names": grouping_names,
            "major_version": 1,
            "minor_version": 1,
            "number_groupings": len(ordered_cols),
        })
        store.close()
    _dedup_zarr_zip(output_dir / "analysis.zarr.zip")


# ---------------------------------------------------------------------------
# Zarr writer: transcripts.zarr.zip
# ---------------------------------------------------------------------------

def _build_gene_index_map(trans_df: pd.DataFrame):
    """Return (gene_names, gene_index_map) from transcript feature_name column.

    Regular genes are sorted alphabetically; NegControlProbe, NegControlCodeword,
    and UnassignedCodeword are appended at the end (in each sub-group's natural sort
    order) to match the ordering Xenium Explorer expects.
    """
    if "feature_name" not in trans_df.columns:
        return [], {}
    all_names = trans_df["feature_name"].dropna().unique().tolist()
    regular, neg_probe, neg_cw, unassigned, other_ctrl = [], [], [], [], []
    for n in all_names:
        if n.startswith("NegControlProbe"):
            neg_probe.append(n)
        elif n.startswith("NegControlCodeword"):
            neg_cw.append(n)
        elif n.startswith("UnassignedCodeword"):
            unassigned.append(n)
        elif n.startswith("NegControl") or n.startswith("Unassigned"):
            other_ctrl.append(n)
        else:
            regular.append(n)
    # NegControlProbe uses reverse sort to match Xenium Explorer's native ordering.
    gene_names = (
        sorted(regular)
        + sorted(neg_probe, reverse=True)
        + sorted(neg_cw)
        + sorted(unassigned)
        + sorted(other_ctrl)
    )
    gene_index_map = {name: i for i, name in enumerate(gene_names)}
    return gene_names, gene_index_map


def _build_tile_gene_offset(gene_ids_hi, gene_ids_lo, n_targets, n_hi):
    """
    Compute gene_offset (n_targets, 4) uint32 for one tile.

    Layout within the tile's arrays (hi transcripts first, then lo):
      [0 .. n_hi)   : high-quality transcripts, sorted by gene
      [n_hi .. N)   : low-quality transcripts, sorted by gene

    gene_offset[k] = [lo_start, lo_end, hi_start, hi_end]

    Uses vectorised np.searchsorted over all gene indices at once.
    """
    gene_offset = np.zeros((n_targets, 4), dtype=np.uint32)
    if n_targets == 0:
        return gene_offset

    all_k = np.arange(n_targets, dtype=np.intp)

    if len(gene_ids_hi) > 0:
        hi_arr = np.asarray(gene_ids_hi)
        gene_offset[:, 2] = np.searchsorted(hi_arr, all_k, side="left")
        gene_offset[:, 3] = np.searchsorted(hi_arr, all_k, side="right")

    if len(gene_ids_lo) > 0:
        lo_arr = np.asarray(gene_ids_lo)
        lo_start = np.searchsorted(lo_arr, all_k, side="left")
        lo_end = np.searchsorted(lo_arr, all_k, side="right")
        has_lo = lo_end > lo_start
        gene_offset[has_lo, 0] = lo_start[has_lo] + n_hi
        gene_offset[has_lo, 1] = lo_end[has_lo] + n_hi
    # else: leave as zeros — native bundles use [0,0] as the sentinel for
    # "no low-quality transcripts for this gene in this tile".

    return gene_offset


_TILE_ARRAY_ATTRS = {
    "location": {
        "column_names": ["x_position", "y_position", "z_position"],
        "column_descriptions": ["X Position", "Y Position", "Z Position"],
    },
    "gene_identity": {
        "column_names": ["gene_call"],
        "column_descriptions": ["Gene Call"],
    },
    "gene_offset": {
        "column_names": ["low_qscore_start", "low_qscore_end", "high_qscore_start", "high_qscore_end"],
        "column_descriptions": [
            "element start index for the gene's low quality clusters",
            "element end index for the gene's low quality clusters (not inclusive)",
            "element start index for the gene's high quality clusters",
            "element end index for the gene's high quality clusters (not inclusive)",
        ],
    },
    "codeword_identity": {
        "column_names": ["codeword1_call", "codeword2_call"],
        "column_descriptions": ["Codeword 1 Call", "Codeword 2 Call"],
    },
    "id": {
        "column_names": ["id", "fov_index"],
        "column_descriptions": ["Rna Identifier", "FOV Index"],
    },
    "uuid": {
        "column_names": ["uuid_w0", "uuid_w1"],
        "column_descriptions": ["Blob Unique Identifier, Word 0", "Blob Unique Identifier, Word 1"],
    },
    "quality_score": {
        "column_names": ["calibrated_codeword_score"],
        "column_descriptions": ["Calibrated Codeword Score"],
    },
    "status": {
        "column_names": ["status"],
        "column_descriptions": ["Rna Status"],
    },
    "valid": {
        "column_names": ["valid"],
        "column_descriptions": ["Valid"],
    },
    "cluster_count": {
        "column_names": ["cluster_count"],
        "column_descriptions": ["Cluster Count"],
    },
}


def _native_blosc_compressor():
    """Return the Blosc/Zstd compressor used by native Xenium Zarr bundles."""
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0)


def _build_gene_category_array(gene_names):
    """
    Build (n_genes, 9) bool array for gene/codeword categories.

    Columns (matching Explorer's gene_category schema):
      0: is_gene, 1: is_negative_control_codeword, 2: is_negative_control_probe,
      3: is_unassigned_codeword, 4: is_predesigned_gene, 5: is_deprecated_codeword,
      6: is_custom_gene, 7: is_genomic_control, 8: is_boosted_gene
    """
    n = len(gene_names)
    cat = np.zeros((n, 9), dtype=bool)
    for i, name in enumerate(gene_names):
        if "NegControlCodeword" in name:
            cat[i, 1] = True
        elif "NegControlProbe" in name:
            cat[i, 2] = True
        elif "UnassignedCodeword" in name:
            cat[i, 3] = True
        else:
            cat[i, 0] = True   # is_gene
            cat[i, 4] = True   # is_predesigned_gene
    return cat


def _build_codeword_category_array(codeword_names):
    """Build (n_codewords, 9) bool array in codeword index order."""
    return _build_gene_category_array(codeword_names)


def _write_transcripts_zarr(
    trans_df: pd.DataFrame,
    output_dir: Path,
    tile_size_um: float = 250.0,
    n_grid_levels: int = 5,
    fov_names: list[str] | None = None,
):
    """
    Write ``transcripts.zarr.zip`` with spatially tiled transcript data.

    Five grid levels are written (grids/0 through grids/4), each at 2× coarser
    resolution than the previous.  Grid 0 is the full-resolution level with all
    nine per-tile arrays.  Grids 1–4 are simplified (location, gene_identity,
    gene_offset, cluster_count) for Explorer's zoom-out density views.

    ``grids/.zattrs`` is written with the tile index metadata Explorer needs
    (grid_size, grid_keys, grid_number_objects, codeword_to_transcript_counts,
    number_objects_per_tile_per_gene) so Explorer can locate tiles.

    Performance notes
    -----------------
    * Transcripts are pre-sorted once by (tile_col, tile_row, is_lo, gene_idx)
      — O(N log N) — then the per-tile loop is O(tile_size).
    * gene_offset is computed with vectorised searchsorted (no Python per-gene loop).
    * The hi/lo quality split uses qv ≥ 20 as the threshold.
    * gene_offset format: [lo_start, lo_end, hi_start, hi_end].
    """
    import zarr
    from zarr.storage import ZipStore

    if trans_df is None or trans_df.empty:
        return

    x_col = "x_location" if "x_location" in trans_df.columns else "x_centroid"
    y_col = "y_location" if "y_location" in trans_df.columns else "y_centroid"
    z_col = "z_location" if "z_location" in trans_df.columns else None
    qv_col = "qv" if "qv" in trans_df.columns else None

    gene_names, gene_index_map = _build_gene_index_map(trans_df)
    n_genes = len(gene_names)
    if n_genes == 0:
        return

    # ---- prepare working columns (avoids repeated Series lookups) -----------
    work = trans_df[[c for c in [x_col, y_col, z_col, qv_col,
                                  "feature_name", "codeword_index",
                                  "transcript_id", "fov_name"]
                     if c and c in trans_df.columns]].copy()

    work["_gene_idx"] = work["feature_name"].map(gene_index_map).fillna(-1).astype(np.int32)
    work = work[work["_gene_idx"] >= 0].reset_index(drop=True)

    # codeword_gene_names: genes sorted by codeword_index (panel detection order).
    # codeword_gene_mapping[i]: alphabetical rank of codeword i in gene_names.
    # Tile-level gene_identity/gene_offset follow alphabetical gene_names order,
    # while codeword_identity and codeword metadata use codeword_index order.
    if "codeword_index" in work.columns:
        cw_pairs = (
            work[["codeword_index", "feature_name"]]
            .drop_duplicates().dropna().copy()
        )
        cw_pairs["codeword_index"] = cw_pairs["codeword_index"].astype(int)
        cw_pairs = cw_pairs.sort_values("codeword_index").reset_index(drop=True)
        n_codewords = int(cw_pairs["codeword_index"].max()) + 1
        codeword_gene_names = [""] * n_codewords
        for cw_idx, feat_name in cw_pairs[["codeword_index", "feature_name"]].itertuples(index=False):
            codeword_gene_names[int(cw_idx)] = feat_name
        codeword_gene_mapping = [gene_index_map.get(name, 0) for name in codeword_gene_names]
        work["_cw_idx"] = work["codeword_index"].fillna(-1).astype(np.int32)
    else:
        codeword_gene_names = gene_names
        n_codewords = n_genes
        codeword_gene_mapping = list(range(n_genes))
        work["_cw_idx"] = work["_gene_idx"]

    if "transcript_id" in work.columns:
        tid_u64 = work["transcript_id"].fillna(0).astype(np.uint64).to_numpy()
        inferred_fov = np.maximum(0, (tid_u64 >> np.uint64(32)) - np.uint64(65536)).astype(np.uint32)
        work["_fov_idx"] = inferred_fov
        if fov_names is not None:
            fov_names = [str(x) for x in fov_names]
        elif "fov_name" in work.columns:
            # Use the inferred numeric FOV index for the tile arrays and build the
            # root fov_names lookup from the transcript table without per-row
            # Python string mapping during export.
            valid = work["fov_name"].notna().to_numpy()
            fov_names_series = pd.Series(np.asarray(work.loc[valid, "fov_name"], dtype=object))
            fov_idx_series = pd.Series(inferred_fov[valid], dtype=np.uint32)
            fov_lookup = (
                pd.DataFrame({"fov_idx": fov_idx_series, "fov_name": fov_names_series})
                .drop_duplicates(subset=["fov_idx"])
                .sort_values("fov_idx")
            )
            max_fov = int(fov_lookup["fov_idx"].max()) if len(fov_lookup) else -1
            fov_names = [str(i) for i in range(max_fov + 1)]
            for idx, name in fov_lookup.itertuples(index=False):
                fov_names[int(idx)] = str(name)
        else:
            max_fov = int(inferred_fov.max()) if len(inferred_fov) else -1
            fov_names = [str(i) for i in range(max_fov + 1)]
    elif "fov_name" in work.columns:
        codes, uniques = pd.factorize(work["fov_name"], sort=True)
        work["_fov_idx"] = np.maximum(codes, 0).astype(np.uint32)
        if fov_names is None:
            fov_names = [str(x) for x in uniques.tolist()]
        else:
            fov_names = [str(x) for x in fov_names]
    else:
        work["_fov_idx"] = np.zeros(len(work), dtype=np.uint32)
        fov_names = [] if fov_names is None else [str(x) for x in fov_names]

    xs = work[x_col].values.astype(np.float32)
    ys = work[y_col].values.astype(np.float32)
    zs = work[z_col].values.astype(np.float32) if z_col and z_col in work.columns else np.zeros(len(work), np.float32)

    x_min, y_min = float(xs.min()), float(ys.min())
    x_max, y_max = float(xs.max()), float(ys.max())

    # hi/lo quality flag (0 = hi, 1 = lo) — sort hi first within a tile
    if qv_col and qv_col in work.columns:
        work["_is_lo"] = (work[qv_col].values < 20.0).astype(np.uint8)
    else:
        work["_is_lo"] = np.zeros(len(work), dtype=np.uint8)

    n_transcripts = len(work)
    dataset_uuid = str(_uuid_mod.uuid4())

    root_attrs = {
        "codeword_count": n_codewords,
        "codeword_gene_mapping": codeword_gene_mapping,
        "codeword_gene_names": codeword_gene_names,
        "coordinate_space": "refined-final_global_micron",
        "data_format": 0,
        "dataset_uuid": dataset_uuid,
        "fov_names": fov_names,
        "gene_index_map": gene_index_map,
        "gene_names": gene_names,
        "major_version": 5,
        "metrics_density_x_count": max(1, int(np.ceil(x_max / 20.0))),
        "metrics_density_x_origin": 0.0,
        "metrics_density_x_spacing": 20.0,
        "metrics_density_y_count": max(1, int(np.ceil(y_max / 20.0))),
        "metrics_density_y_origin": 0.0,
        "metrics_density_y_spacing": 20.0,
        "minor_version": 1,
        "name": "RnaDataset",
        "number_genes": n_genes,
        "number_rnas": n_transcripts,
        "spatial_units": "micron",
    }

    # Per-level accumulation for grids/.zattrs (indexed by codeword_index, not alphabetical rank)
    all_level_keys = [[] for _ in range(n_grid_levels)]
    all_level_counts = [[] for _ in range(n_grid_levels)]
    all_level_hi_per_cw = [np.zeros(n_codewords, dtype=np.int64) for _ in range(n_grid_levels)]
    all_level_lo_per_cw = [np.zeros(n_codewords, dtype=np.int64) for _ in range(n_grid_levels)]

    store_path = str(output_dir / "transcripts.zarr.zip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = ZipStore(store_path, mode="w")
        root = _open_zarr_group_compat(zarr, store, mode="w", force_v2=True)
        compressor = _native_blosc_compressor()
        grids = root.create_group("grids")

        for level in range(n_grid_levels):
            lvl_tile_size = tile_size_um * (2 ** level)

            # Tile indices from absolute origin (0,0) so Explorer can find tiles
            # by computing floor(coord / tile_size) from its own viewport.
            ci = (xs / lvl_tile_size).astype(int)
            ri = (ys / lvl_tile_size).astype(int)
            work["_ci"] = ci
            work["_ri"] = ri

            if level == 0:
                sorted_work = work.sort_values(
                    ["_ci", "_ri", "_is_lo", "_gene_idx"], kind="mergesort"
                )
            else:
                sorted_work = work.sort_values(
                    ["_ci", "_ri", "_gene_idx"], kind="mergesort"
                )

            g = grids.create_group(str(level))

            for (col, row), tile_df in sorted_work.groupby(["_ci", "_ri"], sort=False):
                n_tile = len(tile_df)
                gene_ids = tile_df["_gene_idx"].values.astype(np.int32)
                cw_ids = tile_df["_cw_idx"].values.astype(np.int32)
                loc_xs = tile_df[x_col].values.astype(np.float32)
                loc_ys = tile_df[y_col].values.astype(np.float32)
                loc_zs = tile_df[z_col].values.astype(np.float32) if z_col and z_col in tile_df.columns else np.zeros(n_tile, np.float32)
                location = np.column_stack([loc_xs, loc_ys, loc_zs])

                all_level_keys[level].append(f"{col},{row}")
                all_level_counts[level].append(n_tile)

                if level == 0:
                    n_hi = int((tile_df["_is_lo"].values == 0).sum())
                    hi_gene_ids = gene_ids[:n_hi]
                    lo_gene_ids = gene_ids[n_hi:]
                    # Native bundles include an all-zero sentinel row at the end.
                    gene_offset = _build_tile_gene_offset(hi_gene_ids, lo_gene_ids, n_genes, n_hi)
                    gene_offset = np.vstack([gene_offset, np.zeros((1, 4), dtype=np.uint32)])

                    # Accumulate by codeword_index for grids/.zattrs metadata.
                    hi_cw_ids = cw_ids[:n_hi]
                    lo_cw_ids = cw_ids[n_hi:]
                    valid_hi = hi_cw_ids[(hi_cw_ids >= 0) & (hi_cw_ids < n_codewords)]
                    valid_lo = lo_cw_ids[(lo_cw_ids >= 0) & (lo_cw_ids < n_codewords)]
                    np.add.at(all_level_hi_per_cw[0], valid_hi, 1)
                    np.add.at(all_level_lo_per_cw[0], valid_lo, 1)

                    gene_identity = np.clip(gene_ids, 0, np.iinfo(np.uint16).max).astype(np.uint16).reshape(-1, 1)

                    if "codeword_index" in tile_df.columns:
                        cw_idx_col = tile_df["codeword_index"].fillna(0).astype(np.uint16).values
                    else:
                        cw_idx_col = gene_ids.astype(np.uint16)
                    codeword_identity = np.column_stack([
                        cw_idx_col,
                        np.full(n_tile, np.uint16(65535), dtype=np.uint16),
                    ])

                    if "transcript_id" in tile_df.columns:
                        tid_u64 = tile_df["transcript_id"].fillna(0).astype(np.uint64).to_numpy()
                        tid_lo = (tid_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
                    else:
                        tid_lo = np.zeros(n_tile, dtype=np.uint32)

                    fov_idx = tile_df["_fov_idx"].astype(np.uint32).to_numpy() if "_fov_idx" in tile_df.columns else np.zeros(n_tile, dtype=np.uint32)
                    id_arr = np.column_stack([tid_lo, fov_idx]).astype(np.uint32)
                    uuid_arr = np.column_stack([tid_lo, fov_idx + np.uint32(65536)]).astype(np.uint32)

                    if qv_col and qv_col in tile_df.columns:
                        qv_vals = tile_df[qv_col].fillna(0.0).astype(np.float32).values
                    else:
                        qv_vals = np.zeros(n_tile, np.float32)

                    # BUG 1: per-transcript arrays must be Fortran (column-major) order.
                    # gene_offset is C-order in the original — keep it as-is.
                    _F = np.asfortranarray
                    tile_grp = g.create_group(f"{col},{row}")
                    for arr_name, data, use_f_order in [
                        ("location", location, True),
                        ("gene_identity", gene_identity, True),
                        ("gene_offset", gene_offset, False),
                        ("codeword_identity", codeword_identity, True),
                        ("id", id_arr, True),
                        ("uuid", uuid_arr, True),
                        ("quality_score", qv_vals.reshape(-1, 1), True),
                        ("status", np.zeros((n_tile, 1), np.uint8), True),
                        ("valid", np.ones((n_tile, 1), np.uint8), True),
                    ]:
                        arr_data = _F(data) if use_f_order else data
                        kw = {"order": "F"} if use_f_order else {}
                        tile_grp.create_array(
                            arr_name,
                            data=arr_data,
                            chunks=data.shape,
                            compressor=compressor,
                            **kw,
                        )
                        if arr_name in _TILE_ARRAY_ATTRS:
                            tile_grp[arr_name].attrs.update(_TILE_ARRAY_ATTRS[arr_name])

                else:
                    # Native bundles include an all-zero sentinel row at the end.
                    gene_offset = _build_tile_gene_offset(gene_ids, np.array([], dtype=np.int32), n_genes, n_tile)
                    gene_offset = np.vstack([gene_offset, np.zeros((1, 4), dtype=np.uint32)])
                    gene_identity = np.clip(gene_ids, 0, np.iinfo(np.uint16).max).astype(np.uint16).reshape(-1, 1)
                    cluster_count = np.zeros((n_tile, 1), dtype=np.uint32)

                    valid_cw = cw_ids[(cw_ids >= 0) & (cw_ids < n_codewords)]
                    np.add.at(all_level_hi_per_cw[level], valid_cw, 1)

                    _F = np.asfortranarray
                    tile_grp = g.create_group(f"{col},{row}")
                    for arr_name, data, use_f_order in [
                        ("location", location, True),
                        ("gene_identity", gene_identity, True),
                        ("gene_offset", gene_offset, False),
                        ("cluster_count", cluster_count, True),
                    ]:
                        arr_data = _F(data) if use_f_order else data
                        kw = {"order": "F"} if use_f_order else {}
                        tile_grp.create_array(
                            arr_name,
                            data=arr_data,
                            chunks=data.shape,
                            compressor=compressor,
                            **kw,
                        )
                        if arr_name in _TILE_ARRAY_ATTRS:
                            tile_grp[arr_name].attrs.update(_TILE_ARRAY_ATTRS[arr_name])

        # grids/.zattrs — Explorer reads this to find which tiles exist and their counts
        grids.attrs.update({
            "codeword_to_transcript_counts": (
                all_level_hi_per_cw[0] + all_level_lo_per_cw[0]
            ).tolist(),
            "grid_array_shapes": [[{} for _ in keys] for keys in all_level_keys],
            "grid_key_names": ["grid_x_loc", "grid_y_loc"],
            "grid_keys": all_level_keys,
            "grid_number_objects": all_level_counts,
            "grid_size": [tile_size_um],
            "grid_zip": False,
            "number_levels": n_grid_levels,
            "number_objects_per_tile_per_gene": [
                {
                    "high_qscore": all_level_hi_per_cw[lvl].tolist(),
                    "low_qscore": all_level_lo_per_cw[lvl].tolist(),
                }
                for lvl in range(n_grid_levels)
            ],
        })

        # Top-level category arrays use different index spaces:
        # gene_category -> alphabetical gene_names, codeword_category -> codeword order.
        cat_attrs = {
            "column_names": [
                "is_gene", "is_negative_control_codeword", "is_negative_control_probe",
                "is_unassigned_codeword", "is_predesigned_gene", "is_deprecated_codeword",
                "is_custom_gene", "is_genomic_control", "is_boosted_gene",
            ],
            "column_descriptions": [
                "Codeword is Gene", "Codeword is Negative Control Codeword",
                "Codeword is Negative Control Probe", "Codeword is Unassigned Codeword",
                "Codeword is Predesigned Gene", "Codeword is Deprecated Codeword",
                "Codeword is Custom Gene", "Codeword is Genomic Control Probe",
                "Codeword is Boosted Gene",
            ],
        }
        gene_cat = _build_gene_category_array(gene_names)
        codeword_cat = _build_codeword_category_array(codeword_gene_names)
        root.create_array("gene_category", data=gene_cat, chunks=(n_genes, 1), compressor=compressor)
        root["gene_category"].attrs.update(cat_attrs)
        root.create_array("codeword_category", data=codeword_cat, chunks=(n_codewords, 1), compressor=compressor)
        root["codeword_category"].attrs.update(cat_attrs)

        # density/ group — sparse per-gene density at 10µm resolution
        # Used by Explorer for the "subsampled pyramid" zoomed-out view.
        # Build CSR where rows = (gene_idx * n_y + y_cell), cols = x_cell.
        density_res = 10.0
        n_dx = max(1, int(np.ceil(x_max / density_res)))
        n_dy = max(1, int(np.ceil(y_max / density_res)))
        dxi = np.clip((xs / density_res).astype(np.int32), 0, n_dx - 1)
        dyi = np.clip((ys / density_res).astype(np.int32), 0, n_dy - 1)
        gene_idx_all = work["_gene_idx"].values.astype(np.int32)
        cw_idx_all = work["_cw_idx"].values.astype(np.int32)

        from scipy.sparse import csr_matrix as _csr
        # Row = gene_idx * n_dy + y_cell; col = x_cell; value = count
        row_idx = gene_idx_all * n_dy + dyi
        density_sparse_gene = _csr(
            (np.ones(len(row_idx), dtype=np.uint16), (row_idx, dxi)),
            shape=(n_genes * n_dy, n_dx),
        )
        density_sparse_gene.sum_duplicates()
        density_sparse_gene.sort_indices()

        valid_cw_mask = (cw_idx_all >= 0) & (cw_idx_all < n_codewords)
        density_sparse_codeword = _csr(
            (
                np.ones(int(valid_cw_mask.sum()), dtype=np.uint16),
                (cw_idx_all[valid_cw_mask] * n_dy + dyi[valid_cw_mask], dxi[valid_cw_mask]),
            ),
            shape=(n_codewords * n_dy, n_dx),
        )
        density_sparse_codeword.sum_duplicates()
        density_sparse_codeword.sort_indices()

        density_grp = root.create_group("density")
        density_attrs_gene = {
            "cols": n_dx, "rows": n_dy, "gene_names": gene_names,
            "grid_size": [density_res, density_res], "origin": {"x": 0.0, "y": 0.0},
        }
        # density/codeword uses 'codeword_names' key (not 'gene_names') to match Explorer format
        density_attrs_codeword = {
            "cols": n_dx, "rows": n_dy, "codeword_names": codeword_gene_names,
            "grid_size": [density_res, density_res], "origin": {"x": 0.0, "y": 0.0},
        }
        for sub, density_sparse in (
            ("gene", density_sparse_gene),
            ("codeword", density_sparse_codeword),
        ):
            sg = density_grp.create_group(sub)
            sg.attrs.update(density_attrs_gene if sub == "gene" else density_attrs_codeword)
            data_u16 = density_sparse.data.astype(np.uint16)
            indices_u16 = density_sparse.indices.astype(np.uint16)
            indptr_u32 = density_sparse.indptr.astype(np.uint32)
            sg.create_array("data", data=data_u16, chunks=data_u16.shape, compressor=compressor)
            sg.create_array("indices", data=indices_u16, chunks=indices_u16.shape, compressor=compressor)
            sg.create_array("indptr", data=indptr_u32, chunks=indptr_u32.shape, compressor=compressor)

        # metrics_density — spatial transcript density (n_y_bins, n_x_bins, 4) float32
        # Bin counts at 20µm resolution: channels = [total, hi_qv, lo_qv, unique_genes]
        md_res = 20.0
        n_mdx = max(1, int(np.ceil(x_max / md_res)))
        n_mdy = max(1, int(np.ceil(y_max / md_res)))
        md_xi = np.clip((xs / md_res).astype(np.int32), 0, n_mdx - 1)
        md_yi = np.clip((ys / md_res).astype(np.int32), 0, n_mdy - 1)
        is_lo_all = work["_is_lo"].values
        md = np.zeros((n_mdy, n_mdx, 4), dtype=np.float32)
        np.add.at(md[:, :, 0], (md_yi, md_xi), 1.0)          # total
        np.add.at(md[:, :, 1], (md_yi[is_lo_all == 0], md_xi[is_lo_all == 0]), 1.0)  # hi
        np.add.at(md[:, :, 2], (md_yi[is_lo_all == 1], md_xi[is_lo_all == 1]), 1.0)  # lo
        root.create_array(
            "metrics_density",
            data=md,
            chunks=(min(n_mdy, 91), min(n_mdx, 153), 4),
            compressor=compressor,
        )

        root.attrs.update(root_attrs)
        store.close()
    _dedup_zarr_zip(output_dir / "transcripts.zarr.zip")
