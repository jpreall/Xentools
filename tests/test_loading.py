from __future__ import annotations

import numpy as np


def test_public_plotting_namespace_is_xentools_pl():
    import xentools

    assert xentools.pl.__name__ != "matplotlib.pyplot"
    assert hasattr(xentools.pl, "splat")
    assert hasattr(xentools.pl, "plot_boundaries")


def test_xendata_loads_bundled_xenium_testdata(xdata):
    assert xdata.n_transcripts > 0
    assert xdata.adata is not None
    assert xdata.adata.n_obs > 0
    assert xdata.adata.n_vars > 0
    assert "DAPI" in xdata.images

    frame = np.asarray(xdata.frame)
    assert frame.shape == (2, 2)
    assert frame[0, 0] < frame[0, 1]
    assert frame[1, 0] < frame[1, 1]


def test_xendata_loads_cell_boundaries(xdata):
    assert xdata.cell_boundaries is not None
    assert len(xdata.cell_boundaries) > 0
    assert xdata.nucleus_boundaries is not None
    assert len(xdata.nucleus_boundaries) > 0


def test_xendata_defaults_to_lazy_boundaries(xenium_testdata):
    import xentools

    xdata = xentools.XenData(str(xenium_testdata), verbose=False)

    assert hasattr(xdata.cell_boundaries, "loaded")
    assert hasattr(xdata.nucleus_boundaries, "loaded")
    assert not xdata.cell_boundaries.loaded
    assert not xdata.nucleus_boundaries.loaded


def test_cell_feature_matrix_excludes_total_transcripts_feature(xdata):
    assert "Total transcripts" not in xdata.adata.var_names
    assert "total_transcripts" in xdata.adata.obs.columns


def test_cell_feature_matrix_populates_raw_count_metrics(xdata):
    counts = xdata.adata.layers["counts"] if "counts" in xdata.adata.layers else xdata.adata.X

    if "counts" in xdata.adata.layers:
        np.testing.assert_array_equal(xdata.adata.X.toarray(), xdata.adata.layers["counts"].toarray())

    np.testing.assert_array_equal(
        xdata.adata.obs["n_transcripts"].to_numpy(),
        np.asarray(counts.sum(axis=1)).ravel().astype(np.int64),
    )
    np.testing.assert_array_equal(
        xdata.adata.var["total_counts"].to_numpy(),
        np.asarray(counts.sum(axis=0)).ravel().astype(np.int64),
    )
    np.testing.assert_array_equal(
        xdata.adata.var["n_cells"].to_numpy(),
        np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64),
    )


def test_read_zarr_adata_supports_legacy_flat_cell_feature_matrix(tmp_path):
    import warnings

    import zarr
    from scipy import sparse
    from zarr.storage import ZipStore

    from xentools.io.read.zarr import _open_zarr_group_compat, _read_zarr_adata

    def create_array(group, name, data):
        data = np.asarray(data)
        chunks = data.shape if data.ndim > 0 else None
        if hasattr(group, "create_array"):
            return group.create_array(name, data=data, chunks=chunks)
        return group.create_dataset(name, data=data, chunks=chunks)

    cell_ids = np.array([[1, 0], [2, 0]], dtype=np.uint32)
    dense = np.array(
        [
            [1, 0, 5],
            [0, 3, 7],
        ],
        dtype=np.uint32,
    )
    csc = sparse.csc_matrix(dense)

    cfm_store = ZipStore(str(tmp_path / "cell_feature_matrix.zarr.zip"), mode="w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = _open_zarr_group_compat(zarr, cfm_store, mode="w", force_v2=True)
            cf = root.create_group("cell_features")
            create_array(cf, "cell_id", cell_ids)
            create_array(cf, "data", csc.data.astype(np.uint32))
            create_array(cf, "indices", csc.indices.astype(np.uint32))
            create_array(cf, "indptr", csc.indptr.astype(np.uint32))
            cf.attrs.update(
                {
                    "feature_ids": ["gene-a", "gene-b", "total"],
                    "feature_keys": ["GeneA", "GeneB", "Total transcripts"],
                    "feature_types": ["gene", "gene", "aggregate_gene"],
                    "number_cells": 2,
                    "number_features": 3,
                }
            )
    finally:
        cfm_store.close()

    cells_store = ZipStore(str(tmp_path / "cells.zarr.zip"), mode="w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = _open_zarr_group_compat(zarr, cells_store, mode="w", force_v2=True)
            create_array(root, "cell_id", cell_ids)
            create_array(
                root,
                "cell_summary",
                np.array(
                    [
                        [10.0, 20.0, 30.0, 11.0, 21.0, 12.0],
                        [40.0, 50.0, 60.0, 41.0, 51.0, 22.0],
                    ]
                ),
            )
    finally:
        cells_store.close()

    adata = _read_zarr_adata(str(tmp_path), verbose=False)

    assert adata.shape == (2, 2)
    assert adata.var_names.tolist() == ["GeneA", "GeneB"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[1, 0], [0, 3]]))
    np.testing.assert_array_equal(adata.obs["total_transcripts"].to_numpy(), np.array([5, 7]))
    np.testing.assert_array_equal(adata.obs["n_transcripts"].to_numpy(), np.array([1, 3]))
    np.testing.assert_array_equal(adata.var["total_counts"].to_numpy(), np.array([1, 3]))
    np.testing.assert_array_equal(adata.var["n_cells"].to_numpy(), np.array([1, 1]))
    np.testing.assert_allclose(adata.obsm["spatial"], np.array([[10.0, 20.0], [40.0, 50.0]], dtype=np.float32))
