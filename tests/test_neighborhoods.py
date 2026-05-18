from __future__ import annotations

import numpy as np


def _roi_around_cells(xdata, n=20, pad=20):
    import xentools

    coords = np.asarray(xdata.adata.obsm["spatial"][:n], dtype=float)
    xmin, ymin = coords.min(axis=0) - pad
    xmax, ymax = coords.max(axis=0) + pad
    return xentools.ROI.from_bounds(xmin, xmax, ymin, ymax, name="test_roi")


def test_neighborhood_composition_defaults_to_active_roi(xenium_testdata):
    import xentools

    xdata = xentools.XenData(str(xenium_testdata), verbose=False)
    roi = _roi_around_cells(xdata)
    xdata.set_active_roi(roi)

    out = xdata.neighborhood_composition(label_key="Cluster", n_neighbors=3)
    coords = np.asarray(xdata.adata.obsm["spatial"], dtype=float)
    expected_mask = roi.contains_points(coords[:, 0], coords[:, 1])

    assert len(out) == int(expected_mask.sum())
    assert out.index.equals(xdata.adata.obs_names[expected_mask])
    assert "_neighbor_total" in out.columns
    positive = out["_neighbor_total"] > 0
    np.testing.assert_allclose(
        out.loc[positive, out.columns != "_neighbor_total"].sum(axis=1).to_numpy(),
        np.ones(int(positive.sum())),
    )


def test_neighborhood_composition_can_ignore_active_roi(xenium_testdata):
    import xentools

    xdata = xentools.XenData(str(xenium_testdata), verbose=False)
    xdata.set_active_roi(_roi_around_cells(xdata))

    out = xentools.neighborhood_composition(
        xdata,
        label_key="Cluster",
        roi=None,
        n_neighbors=3,
        normalize="count",
    )

    assert len(out) == xdata.adata.n_obs
    assert out.index.equals(xdata.adata.obs_names)
    assert (out["_neighbor_total"] >= 0).all()


def test_neighborhood_composition_accepts_named_roi(xenium_testdata):
    import xentools

    xdata = xentools.XenData(str(xenium_testdata), verbose=False)
    roi = _roi_around_cells(xdata)
    xdata.ROIs.add(roi)

    out = xentools.analysis.neighborhood_composition(
        xdata,
        label_key="Cluster",
        roi="test_roi",
        n_neighbors=3,
    )

    assert len(out) > 0
    assert out.attrs["roi"] == "test_roi"
