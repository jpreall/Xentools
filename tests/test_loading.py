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


def test_cell_feature_matrix_excludes_total_transcripts_feature(xdata):
    assert "Total transcripts" not in xdata.adata.var_names
    assert "total_transcripts" in xdata.adata.obs.columns
