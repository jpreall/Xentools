from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def _housekeeping_adata():
    rng = np.random.default_rng(12)
    n = 80
    groups = np.repeat(["core_a", "core_b"], n // 2)
    stable = rng.poisson(8, n)
    unstable = np.r_[rng.poisson(1, n // 2), rng.poisson(18, n // 2)]
    rare = rng.binomial(1, 0.05, n) * 20
    noisy = rng.negative_binomial(1, 0.12, n)
    counts = np.column_stack([stable, unstable, rare, noisy])
    adata = AnnData(counts)
    adata.var_names = ["STABLE", "TISSUE_SPECIFIC", "RARE", "NOISY"]
    adata.obs["tissue"] = pd.Categorical(groups)
    return adata


def test_find_housekeeping_genes_ranks_stable_gene():
    from xentools.analysis import find_housekeeping_genes

    result = find_housekeeping_genes(
        _housekeeping_adata(),
        groupby="tissue",
        n_genes=1,
        min_cells=10,
        min_detection=0.5,
        min_expression_quantile=0,
    )

    assert result.index[0] == "STABLE"
    assert result.loc["STABLE", "selected"]
    assert not result.loc["RARE", "eligible"]
    assert result.loc["TISSUE_SPECIFIC", "between_group_log_variance"] > result.loc[
        "STABLE", "between_group_log_variance"
    ]
    assert {"mean_count__core_a", "detection__core_b"}.issubset(result.columns)
    assert result.attrs["normalization"] == "none (raw counts)"


def test_find_housekeeping_genes_accepts_roi_objects():
    from shapely.geometry import box

    from xentools.analysis import find_housekeeping_genes
    from xentools.core.rois import ROI

    adata = _housekeeping_adata()
    adata.obsm["spatial"] = np.column_stack([np.arange(adata.n_obs), np.zeros(adata.n_obs)])

    class Container:
        pass

    data = Container()
    data.adata = adata
    left = ROI.from_geometry(box(-1, -1, 39.5, 1), name="left")
    right = ROI.from_geometry(box(39.5, -1, 81, 1), name="right")
    result = find_housekeeping_genes(
        data,
        rois=[left, right],
        min_cells=10,
        min_detection=0.5,
        min_expression_quantile=0,
    )
    assert result.attrs["groups"] == ["left", "right"]


def test_find_housekeeping_genes_accepts_named_roi_groups():
    from shapely.geometry import box

    from xentools.analysis import find_housekeeping_genes
    from xentools.core.rois import ROI, ROICollection

    adata = _housekeeping_adata()
    adata.obsm["spatial"] = np.column_stack([np.arange(adata.n_obs), np.zeros(adata.n_obs)])

    class Container:
        pass

    data = Container()
    data.adata = adata
    data.rois = ROICollection()
    for name, bounds in {
        "a_left": (-1, 19.5),
        "a_right": (19.5, 39.5),
        "b_left": (39.5, 59.5),
        "b_right": (59.5, 81),
    }.items():
        data.rois.add(ROI.from_geometry(box(bounds[0], -1, bounds[1], 1), name=name))
    data.rois.create_group("core_a", ["a_left", "a_right"])
    data.rois.create_group("core_b", ["b_left", "b_right"])

    result = find_housekeeping_genes(
        data,
        rois=["core_a", "core_b"],
        min_cells=10,
        min_expression_quantile=0,
    )
    assert result.attrs["groups"] == ["core_a", "core_b"]


def test_find_housekeeping_validates_small_groups():
    import pytest

    from xentools.analysis import find_housekeeping_genes

    with pytest.raises(ValueError, match="fewer than min_cells"):
        find_housekeeping_genes(
            _housekeeping_adata(), groupby="tissue", min_cells=50
        )


def test_housekeeping_statistics_are_not_compositional():
    from xentools.analysis import find_housekeeping_genes

    original = _housekeeping_adata()
    changed = original.copy()
    changed.X[:, changed.var_names.get_loc("NOISY")] *= 100

    first = find_housekeeping_genes(
        original, groupby="tissue", min_cells=10, min_expression_quantile=0
    )
    second = find_housekeeping_genes(
        changed, groupby="tissue", min_cells=10, min_expression_quantile=0
    )

    for column in [
        "mean_expression",
        "within_group_dispersion",
        "between_group_log_variance",
    ]:
        assert first.loc["STABLE", column] == second.loc["STABLE", column]


def test_housekeeping_rejects_normalized_values():
    import pytest

    from xentools.analysis import find_housekeeping_genes

    adata = _housekeeping_adata()
    adata.X = adata.X.astype(float) / 3
    with pytest.raises(ValueError, match="raw counts"):
        find_housekeeping_genes(adata, groupby="tissue", min_cells=10)


def test_housekeeping_auto_uses_raw_count_layer_when_x_is_normalized():
    from xentools.analysis import find_housekeeping_genes

    adata = _housekeeping_adata()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X.astype(float) / 3

    result = find_housekeeping_genes(adata, groupby="tissue", min_cells=10)

    assert result.attrs["count_source"] == "adata.layers['counts']"
    assert result.index[0] == "STABLE"


def test_housekeeping_none_requires_counts_in_x_even_when_layer_exists():
    import pytest

    from xentools.analysis import find_housekeeping_genes

    adata = _housekeeping_adata()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X.astype(float) / 3
    with pytest.raises(ValueError, match="layer='auto'"):
        find_housekeeping_genes(adata, groupby="tissue", min_cells=10, layer=None)
