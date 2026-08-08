from __future__ import annotations

import matplotlib.axes
import numpy as np
import pandas as pd


def _ranking():
    genes = ["STABLE", "SECOND", "VARIABLE", "RARE"]
    frame = pd.DataFrame(
        {
            "mean_expression": [8.0, 6.0, 9.0, 0.2],
            "min_detection": [0.9, 0.8, 0.7, 0.05],
            "stability_score": [0.95, 0.85, 0.20, 0.70],
            "eligible": [True, True, True, False],
            "selected": [True, True, False, False],
            "mean_count__core_a": [8.0, 6.0, 2.0, 0.1],
            "mean_count__core_b": [8.2, 5.8, 15.0, 0.2],
            "detection__core_a": [0.95, 0.82, 0.72, 0.05],
            "detection__core_b": [0.90, 0.80, 0.90, 0.08],
        },
        index=genes,
    )
    frame.attrs["min_detection"] = 0.25
    return frame


def test_housekeeping_diagnostics_builds_three_panel_dashboard():
    import xentools

    axes = xentools.plot_housekeeping_diagnostics(_ranking())

    assert axes.shape == (3,)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)
    assert axes[0].get_title() == "Expression–stability tradeoff"
    assert len(axes[1].images) == 1
    assert len(axes[2].collections) >= 2


def test_housekeeping_diagnostics_can_save(tmp_path):
    import xentools

    path = tmp_path / "housekeeping.png"
    xentools.pl.plot_housekeeping_diagnostics(_ranking(), save=path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_housekeeping_diagnostics_rejects_unknown_genes():
    import pytest
    import xentools

    with pytest.raises(KeyError, match="not found"):
        xentools.plot_housekeeping_diagnostics(_ranking(), genes=["NOT_A_GENE"])


def test_housekeeping_diagnostics_omits_zero_count_genes_from_log_scatter():
    import xentools

    ranking = _ranking()
    ranking.loc["ZERO"] = {
        "mean_expression": 0.0,
        "min_detection": 0.0,
        "stability_score": 1.0,
        "eligible": False,
        "selected": False,
        "mean_count__core_a": 0.0,
        "mean_count__core_b": 0.0,
        "detection__core_a": 0.0,
        "detection__core_b": 0.0,
    }
    axes = xentools.plot_housekeeping_diagnostics(ranking)

    plotted_x = np.concatenate(
        [collection.get_offsets()[:, 0] for collection in axes[0].collections]
    )
    assert np.isfinite(plotted_x).all()
    assert plotted_x.min() > -10
    assert "omitted" in axes[0].get_legend().get_texts()[0].get_text()


def test_housekeeping_diagnostics_explains_empty_selection():
    import pytest
    import xentools

    ranking = _ranking()
    ranking["eligible"] = False
    ranking["selected"] = False
    ranking.attrs["expression_cutoff"] = 10.0
    ranking.attrs["min_detection"] = 0.95

    with pytest.raises(ValueError) as error:
        xentools.plot_housekeeping_diagnostics(ranking)

    message = str(error.value)
    assert "4 genes" in message
    assert "pass the expression cutoff" in message
    assert "detection in every group ≥ 95.0%" in message
    assert "lower min_detection or min_expression_quantile" in message
    assert "borderline genes explicitly" in message


def test_housekeeping_diagnostics_explains_empty_explicit_gene_list():
    import pytest
    import xentools

    with pytest.raises(ValueError, match="No genes were provided"):
        xentools.plot_housekeeping_diagnostics(_ranking(), genes=[])
