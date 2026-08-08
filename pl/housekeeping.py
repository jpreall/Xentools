"""Diagnostic plots for housekeeping-gene selection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._save import save_figure

__all__ = ["plot_housekeeping_diagnostics"]


def _validate_ranking(ranking):
    if not isinstance(ranking, pd.DataFrame):
        raise TypeError("ranking must be the DataFrame returned by find_housekeeping_genes().")
    required = {
        "mean_expression",
        "min_detection",
        "stability_score",
        "eligible",
        "selected",
    }
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise ValueError(f"ranking is missing required columns: {', '.join(missing)}.")


def _display_genes(ranking, genes, max_genes):
    if genes is None:
        names = ranking.index[ranking["selected"].astype(bool)].astype(str).tolist()
        if not names:
            names = ranking.index[ranking["eligible"].astype(bool)][:max_genes].astype(str).tolist()
    else:
        names = [str(genes)] if isinstance(genes, str) else [str(g) for g in genes]
        missing = [gene for gene in names if gene not in ranking.index]
        if missing:
            raise KeyError(f"Genes not found in ranking: {', '.join(missing)}.")
    if max_genes is not None:
        names = names[: int(max_genes)]
    if not names:
        raise ValueError(_no_eligible_genes_message(ranking, genes=genes))
    return names


def _no_eligible_genes_message(ranking, *, genes=None):
    if genes is not None:
        return (
            "No genes were provided for the detailed diagnostic panels. "
            "Pass one or more gene names with genes=['GENE1', 'GENE2'], or omit "
            "genes to use those selected by find_housekeeping_genes()."
        )

    n_total = len(ranking)
    n_selected = int(ranking["selected"].astype(bool).sum())
    n_eligible = int(ranking["eligible"].astype(bool).sum())
    expression_cutoff = ranking.attrs.get("expression_cutoff")
    detection_cutoff = ranking.attrs.get("min_detection")

    expression = ranking["mean_expression"].to_numpy(dtype=float)
    detection = ranking["min_detection"].to_numpy(dtype=float)
    expression_pass = (
        np.isfinite(expression)
        if expression_cutoff is None
        else np.isfinite(expression) & (expression >= float(expression_cutoff))
    )
    detection_pass = (
        np.isfinite(detection)
        if detection_cutoff is None
        else np.isfinite(detection) & (detection >= float(detection_cutoff))
    )

    threshold_parts = []
    if expression_cutoff is not None:
        threshold_parts.append(f"mean raw count ≥ {float(expression_cutoff):.4g}")
    if detection_cutoff is not None:
        threshold_parts.append(f"detection in every group ≥ {float(detection_cutoff):.1%}")
    thresholds = " and ".join(threshold_parts) or "the stored eligibility thresholds"

    return (
        "No housekeeping genes are available for the detailed diagnostic panels. "
        f"The ranking contains {n_total} genes: {int(expression_pass.sum())} pass the "
        f"expression cutoff, {int(detection_pass.sum())} pass the detection cutoff, "
        f"{int((expression_pass & detection_pass).sum())} pass both, {n_eligible} are "
        f"marked eligible, and {n_selected} are selected. Current criteria: {thresholds}.\n"
        "Try rerunning find_housekeeping_genes() with a lower min_detection or "
        "min_expression_quantile, confirm that the requested ROIs contain enough "
        "representative cells and that layer points to raw counts, or inspect borderline "
        "genes explicitly with genes=ranking.sort_values('score', "
        "ascending=False).index[:10]."
    )


def plot_housekeeping_diagnostics(
    ranking,
    *,
    genes=None,
    max_genes: int = 20,
    figsize=None,
    cmap: str = "viridis",
    selected_color: str = "#d95f02",
    candidate_color: str = "#8c8c8c",
    rejected_color: str = "#d9d9d9",
    min_plot_mean_count: float = 0.0,
    annotate: bool = True,
    title: Optional[str] = None,
    save=None,
    save_kwargs: Optional[dict] = None,
):
    """
    Plot selection tradeoffs, cross-tissue expression, and detection.

    The three panels answer complementary diagnostic questions:

    1. Are selected genes both sufficiently expressed and unusually stable?
    2. Do their expression levels remain similar across every tissue or ROI?
    3. Are they consistently detected rather than driven by a subset of cells?

    Parameters
    ----------
    ranking
        DataFrame returned by :func:`xentools.find_housekeeping_genes`.
    genes
        Genes to emphasize in the heatmap and detection panel. By default,
        genes marked ``selected`` are used.
    max_genes
        Maximum number of genes in the detailed panels.
    min_plot_mean_count
        Omit genes at or below this raw mean count from the selection scatter.
        The default removes zero-count genes, which are undefined on a log
        scale. Increase this value to hide uninformative very-low-count genes;
        it affects only plotting, not ranking or selection.

    Returns
    -------
    numpy.ndarray
        Three matplotlib axes: selection map, expression heatmap, and detection
        range plot.
    """
    import matplotlib.pyplot as plt

    _validate_ranking(ranking)
    if max_genes is not None and max_genes < 1:
        raise ValueError("max_genes must be at least 1.")
    if min_plot_mean_count < 0:
        raise ValueError("min_plot_mean_count must be non-negative.")
    display_genes = _display_genes(ranking, genes, max_genes)
    mean_cols = [col for col in ranking.columns if str(col).startswith("mean_count__")]
    detection_cols = [col for col in ranking.columns if str(col).startswith("detection__")]
    if not mean_cols or not detection_cols:
        raise ValueError("ranking does not contain per-group expression and detection columns.")

    n_rows = len(display_genes)
    fig_height = max(4.8, 0.32 * n_rows + 2.2)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize or (17, fig_height),
        gridspec_kw={"width_ratios": [1.15, max(1.0, 0.22 * len(mean_cols)), 1.0]},
        constrained_layout=True,
    )

    eligible = ranking["eligible"].astype(bool).to_numpy()
    selected = ranking["selected"].astype(bool).to_numpy()
    mean_expression = ranking["mean_expression"].to_numpy(float)
    shown = np.isfinite(mean_expression) & (mean_expression > min_plot_mean_count)
    omitted = int((~shown).sum())
    x = np.full_like(mean_expression, np.nan, dtype=float)
    x[shown] = np.log10(mean_expression[shown])
    y = 1.0 - ranking["stability_score"].to_numpy(float)
    rejected_label = "Rejected"
    if omitted:
        rejected_label += f" ({omitted} low/zero-count omitted)"
    axes[0].scatter(
        x[shown & ~eligible],
        y[shown & ~eligible],
        s=14,
        c=rejected_color,
        alpha=0.55,
        label=rejected_label,
    )
    axes[0].scatter(
        x[shown & eligible & ~selected],
        y[shown & eligible & ~selected],
        s=18,
        c=candidate_color,
        alpha=0.65,
        label="Eligible",
    )
    axes[0].scatter(
        x[shown & selected],
        y[shown & selected],
        s=34,
        c=selected_color,
        edgecolor="white",
        linewidth=0.5,
        label="Selected",
    )
    if annotate:
        for gene in display_genes:
            row = ranking.loc[gene]
            if not np.isfinite(row["mean_expression"]) or row["mean_expression"] <= min_plot_mean_count:
                continue
            axes[0].annotate(
                gene,
                (
                    np.log10(float(row["mean_expression"])),
                    1.0 - float(row["stability_score"]),
                ),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_xlabel("Mean expression (log10 raw counts)")
    axes[0].set_ylabel("Instability score (lower is more stable)")
    axes[0].set_title("Expression–stability tradeoff")
    axes[0].margins(x=0.12, y=0.08)
    axes[0].legend(frameon=False, fontsize=8)

    detail = ranking.loc[display_genes]
    expression = np.log1p(detail[mean_cols].to_numpy(dtype=float))
    group_names = [str(col).split("mean_count__", 1)[1] for col in mean_cols]
    image = axes[1].imshow(expression, aspect="auto", cmap=cmap)
    axes[1].set_xticks(np.arange(len(group_names)))
    axes[1].set_xticklabels(group_names, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(n_rows))
    axes[1].set_yticklabels(display_genes)
    axes[1].set_xlabel("Tissue / ROI")
    axes[1].set_title("Mean expression by tissue")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="log1p mean raw count")

    detection = detail[detection_cols].to_numpy(dtype=float)
    y_pos = np.arange(n_rows)
    axes[2].hlines(
        y_pos,
        detection.min(axis=1),
        detection.max(axis=1),
        color=candidate_color,
        linewidth=2,
        alpha=0.75,
    )
    for j, col in enumerate(detection_cols):
        label = str(col).split("detection__", 1)[1]
        axes[2].scatter(detection[:, j], y_pos, s=24, alpha=0.8, label=label)
    threshold = ranking.attrs.get("min_detection")
    if threshold is not None:
        axes[2].axvline(
            float(threshold), color="#b2182b", linestyle="--", linewidth=1, label="Minimum threshold"
        )
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(display_genes)
    axes[2].set_xlabel("Fraction of cells detected")
    axes[2].set_title("Detection consistency")
    axes[2].legend(frameon=False, fontsize=7, loc="best")

    for ax in (axes[1], axes[2]):
        ax.set_ylim(n_rows - 0.5, -0.5)
    fig.suptitle(title or "Housekeeping-gene diagnostics", fontsize=14)
    save_figure(axes[0], save=save, save_kwargs=save_kwargs)
    return axes
