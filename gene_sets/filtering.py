"""Filtering and preparation helpers for gene sets."""

from __future__ import annotations

from typing import Iterable, Literal

import pandas as pd

from .orthologs import convert_gene_sets_species


def infer_species_from_genes(genes: Iterable[str]) -> Literal["human", "mouse", "unknown"]:
    """
    Infer broad human/mouse symbol style from gene names.

    This is intentionally conservative and only uses symbol casing. Explicit
    species arguments should be preferred when ambiguity matters.
    """
    human_like = 0
    mouse_like = 0
    for gene in genes:
        gene = str(gene)
        letters = [c for c in gene if c.isalpha()]
        if len(letters) < 2:
            continue
        if gene == gene.upper():
            human_like += 1
        elif gene[0].isupper() and any(c.islower() for c in gene[1:]):
            mouse_like += 1

    if human_like == 0 and mouse_like == 0:
        return "unknown"
    if mouse_like > human_like * 1.5:
        return "mouse"
    if human_like > mouse_like * 1.5:
        return "human"
    return "unknown"


def infer_xendata_species(xdata) -> Literal["human", "mouse", "unknown"]:
    """Infer species from ``xdata.adata.var_names``."""
    if not hasattr(xdata, "adata") or xdata.adata is None:
        return "unknown"
    return infer_species_from_genes(xdata.adata.var_names)


def infer_database_species(database: str, genes: Iterable[str] | None = None) -> Literal["human", "mouse", "unknown"]:
    """Infer gene-set source species from a database name, falling back to genes."""
    db = database.lower()
    if any(token in db for token in ("mouse", "muris")):
        return "mouse"
    if any(token in db for token in ("human", "sapiens")):
        return "human"
    if genes is not None:
        return infer_species_from_genes(genes)
    return "unknown"


def filter_to_data(
    gene_sets: dict[str, list[str]],
    data_genes: Iterable[str],
    *,
    min_genes: int = 1,
) -> dict[str, list[str]]:
    """Filter gene sets to genes present in ``data_genes``."""
    data_symbols = [str(g) for g in data_genes]
    data = frozenset(data_symbols)
    upper_to_symbols: dict[str, list[str]] = {}
    for symbol in data_symbols:
        upper_to_symbols.setdefault(symbol.upper(), []).append(symbol)

    out = {}
    for name, genes in gene_sets.items():
        kept = []
        seen = set()
        for gene in genes:
            gene = str(gene)
            if gene in data:
                matched = gene
            else:
                matches = upper_to_symbols.get(gene.upper(), [])
                matched = matches[0] if len(matches) == 1 else None
            if matched is not None and matched not in seen:
                kept.append(matched)
                seen.add(matched)
        if len(kept) >= min_genes:
            out[name] = kept
    return out


def coverage_summary(
    original_sets: dict[str, list[str]],
    filtered_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize how many genes remain after filtering."""
    rows = []
    for name, genes in original_sets.items():
        n_before = len(genes)
        n_after = len(filtered_sets.get(name, []))
        rows.append(
            {
                "gene_set": name,
                "n_before": n_before,
                "n_after": n_after,
                "fraction_retained": n_after / n_before if n_before else 0.0,
                "retained": name in filtered_sets,
            }
        )
    columns = ["gene_set", "n_before", "n_after", "fraction_retained", "retained"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["retained", "fraction_retained", "n_after"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def prepare_for_xendata(
    gene_sets: dict[str, list[str]],
    xdata,
    *,
    source_species: Literal["human", "mouse", "auto", "unknown"] = "auto",
    target_species: Literal["human", "mouse", "auto", "unknown"] = "auto",
    ortholog_strategy: Literal["all", "one_to_one", "best"] = "best",
    mouse_case: Literal["mgi", "upper", "lower"] | None = "mgi",
    min_genes: int = 1,
    cache_path=None,
    ortholog_mapping: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Convert species when needed and filter gene sets to ``xdata`` features.

    ``source_species="auto"`` infers source species from the input gene symbols.
    ``target_species="auto"`` infers target species from ``xdata.adata.var_names``.
    """
    if not hasattr(xdata, "adata") or xdata.adata is None:
        raise ValueError("xdata must have an AnnData object at xdata.adata.")

    data_genes = xdata.adata.var_names
    resolved_target = infer_xendata_species(xdata) if target_species == "auto" else target_species
    all_source_genes = [gene for genes in gene_sets.values() for gene in genes]
    resolved_source = infer_species_from_genes(all_source_genes) if source_species == "auto" else source_species

    prepared = gene_sets
    if (
        resolved_source in {"human", "mouse"}
        and resolved_target in {"human", "mouse"}
        and resolved_source != resolved_target
    ):
        prepared = convert_gene_sets_species(
            gene_sets,
            source_species=resolved_source,
            target_species=resolved_target,
            strategy=ortholog_strategy,
            mouse_case=mouse_case,
            min_genes=1,
            cache_path=cache_path,
            ortholog_mapping=ortholog_mapping,
        )

    return filter_to_data(prepared, data_genes, min_genes=min_genes)
