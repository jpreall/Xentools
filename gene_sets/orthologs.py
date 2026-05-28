"""Ortholog utilities for gene-set preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import urllib.request

import pandas as pd

_MGI_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
_HUMAN_TAXON = 9606
_MOUSE_TAXON = 10090


def _default_cache_path() -> Path:
    return Path.home() / ".cache" / "xentools" / "HOM_MouseHumanSequence.rpt"


def fetch_ortholog_table(
    cache_path: str | Path | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Return the MGI human-mouse ortholog table.

    The table is downloaded on first use and cached under
    ``~/.cache/xentools`` by default.
    """
    path = _default_cache_path() if cache_path is None else Path(cache_path)
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MGI_URL, path)
    return pd.read_csv(path, sep="\t")


def _ortholog_pairs(cache_path: str | Path | None = None) -> pd.DataFrame:
    df = fetch_ortholog_table(cache_path)
    df.columns = [c.strip() for c in df.columns]
    taxon_col = next(c for c in df.columns if "Taxon" in c)
    symbol_col = next(c for c in df.columns if c == "Symbol")
    group_col = next((c for c in df.columns if "Class Key" in c or "HomoloGene" in c), None)
    if group_col is None:
        raise ValueError(f"Could not find ortholog group column. Columns: {list(df.columns)}")

    human = (
        df[df[taxon_col] == _HUMAN_TAXON][[group_col, symbol_col]]
        .rename(columns={group_col: "ortholog_group", symbol_col: "human"})
    )
    mouse = (
        df[df[taxon_col] == _MOUSE_TAXON][[group_col, symbol_col]]
        .rename(columns={group_col: "ortholog_group", symbol_col: "mouse"})
    )
    return human.merge(mouse, on="ortholog_group")


def build_ortholog_mapping(
    *,
    source_species: Literal["human", "mouse"],
    target_species: Literal["human", "mouse"],
    strategy: Literal["all", "one_to_one", "best"] = "best",
    mouse_case: Literal["mgi", "upper", "lower"] | None = "mgi",
    cache_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """
    Build a gene-symbol mapping between human and mouse.

    ``strategy`` controls paralog handling:

    ``"all"``
        Keep every known ortholog.
    ``"one_to_one"``
        Keep only groups with exactly one human and one mouse gene.
    ``"best"``
        Use one-to-one mappings where available, then keep all orthologs for
        genes without a one-to-one mapping.
    """
    if source_species == target_species:
        raise ValueError("source_species and target_species must differ.")
    if {source_species, target_species} != {"human", "mouse"}:
        raise ValueError("Only human <-> mouse ortholog conversion is currently supported.")

    pairs = _ortholog_pairs(cache_path)

    if strategy == "one_to_one":
        group_size = pairs.groupby("ortholog_group").size()
        pairs = pairs[pairs["ortholog_group"].isin(group_size[group_size == 1].index)]
    elif strategy == "best":
        group_size = pairs.groupby("ortholog_group").size()
        one_to_one_ids = group_size[group_size == 1].index
        one_to_one = pairs[pairs["ortholog_group"].isin(one_to_one_ids)]
        source_col = source_species
        mapped_sources = set(one_to_one[source_col])
        rest = pairs[~pairs[source_col].isin(mapped_sources)]
        pairs = pd.concat([one_to_one, rest], ignore_index=True)
    elif strategy != "all":
        raise ValueError("strategy must be 'all', 'one_to_one', or 'best'.")

    def _mouse_case(symbol: str) -> str:
        if mouse_case == "upper":
            return symbol.upper()
        if mouse_case == "lower":
            return symbol.lower()
        return symbol

    mapping: dict[str, list[str]] = {}
    for _, row in pairs.iterrows():
        source = str(row[source_species])
        target = str(row[target_species])
        if target_species == "mouse":
            target = _mouse_case(target)
        mapping.setdefault(source, [])
        if target not in mapping[source]:
            mapping[source].append(target)
    return mapping


def convert_gene_sets_species(
    gene_sets: dict[str, list[str]],
    *,
    source_species: Literal["human", "mouse"],
    target_species: Literal["human", "mouse"],
    strategy: Literal["all", "one_to_one", "best"] = "best",
    mouse_case: Literal["mgi", "upper", "lower"] | None = "mgi",
    min_genes: int = 1,
    cache_path: str | Path | None = None,
    ortholog_mapping: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Convert a gene-set dictionary between human and mouse symbols."""
    if source_species == target_species:
        return {name: list(genes) for name, genes in gene_sets.items() if len(genes) >= min_genes}

    mapping = ortholog_mapping
    if mapping is None:
        mapping = build_ortholog_mapping(
            source_species=source_species,
            target_species=target_species,
            strategy=strategy,
            mouse_case=mouse_case,
            cache_path=cache_path,
        )

    converted: dict[str, list[str]] = {}
    for name, genes in gene_sets.items():
        out = []
        seen = set()
        for gene in genes:
            for mapped in mapping.get(str(gene), []):
                if mapped not in seen:
                    out.append(mapped)
                    seen.add(mapped)
        if len(out) >= min_genes:
            converted[name] = out
    return converted


def ortholog_summary(
    source_sets: dict[str, list[str]],
    converted_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize gene-set size changes after ortholog conversion."""
    rows = []
    for name in sorted(set(source_sets) | set(converted_sets)):
        n_source = len(source_sets.get(name, []))
        n_converted = len(converted_sets.get(name, []))
        rows.append(
            {
                "gene_set": name,
                "n_source": n_source,
                "n_converted": n_converted,
                "fraction_converted": n_converted / n_source if n_source else 0.0,
                "n_lost": n_source - n_converted,
            }
        )
    return pd.DataFrame(rows).sort_values("fraction_converted").reset_index(drop=True)
