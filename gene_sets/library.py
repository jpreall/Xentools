"""Gene-set library helpers for xentools.

The bundled curated library uses a compact HDF5 layout:

``/vocab``
    Global sorted vocabulary of unique gene symbols.
``/index``
    Names of available source databases/libraries.
``/<database>/names``
    Gene-set names for one database.
``/<database>/gene_ids`` and ``/<database>/offsets``
    CSR-style integer encoding of each gene set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd

CURATED_H5_NAME = "xentools_curated_gene_sets.h5"


def curated_gene_sets_path() -> Path:
    """Return the path to the bundled curated gene-set HDF5 archive."""
    return Path(__file__).resolve().parents[1] / "files" / "gene_lists" / CURATED_H5_NAME


def parse_enrichr_txt(path: str | Path) -> dict[str, list[str]]:
    """
    Parse one Enrichr-style tab-delimited text file.

    Enrichr libraries are commonly distributed as one gene set per line:
    ``set_name<TAB>description<TAB>GENE1<TAB>GENE2...``. Some libraries store
    tokens as ``GENE,score``; the score suffix is stripped.
    """
    path = Path(path)
    gene_sets: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = []
            seen = set()
            for token in parts[2:]:
                gene = token.strip().split(",")[0]
                if gene and gene not in seen:
                    genes.append(gene)
                    seen.add(gene)
            if name and genes:
                gene_sets[name] = genes
    return gene_sets


def build_gene_set_h5(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "*.txt",
    compression: str = "gzip",
    compression_opts: int = 4,
    overwrite: bool = False,
) -> Path:
    """
    Build an HDF5 gene-set archive from Enrichr-style ``.txt`` files.

    Parameters
    ----------
    input_dir
        Directory containing one or more Enrichr-style tab-delimited files.
        Each file becomes one database, named by the file stem.
    output_path
        Destination ``.h5`` path.
    pattern
        Filename glob used to discover input files. Default ``"*.txt"``.
    compression, compression_opts
        HDF5 compression settings for datasets.
    overwrite
        If ``False`` and ``output_path`` exists, raise ``FileExistsError``.

    Returns
    -------
    pathlib.Path
        Path to the written HDF5 archive.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Pass overwrite=True to replace it.")

    txt_files = sorted(input_dir.glob(pattern))
    if not txt_files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {input_dir}")

    all_data: list[tuple[str, dict[str, list[str]]]] = []
    vocab_set: set[str] = set()
    for txt in txt_files:
        gene_sets = parse_enrichr_txt(txt)
        if not gene_sets:
            continue
        all_data.append((txt.stem, gene_sets))
        for genes in gene_sets.values():
            vocab_set.update(genes)

    if not all_data:
        raise ValueError(f"No valid gene sets found in {input_dir}")

    vocab = np.array(sorted(vocab_set), dtype=object)
    gene_to_idx = {gene: i for i, gene in enumerate(vocab)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["description"] = "xentools gene-set library; vocab+CSR encoding"
        h5.attrs["format"] = "xentools_gene_sets_h5"
        h5.attrs["format_version"] = "1"
        h5.attrs["n_databases"] = len(all_data)
        h5.attrs["n_vocab"] = len(vocab)
        h5.attrs["source_pattern"] = pattern

        h5.create_dataset(
            "vocab",
            data=vocab,
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=compression,
            compression_opts=compression_opts,
        )

        db_names = []
        for db_name, gene_sets in all_data:
            db_names.append(db_name)
            names = list(gene_sets.keys())
            gene_lists = [gene_sets[name] for name in names]
            flat_ids = np.array(
                [gene_to_idx[gene] for genes in gene_lists for gene in genes],
                dtype=np.uint32,
            )
            lengths = np.array([len(genes) for genes in gene_lists], dtype=np.uint64)
            offsets = np.zeros(len(lengths) + 1, dtype=np.uint64)
            offsets[1:] = np.cumsum(lengths)

            grp = h5.create_group(db_name)
            grp.create_dataset(
                "names",
                data=np.array(names, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression=compression,
                compression_opts=compression_opts,
            )
            grp.create_dataset(
                "gene_ids",
                data=flat_ids,
                compression=compression,
                compression_opts=compression_opts,
            )
            grp.create_dataset(
                "offsets",
                data=offsets,
                compression=compression,
                compression_opts=compression_opts,
            )
            grp.attrs["n_sets"] = len(names)
            grp.attrs["n_genes_total"] = int(flat_ids.size)
            grp.attrs["source_file"] = f"{db_name}.txt"

        h5.create_dataset(
            "index",
            data=np.array(db_names, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    return output_path


class GeneSetLibrary:
    """
    Lightweight reader for xentools gene-set HDF5 archives.

    Parameters
    ----------
    path
        Path to a gene-set HDF5 archive. If omitted, the bundled curated
        Xentools library is used.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = curated_gene_sets_path() if path is None else Path(path)
        self._h5 = h5py.File(self.path, "r")
        self._vocab: Optional[np.ndarray] = None
        self._names_cache: dict[str, np.ndarray] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Close the underlying HDF5 file handle."""
        self._h5.close()

    @staticmethod
    def _decode(arr) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype.kind == "O" and arr.size and isinstance(arr.flat[0], bytes):
            return np.array([x.decode("utf-8") for x in arr], dtype=object)
        if arr.dtype.kind == "S":
            return arr.astype(str)
        return arr

    @property
    def databases(self) -> list[str]:
        """Available source database/library names."""
        return [str(x) for x in self._decode(self._h5["index"][:])]

    @property
    def vocab(self) -> np.ndarray:
        """Global gene-symbol vocabulary for the archive."""
        if self._vocab is None:
            self._vocab = self._decode(self._h5["vocab"][:])
        return self._vocab

    def _names(self, database: str) -> np.ndarray:
        if database not in self._names_cache:
            self._names_cache[database] = self._decode(self._h5[database]["names"][:])
        return self._names_cache[database]

    def list_sets(self, database: str | None = None) -> pd.DataFrame:
        """
        List available gene sets.

        If ``database`` is supplied, only that source library is listed.
        """
        databases = [database] if database is not None else self.databases
        rows = []
        for db in databases:
            names = self._names(db)
            offsets = self._h5[db]["offsets"][:]
            lengths = np.diff(offsets)
            rows.extend(
                {"database": db, "gene_set": str(name), "n_genes": int(n)}
                for name, n in zip(names, lengths)
            )
        return pd.DataFrame(rows)

    def search(
        self,
        query: str,
        *,
        database: str | None = None,
        case: bool = False,
    ) -> pd.DataFrame:
        """Search gene-set names and return matching rows from ``list_sets``."""
        table = self.list_sets(database)
        if table.empty:
            return table
        pattern = query if case else query.lower()
        names = table["gene_set"] if case else table["gene_set"].str.lower()
        return table[names.str.contains(pattern, regex=False)].reset_index(drop=True)

    def get(self, database: str, gene_set: str) -> list[str]:
        """Return one gene set as a list of gene symbols."""
        grp = self._h5[database]
        names = self._names(database)
        matches = np.where(names == gene_set)[0]
        if len(matches) == 0:
            raise KeyError(f"{gene_set!r} not found in {database!r}")
        i = int(matches[0])
        offsets = grp["offsets"]
        start, end = int(offsets[i]), int(offsets[i + 1])
        ids = grp["gene_ids"][start:end]
        return [str(gene) for gene in self.vocab[ids]]

    def get_many(self, database: str, gene_sets: Iterable[str]) -> dict[str, list[str]]:
        """Return multiple gene sets from one database."""
        return {name: self.get(database, name) for name in gene_sets}

    def get_all(self, database: str) -> dict[str, list[str]]:
        """Return all gene sets from one database as ``{name: [genes]}``."""
        return self.get_many(database, self._names(database))

    def info(self) -> dict:
        """Return lightweight archive metadata."""
        return {
            "path": str(self.path),
            "databases": len(self.databases),
            "gene_sets": int(sum(self._h5[db].attrs["n_sets"] for db in self.databases)),
            "vocab_size": int(self._h5.attrs.get("n_vocab", len(self.vocab))),
            "format": self._h5.attrs.get("format", "unknown"),
            "format_version": self._h5.attrs.get("format_version", "unknown"),
        }

    def __repr__(self):
        info = self.info()
        return (
            "GeneSetLibrary("
            f"{info['databases']} databases, "
            f"{info['gene_sets']:,} gene sets, "
            f"{info['vocab_size']:,} genes)"
        )


def load_curated_gene_sets() -> GeneSetLibrary:
    """Open the bundled curated Xentools gene-set library."""
    return GeneSetLibrary(curated_gene_sets_path())
