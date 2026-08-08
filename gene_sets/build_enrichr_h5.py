#!/usr/bin/env python3
"""
Build a compact HDF5 archive of all Enrichr gene set libraries.

Storage design — vocabulary + CSR integer encoding:
  /vocab                     — sorted array of ALL unique gene symbols (global)
  /index                     — string array of database names
  /<database>/
      names                  — string array of gene set names     (N,)
      gene_ids               — flat uint32 array, all gene indices concatenated
      offsets                — uint64 array (N+1,), CSR row-pointer into gene_ids
      attrs: n_sets, n_genes_total, source_file

This mirrors AnnData's CSR format and compresses very well — repeated gene
symbols across thousands of sets are encoded once in /vocab.

Usage:
  python build_enrichr_h5.py               # writes enrichr.h5 in same folder
  python build_enrichr_h5.py --out /path/to/output.h5

Quick access (importable):
  from build_enrichr_h5 import EnrichrH5
  with EnrichrH5("enrichr.h5") as db:
      genes = db.get("MSigDB_Hallmark_2020", "HALLMARK_HYPOXIA")
      all_sets = db.get_all("KEGG_2021_Human")
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ENRICHR_DIR = Path(__file__).parent
DEFAULT_OUT = ENRICHR_DIR / "enrichr.h5"
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_enrichr_file(path: Path) -> dict[str, list[str]]:
    """
    Parse one Enrichr .txt into {gene_set_name: [gene, ...]}.

    Handles two Enrichr formats:
      plain:  name \\t [blank] \\t GENE1 \\t GENE2 ...
      scored: name \\t [blank] \\t GENE1,score \\t GENE2,score ...
    """
    genesets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            name = parts[0]
            # Field 1 is an optional score/description — skip it.
            # Each subsequent field is either "GENE" or "GENE,score"
            genes = []
            for tok in parts[2:]:
                tok = tok.strip()
                if not tok:
                    continue
                gene = tok.split(",")[0]  # strip trailing ,score if present
                if gene:
                    genes.append(gene)
            if genes:
                genesets[name] = genes
    return genesets


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(out_path: Path):
    txt_files = sorted(ENRICHR_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit("No .txt files found in " + str(ENRICHR_DIR))

    print(f"Pass 1/2 — building global gene vocabulary from {len(txt_files)} files...")
    t0 = time.time()

    # Collect all parsed data in memory (modest: gene names are short strings)
    all_data: list[tuple[str, dict[str, list[str]]]] = []
    vocab_set: set[str] = set()
    for txt in txt_files:
        genesets = parse_enrichr_file(txt)
        all_data.append((txt.stem, genesets))
        for genes in genesets.values():
            vocab_set.update(genes)

    vocab = np.array(sorted(vocab_set), dtype=object)  # sorted for reproducibility
    gene2idx = {g: i for i, g in enumerate(vocab)}
    print(f"  {len(vocab):,} unique gene symbols  ({time.time()-t0:.1f}s)")

    str_dtype = h5py.string_dtype()
    uint32 = np.uint32
    uint64 = np.uint64

    print(f"\nPass 2/2 — writing {out_path.name} ...")
    with h5py.File(out_path, "w") as h5:
        h5.attrs["description"] = "Enrichr gene set libraries — vocab+CSR encoding"
        h5.attrs["n_databases"] = len(txt_files)
        h5.attrs["n_vocab"] = len(vocab)
        h5.attrs["built"] = time.strftime("%Y-%m-%d")

        # Global gene vocabulary — use variable-len utf-8 strings
        h5.create_dataset(
            "vocab",
            data=vocab,
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
        )

        db_names = []
        for db_name, genesets in all_data:
            db_names.append(db_name)
            names = list(genesets.keys())
            gene_lists = [genesets[n] for n in names]

            # Build CSR arrays
            flat_ids = np.array(
                [gene2idx[g] for gl in gene_lists for g in gl], dtype=uint32
            )
            lengths = np.array([len(gl) for gl in gene_lists], dtype=uint64)
            offsets = np.zeros(len(lengths) + 1, dtype=uint64)
            offsets[1:] = np.cumsum(lengths)

            grp = h5.create_group(db_name)
            grp.create_dataset(
                "names",
                data=np.array(names, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "gene_ids",
                data=flat_ids,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "offsets",
                data=offsets,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
            )
            grp.attrs["n_sets"] = len(names)
            grp.attrs["n_genes_total"] = int(flat_ids.size)
            grp.attrs["source_file"] = db_name + ".txt"
            print(f"  {db_name}: {len(names):,} sets, {flat_ids.size:,} gene entries")

        # Top-level index
        h5.create_dataset(
            "index",
            data=np.array(db_names, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {time.time()-t0:.1f}s  →  {size_mb:.1f} MB  ({out_path})")


# ---------------------------------------------------------------------------
# Read interface
# ---------------------------------------------------------------------------

class EnrichrH5:
    """
    Lightweight read interface for enrichr.h5.

    Examples
    --------
    with EnrichrH5("enrichr.h5") as db:
        print(db.databases)
        genes = db.get("MSigDB_Hallmark_2020", "HALLMARK_HYPOXIA")
        all_sets = db.get_all("KEGG_2021_Human")   # {name: [genes]}
        hits = db.find_gene("TP53", databases=["KEGG_2021_Human"])
    """

    def __init__(self, path=DEFAULT_OUT):
        self._path = Path(path)
        self._h5 = h5py.File(self._path, "r")
        # Cache vocab as numpy array; decode bytes → str on demand
        self._vocab: np.ndarray | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._h5.close()

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _decode(arr: np.ndarray) -> np.ndarray:
        """Decode bytes array to str (h5py returns bytes for utf-8 datasets)."""
        if arr.dtype.kind == "O" and len(arr) and isinstance(arr.flat[0], bytes):
            return np.array([x.decode("utf-8") for x in arr])
        if arr.dtype.kind == "S":
            return arr.astype(str)
        return arr

    def _vocab_arr(self) -> np.ndarray:
        if self._vocab is None:
            self._vocab = self._decode(self._h5["vocab"][:])
        return self._vocab

    def _decode_names(self, grp: h5py.Group) -> np.ndarray:
        return self._decode(grp["names"][:])

    def _genes_for_row(self, grp: h5py.Group, i: int) -> list[str]:
        offsets = grp["offsets"]
        start, end = int(offsets[i]), int(offsets[i + 1])
        ids = grp["gene_ids"][start:end]
        vocab = self._vocab_arr()
        return [str(g) for g in vocab[ids]]

    # -- public API ----------------------------------------------------------

    @property
    def databases(self) -> list[str]:
        return [str(d) for d in self._decode(self._h5["index"][:])]

    def get(self, database: str, geneset: str) -> list[str]:
        """Return gene list for one named gene set."""
        grp = self._h5[database]
        names = self._decode_names(grp)
        idx = np.where(names == geneset)[0]
        if len(idx) == 0:
            raise KeyError(f"{geneset!r} not found in {database!r}")
        return self._genes_for_row(grp, int(idx[0]))

    def get_all(self, database: str) -> dict[str, list[str]]:
        """Return all gene sets in a database as {name: [genes]}."""
        grp = self._h5[database]
        names = self._decode_names(grp)
        offsets = grp["offsets"][:]
        gene_ids = grp["gene_ids"][:]
        vocab = self._vocab_arr()
        result = {}
        for i, name in enumerate(names):
            ids = gene_ids[offsets[i]: offsets[i + 1]]
            result[str(name)] = [str(g) for g in vocab[ids]]
        return result

    def search_databases(self, pattern: str) -> list[str]:
        """Return database names containing pattern (case-insensitive)."""
        p = pattern.lower()
        return [d for d in self.databases if p in d.lower()]

    def find_gene(self, gene: str, databases=None) -> dict[str, list[str]]:
        """
        Return {database/geneset: [genes]} for every gene set containing gene.
        Optionally restrict to a subset of databases.
        """
        vocab = self._vocab_arr()
        hits = np.where(vocab == gene)[0]
        if len(hits) == 0:
            return {}
        gene_idx = int(hits[0])

        results = {}
        for db in (databases or self.databases):
            grp = self._h5[db]
            gene_ids = grp["gene_ids"][:]
            offsets = grp["offsets"][:]
            names = self._decode_names(grp)
            rows = np.where(gene_ids == gene_idx)[0]  # positions in flat array
            # Map flat positions back to row indices via searchsorted
            row_ids = np.searchsorted(offsets[1:], rows, side="right")
            for row in np.unique(row_ids):
                ids = gene_ids[offsets[row]: offsets[row + 1]]
                results[f"{db}/{str(names[row])}"] = [str(g) for g in vocab[ids]]
        return results

    def info(self) -> dict:
        return {
            "databases": len(self.databases),
            "vocab_size": int(self._h5.attrs.get("n_vocab", 0)),
            "built": self._h5.attrs.get("built", "unknown"),
            "path": str(self._path),
        }


def needs_rebuild(out_path: Path) -> bool:
    """Return True if any .txt is newer than the existing .h5, or .h5 is missing."""
    if not out_path.exists():
        return True
    h5_mtime = out_path.stat().st_mtime
    txt_files = list(ENRICHR_DIR.glob("*.txt"))
    if not txt_files:
        return False
    newest_txt = max(f.stat().st_mtime for f in txt_files)
    return newest_txt > h5_mtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output .h5 path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Skip build if enrichr.h5 is already up-to-date with all .txt files",
    )
    args = parser.parse_args()
    out = Path(args.out)
    if args.check and not needs_rebuild(out):
        print(f"enrichr.h5 is up-to-date — nothing to do. (pass without --check to force rebuild)")
    else:
        build(out)
