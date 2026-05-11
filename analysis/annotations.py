"""Annotation utilities for XenData-like objects."""

from __future__ import annotations

import pandas as pd

__all__ = ["import_cell_annotations"]


def import_cell_annotations(xdata, cell_annotations_file):
    """
    Import a one-column cell annotation CSV into ``xdata.adata.obs``.

    The CSV index should contain cell IDs. The first column is merged into
    ``xdata.adata.obs`` when it is not already present.
    """
    anno = pd.read_csv(cell_annotations_file, index_col=0)
    groups_col = anno.columns[0]
    if groups_col not in xdata.adata.obs.columns:
        xdata.adata.obs = xdata.adata.obs.merge(
            anno[groups_col],
            left_index=True,
            right_index=True,
            how="left",
        )
