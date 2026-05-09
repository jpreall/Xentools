"""Shared metadata helpers for xentools."""

from __future__ import annotations

import numpy as np

__all__ = ["_encode_xenium_cell_ids"]


def _encode_xenium_cell_ids(prefix_arr, suffix_arr):
    """Convert raw cell_id columns to Xenium hash strings (e.g. 'aaaabbbb-12345')."""
    prefix_arr = prefix_arr.astype(np.uint32)
    suffix_arr = suffix_arr.astype(np.uint32)
    hex_strs = np.char.zfill(np.vectorize(lambda x: format(x, "x"))(prefix_arr), 8)
    trans = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    shifted = np.char.translate(hex_strs, trans)
    return np.char.add(np.char.add(shifted, "-"), suffix_arr.astype(str))
