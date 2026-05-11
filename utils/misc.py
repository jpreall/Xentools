"""General-purpose utility helpers."""

from __future__ import annotations

import json

__all__ = ["read_json"]


def read_json(json_file):
    """
    Read a JSON file and return its contents as a Python object.
    """
    with open(json_file) as f:
        return json.load(f)
