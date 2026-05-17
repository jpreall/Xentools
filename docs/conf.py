"""Sphinx configuration for xentools documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

project = "xentools"
author = "Jonathan Preall"
copyright = "2026, Jonathan Preall"

release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "xentools documentation"

# Avoid import-time cache writes in constrained ReadTheDocs or CI environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/xentools-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/xentools-numba")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
