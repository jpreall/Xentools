from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/xentools-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/xentools-numba")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def xenium_testdata(repo_root: Path) -> Path:
    path = repo_root / "files" / "xenium_v1_testdata"
    if not path.exists():
        pytest.skip(f"Bundled Xenium test data not found: {path}")
    return path


@pytest.fixture(scope="session")
def xdata(xenium_testdata: Path):
    import xentools

    return xentools.XenData(str(xenium_testdata), verbose=False, lazy_boundaries=True)
