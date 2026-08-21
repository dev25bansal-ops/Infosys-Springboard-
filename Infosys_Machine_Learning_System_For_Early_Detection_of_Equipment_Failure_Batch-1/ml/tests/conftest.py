"""Hermetic test setup (TST-02).

All tests must resolve repo-relative paths from THIS file's location, never from
the process CWD, so the suite passes identically whether pytest runs from ml/,
the repo root, or CI.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def pipeline_config():
    """Absolute path to configs/pipeline.yml (CWD-independent)."""
    return REPO_ROOT / "configs" / "pipeline.yml"


@pytest.fixture
def repo_root():
    """Absolute repo root (CWD-independent), for data paths in tests."""
    return REPO_ROOT
