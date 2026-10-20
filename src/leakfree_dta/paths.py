"""Canonical filesystem layout.

Every stage of the pipeline reads from and writes to a directory named here, so a
run can be reproduced by pointing ``LEAKFREE_ROOT`` at a fresh tree and replaying
the CLI commands in order.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repository root, or whatever ``LEAKFREE_ROOT`` points at.

    The environment variable exists so the same code runs unchanged on Kaggle or
    Colab, where the repo is cloned somewhere other than the working directory.
    """
    env = os.environ.get("LEAKFREE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/leakfree_dta/paths.py -> src/leakfree_dta -> src -> repo root
    return Path(__file__).resolve().parents[2]


ROOT = project_root()

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
EXTERNAL = DATA / "external"

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
AUDIT = RESULTS / "audit"

CONFIGS = ROOT / "configs"
PAPER = ROOT / "paper"

_ALL_DIRS = (
    RAW,
    INTERIM,
    PROCESSED,
    SPLITS,
    EXTERNAL,
    RUNS,
    TABLES,
    FIGURES,
    AUDIT,
)


def ensure_dirs() -> None:
    """Create every pipeline directory. Safe to call repeatedly."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def cache_dir(name: str) -> Path:
    """A named cache under ``data/interim``, created on demand.

    Used for anything expensive but derivable: fingerprint matrices, sequence
    identity graphs, ESM embeddings.
    """
    path = INTERIM / name
    path.mkdir(parents=True, exist_ok=True)
    return path
