"""Session-sized chunking of the Boltz-2 subset.

The chunker exists because 35 GPU-hours cannot be sat through: the run has to be
divisible into sessions and has to resume after one is killed. Both of those are
properties that fail quietly. A packing bug produces chunks that look fine in the
summary table and then overrun the session, and a progress bug reports work as
done that was never done -- which, on an external run, is how a partial result
gets written up as a complete one.

So the two claims the documentation makes are tested here: that no chunk exceeds
the budget, and that progress is read from what is actually on disk.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def chunker():
    spec = importlib.util.spec_from_file_location(
        "chunk_boltz2_subset", REPO_ROOT / "scripts" / "chunk_boltz2_subset.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["chunk_boltz2_subset"] = module
    spec.loader.exec_module(module)
    return module


def make_subset(lengths: list[int], chunker) -> pd.DataFrame:
    from leakfree_dta.data import schema

    return pd.DataFrame(
        {
            schema.ROW_KEY: [f"row{i}" for i in range(len(lengths))],
            schema.TARGET_SEQUENCE: ["A" * n for n in lengths],
        }
    )


def test_cost_grows_faster_than_length(chunker):
    """A 2x longer protein must cost more than 2x, or packing by length is pointless."""
    base = chunker.estimate_minutes(400)
    assert base == pytest.approx(chunker.BASE_MINUTES)
    assert chunker.estimate_minutes(800) > 2 * base


def test_no_chunk_exceeds_the_budget(chunker):
    """The property the session length depends on.

    The budget is deliberately smaller than the total cost. A budget nothing
    reaches makes this assertion pass without packing anything, which is how a
    test like this stops testing and nobody notices -- hence the second assert.
    """
    from leakfree_dta.data import schema

    lengths = [300, 1306, 450, 1052, 714, 380, 900, 465, 320, 1100]
    subset = make_subset(lengths, chunker)
    budget = 20.0
    assert sum(chunker.estimate_minutes(n) for n in lengths) > 2 * budget

    chunks = chunker.pack(subset, budget)

    assert len(chunks) > 1, "budget is not binding; this test would pass vacuously"
    for chunk in chunks:
        minutes = chunk[schema.TARGET_SEQUENCE].str.len().map(chunker.estimate_minutes).sum()
        # A complex whose own cost exceeds the budget gets a chunk to itself and
        # a warning: there is nowhere smaller to put it, and dropping it would
        # be worse. Every other chunk must fit.
        assert minutes <= budget or len(chunk) == 1


def test_packing_loses_no_complexes_and_duplicates_none(chunker):
    """A dropped row is a complex that never gets predicted and is never missed."""
    from leakfree_dta.data import schema

    lengths = [300, 1306, 450, 1052, 714, 380, 900, 465, 320, 1100, 600, 250]
    subset = make_subset(lengths, chunker)

    chunks = chunker.pack(subset, 20.0)
    packed = [key for chunk in chunks for key in chunk[schema.ROW_KEY]]

    assert len(chunks) > 1, "budget is not binding; this test would pass vacuously"
    assert sorted(packed) == sorted(subset[schema.ROW_KEY])
    assert len(packed) == len(set(packed))


def test_a_single_oversized_complex_gets_its_own_chunk(chunker):
    """It must not be silently dropped; a warning plus a chunk of its own is the deal."""
    from leakfree_dta.data import schema

    subset = make_subset([4000, 300, 320], chunker)
    chunks = chunker.pack(subset, 10.0)

    packed = [key for chunk in chunks for key in chunk[schema.ROW_KEY]]
    assert sorted(packed) == sorted(subset[schema.ROW_KEY])
    oversized = [c for c in chunks if (c[schema.TARGET_SEQUENCE].str.len() == 4000).any()]
    assert len(oversized) == 1 and len(oversized[0]) == 1


def test_progress_counts_only_predictions_that_exist(chunker, tmp_path):
    """Read from disk, so a killed session cannot be reported as a finished one."""
    chunk_dir = tmp_path / "chunk01"
    chunk_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (chunk_dir / f"{name}.yaml").write_text("sequences: []\n", encoding="utf-8")

    outputs = tmp_path / "outputs"
    assert chunker.chunk_progress(chunk_dir, outputs) == (0, 3)

    # Boltz's layout: boltz_results_<name>/predictions/<name>/affinity_<name>.json
    for name in ("alpha", "beta"):
        target = outputs / f"boltz_results_{name}" / "predictions" / name
        target.mkdir(parents=True)
        (target / f"affinity_{name}.json").write_text(
            json.dumps({"affinity_pred_value": 6.5}), encoding="utf-8"
        )

    assert chunker.chunk_progress(chunk_dir, outputs) == (2, 3)


def test_progress_ignores_predictions_from_another_chunk(chunker, tmp_path):
    """Chunks share one output directory, so the count has to be chunk-specific."""
    chunk_dir = tmp_path / "chunk01"
    chunk_dir.mkdir()
    (chunk_dir / "alpha.yaml").write_text("sequences: []\n", encoding="utf-8")

    outputs = tmp_path / "outputs"
    target = outputs / "boltz_results_stranger" / "predictions" / "stranger"
    target.mkdir(parents=True)
    (target / "affinity_stranger.json").write_text(
        json.dumps({"affinity_pred_value": 5.0}), encoding="utf-8"
    )

    assert chunker.chunk_progress(chunk_dir, outputs) == (0, 1)
