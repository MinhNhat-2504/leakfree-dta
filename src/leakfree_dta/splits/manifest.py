"""Frozen split manifests: the artifact the paper publishes.

A manifest records which row went to which fold, keyed by ``row_key`` rather than
by position, together with digests of each fold and of the dataset as a whole.
That combination is what makes the benchmark usable by someone else:

* They can apply the split to their own copy of the data, rather than downloading
  ours and trusting that it matches.
* The digest tells them immediately if their copy differs from the one the paper
  reports on -- a near-universal failure mode when "the same" ChEMBL extract is
  rebuilt a year later.

The paper's claim to be a reusable resource rests on this file format, so it is
versioned and validated rather than being an ad-hoc dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.splits.base import FOLDS, SplitAssignment, SplitSpec
from leakfree_dta.utils.hashing import digest_frame_keys
from leakfree_dta.utils.io import read_json, write_json

logger = get_logger(__name__)

MANIFEST_VERSION = 1


def save_manifest(
    assignment: SplitAssignment,
    df: pd.DataFrame,
    *,
    dataset: str,
    directory: str | Path,
) -> Path:
    """Write ``<directory>/<split name>.json``."""
    payload = assignment.manifest(df, dataset=dataset)
    payload["manifest_version"] = MANIFEST_VERSION

    path = Path(directory) / f"{assignment.spec.name}.json"
    write_json(payload, path)
    logger.info(
        "wrote split manifest %s (%s)",
        path.name,
        ", ".join(f"{fold}={count:,}" for fold, count in assignment.counts().items()),
    )
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read and sanity-check a manifest file."""
    payload = read_json(path)
    version = payload.get("manifest_version")
    if version != MANIFEST_VERSION:
        logger.warning(
            "manifest %s has version %s, this build writes version %d",
            Path(path).name,
            version,
            MANIFEST_VERSION,
        )
    for key in ("dataset", "spec", "row_keys", "digests"):
        if key not in payload:
            raise ValueError(f"{path} is not a valid split manifest (missing {key!r})")
    return payload


def apply_manifest(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    verify: bool = True,
    allow_partial: bool = False,
) -> SplitAssignment:
    """Reconstruct a split on ``df`` from a manifest.

    This is the function a reader of the paper calls. ``verify=True`` checks the
    dataset digest and refuses to proceed on a mismatch, because applying a split
    to a different dataset than it was built for produces numbers that look
    plausible and mean nothing.

    ``allow_partial=True`` relaxes that into a warning and labels rows the manifest
    does not mention as ``excluded`` -- appropriate when a reader has a subset of
    the data, for instance only one target family.
    """
    keys = df[schema.ROW_KEY].astype(str)
    expected = manifest.get("dataset_digest")
    actual = digest_frame_keys(keys.tolist())

    if expected and expected != actual:
        message = (
            f"dataset digest mismatch: the manifest was built on a dataset with digest "
            f"{expected[:12]}..., this frame has {actual[:12]}.... The curated data differs "
            "from the version the split was built on."
        )
        if verify and not allow_partial:
            raise ValueError(message)
        logger.warning(message)

    lookup: dict[str, str] = {}
    for fold in FOLDS:
        for key in manifest["row_keys"].get(fold, []):
            lookup[str(key)] = fold

    fold = keys.map(lookup)
    n_unknown = int(fold.isna().sum())
    if n_unknown:
        message = f"{n_unknown:,} of {len(df):,} rows are absent from the manifest"
        if allow_partial:
            logger.warning("%s; labelling them 'excluded'", message)
        elif verify:
            raise ValueError(message + "; pass allow_partial=True to label them 'excluded'")
        else:
            logger.warning(message)
    fold = fold.fillna("excluded")

    spec = SplitSpec(**{**manifest["spec"], "holdout_families": tuple(manifest["spec"].get("holdout_families", []))})
    return SplitAssignment(
        fold=fold.reset_index(drop=True),
        spec=spec,
        notes={**manifest.get("notes", {}), "loaded_from_manifest": True},
    )


def manifest_summary(directory: str | Path) -> pd.DataFrame:
    """A table of every manifest in a directory, for the paper's splits table."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            payload = load_manifest(path)
        except ValueError as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            continue
        spec = payload["spec"]
        counts = payload.get("counts", {})
        total = sum(counts.values()) or 1
        rows.append(
            {
                "split": spec.get("name", path.stem),
                "ligand_axis": spec.get("ligand_axis"),
                "protein_axis": spec.get("protein_axis"),
                "temporal_axis": spec.get("temporal_axis"),
                "n_train": counts.get("train", 0),
                "n_valid": counts.get("valid", 0),
                "n_test": counts.get("test", 0),
                "n_excluded": counts.get("excluded", 0),
                "excluded_pct": round(100 * counts.get("excluded", 0) / total, 2),
                "spec_digest": payload.get("spec_digest", "")[:12],
            }
        )
    return pd.DataFrame(rows)
