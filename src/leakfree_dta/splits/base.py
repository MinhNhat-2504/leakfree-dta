"""Split representation.

A split is a labelling of rows, plus the group structure that produced it, plus
enough provenance to recreate it. All three are needed:

* The **labelling** is what a model trains on.
* The **group structure** is what per-group metrics and the group-DRO baseline
  consume, so it is carried alongside rather than recomputed.
* The **provenance** is what makes the published split artifact verifiable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.utils.hashing import digest_config, digest_frame_keys, hash_unit

Fold = Literal["train", "valid", "test", "excluded"]
FOLDS: tuple[Fold, ...] = ("train", "valid", "test", "excluded")


@dataclass(frozen=True)
class SplitSpec:
    """How a split is to be built.

    ``excluded`` exists as a fold because closing a leakage channel sometimes
    means a row can go nowhere: a ligand cluster straddling the boundary, or a
    measurement with no publication year under a temporal split. Discarding such
    rows silently would make the split sizes misleading, so they are labelled.
    """

    name: str
    ligand_axis: str = "random"
    protein_axis: str = "random"
    temporal_axis: str = "none"
    test_fraction: float = 0.2
    valid_fraction: float = 0.1
    protein_identity: float = 0.30
    ligand_cluster_cutoff: float = 0.4
    temporal_cutoff_year: int | None = None
    holdout_families: tuple[str, ...] = ()
    seed: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")
        if not 0.0 <= self.valid_fraction < 1.0:
            raise ValueError(f"valid_fraction must be in [0, 1), got {self.valid_fraction}")
        if self.test_fraction + self.valid_fraction >= 1.0:
            raise ValueError("test_fraction + valid_fraction must leave room for training data")

    @property
    def salt(self) -> str:
        """The hash salt that makes this spec's assignment distinct and reproducible."""
        return f"{self.name}|{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ligand_axis": self.ligand_axis,
            "protein_axis": self.protein_axis,
            "temporal_axis": self.temporal_axis,
            "test_fraction": self.test_fraction,
            "valid_fraction": self.valid_fraction,
            "protein_identity": self.protein_identity,
            "ligand_cluster_cutoff": self.ligand_cluster_cutoff,
            "temporal_cutoff_year": self.temporal_cutoff_year,
            "holdout_families": list(self.holdout_families),
            "seed": self.seed,
            "options": dict(self.options),
        }

    @property
    def digest(self) -> str:
        return digest_config(self.to_dict())


@dataclass
class SplitAssignment:
    """A built split: fold labels aligned to a dataset, plus its group structure."""

    fold: pd.Series
    spec: SplitSpec
    groups: dict[str, pd.Series] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.fold.unique()) - set(FOLDS)
        if unknown:
            raise ValueError(f"fold contains unknown labels {sorted(unknown)}")

    def subset(self, df: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Rows of ``df`` in ``fold``, with the group columns attached.

        The group columns ride along because every downstream consumer needs them
        -- per-group metrics, Mondrian conformal calibration, group DRO -- and
        recomputing protein clusters per consumer would be both slow and a chance
        for them to disagree.
        """
        if len(self.fold) != len(df):
            raise ValueError(
                f"split has {len(self.fold)} labels but the frame has {len(df)} rows; "
                "the split must be applied to the dataset it was built from"
            )
        mask = (self.fold == fold).to_numpy()
        out = df.loc[mask].copy()
        for name, labels in self.groups.items():
            out[name] = labels.to_numpy()[mask]
        return out.reset_index(drop=True)

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.subset(df, "train")

    def valid(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.subset(df, "valid")

    def test(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.subset(df, "test")

    def counts(self) -> dict[str, int]:
        counts = self.fold.value_counts().to_dict()
        return {fold: int(counts.get(fold, 0)) for fold in FOLDS}

    def is_degenerate(self, *, min_rows: int = 50) -> bool:
        """Whether any fold is too small for the split to be usable.

        A strict three-axis split on a sparse family can empty the test set; the
        pipeline must notice that rather than report metrics over nine rows.
        """
        counts = self.counts()
        return counts["train"] < min_rows or counts["test"] < min_rows

    def manifest(self, df: pd.DataFrame, *, dataset: str) -> dict[str, Any]:
        """The published artifact: row keys per fold, with digests.

        Row keys rather than row indices, because indices depend on the order the
        dataset happened to be built in, and the whole point of a published split
        is that someone else can apply it to their own copy.
        """
        if len(self.fold) != len(df):
            raise ValueError("manifest requires the dataset the split was built from")

        keys = df[schema.ROW_KEY].to_numpy()
        per_fold = {
            fold: sorted(keys[(self.fold == fold).to_numpy()].tolist()) for fold in FOLDS
        }
        return {
            "dataset": dataset,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "counts": self.counts(),
            "notes": self.notes,
            "digests": {fold: digest_frame_keys(rows) for fold, rows in per_fold.items()},
            "dataset_digest": digest_frame_keys(keys.tolist()),
            "row_keys": per_fold,
        }

    def group_frame(self) -> pd.DataFrame:
        """Group labels as a table, for inspection."""
        return pd.DataFrame({"fold": self.fold, **self.groups})


@runtime_checkable
class Splitter(Protocol):
    """Anything that can turn a dataset into a :class:`SplitAssignment`."""

    def split(self, df: pd.DataFrame) -> SplitAssignment: ...


def assign_groups_to_folds(
    group_labels: pd.Series,
    *,
    test_fraction: float,
    valid_fraction: float,
    salt: str,
) -> pd.Series:
    """Assign whole groups to folds by hashing the group label.

    Hashing rather than shuffling is what keeps a split stable as the dataset
    grows: a group's fold depends only on its own label, so adding new targets
    does not move existing ones. That property is what lets the published
    benchmark be extended without invalidating earlier results.

    Groups are assigned so that the *number of groups* matches the requested
    fractions. Row counts will differ from the fractions whenever groups differ in
    size, which they always do; :func:`rebalance_by_rows` is available when row
    balance matters more than group balance.
    """
    unique = pd.Index(pd.unique(group_labels.dropna()))
    draws = np.array([hash_unit(label, salt=salt) for label in unique], dtype=float)
    folds = np.where(
        draws < test_fraction,
        "test",
        np.where(draws < test_fraction + valid_fraction, "valid", "train"),
    )
    mapping = dict(zip(unique, folds, strict=True))
    return group_labels.map(mapping).fillna("excluded").astype(object)


def rebalance_by_rows(
    group_labels: pd.Series,
    *,
    test_fraction: float,
    valid_fraction: float,
    salt: str,
) -> pd.Series:
    """Assign groups to folds so that *row* counts approach the target fractions.

    Groups are visited in hash order and placed into test, then valid, while the
    fold is still under quota; everything else goes to training. A group larger
    than the whole quota is sent to training rather than being allowed to swallow
    the test set, and a final pass guarantees no fold ends up empty.
    """
    sizes = group_labels.value_counts()
    order = sorted(sizes.index, key=lambda label: hash_unit(label, salt=salt))
    total = int(sizes.sum())

    quota = {"test": test_fraction * total, "valid": valid_fraction * total}
    filled = {"test": 0, "valid": 0}
    mapping: dict[object, str] = {}

    for label in order:
        size = int(sizes[label])
        placed = False
        for fold in ("test", "valid"):
            if quota[fold] <= 0:
                continue
            # Accept the group while the fold is under quota. Overshooting by part
            # of one group is unavoidable with group-level assignment; accepting a
            # group that would more than double the quota is not.
            if filled[fold] < quota[fold] and size <= 2 * quota[fold]:
                mapping[label] = fold
                filled[fold] += size
                placed = True
                break
        if not placed:
            mapping[label] = "train"

    # Guarantee non-empty folds: promote the smallest training groups if needed.
    for fold in ("test", "valid"):
        if quota[fold] > 0 and filled[fold] == 0:
            candidates = [label for label in order if mapping[label] == "train"]
            if len(candidates) > 1:
                smallest = min(candidates, key=lambda label: int(sizes[label]))
                mapping[smallest] = fold
                filled[fold] += int(sizes[smallest])

    return group_labels.map(mapping).fillna("excluded").astype(object)
