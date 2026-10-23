"""Composing the three axes into one split.

The composition rule is strict and it costs rows, deliberately.

Each axis independently assigns its groups to a side -- train, valid or test. A
row is placed on a side only when *every* constrained axis agrees on that side.
Where the axes disagree, the row goes to ``excluded``.

Disagreement is not an edge case, it is most of the data. Under a ligand-cluster
and protein-cluster split, a row whose ligand is in a training cluster but whose
protein is in a test cluster can go nowhere: putting it in training reveals a test
protein, and putting it in test reveals a training ligand. This is the "both-cold"
block structure, and the discarded fraction is reported rather than hidden, since
it is the honest price of a leakage-free evaluation.
"""

from __future__ import annotations

import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.splits.axes import (
    WILDCARD,
    ligand_groups,
    protein_groups,
    temporal_sides,
)
from leakfree_dta.splits.base import (
    SplitAssignment,
    SplitSpec,
    assign_groups_to_folds,
    rebalance_by_rows,
)
from leakfree_dta.utils.hashing import hash_unit

logger = get_logger(__name__)


class ThreeAxisSplitter:
    """Builds a split from a :class:`SplitSpec`."""

    def __init__(self, spec: SplitSpec, *, balance_by_rows: bool = True) -> None:
        self.spec = spec
        self.balance_by_rows = balance_by_rows

    def split(self, df: pd.DataFrame) -> SplitAssignment:
        spec = self.spec
        groups: dict[str, pd.Series] = {}
        sides: list[pd.Series] = []

        ligand = ligand_groups(df, spec.ligand_axis, cluster_cutoff=spec.ligand_cluster_cutoff)
        if ligand is not None:
            groups["ligand_group"] = ligand.reset_index(drop=True)
            sides.append(
                self._groups_to_sides(ligand, salt=f"{spec.salt}|ligand").rename("ligand_side")
            )

        protein = protein_groups(
            df,
            spec.protein_axis,
            identity=spec.protein_identity,
            holdout_families=spec.holdout_families,
        )
        if protein is not None:
            groups["protein_group"] = protein.reset_index(drop=True)
            if spec.protein_axis == "family":
                sides.append(self._family_sides(protein).rename("protein_side"))
            else:
                sides.append(
                    self._groups_to_sides(protein, salt=f"{spec.salt}|protein").rename("protein_side")
                )

        temporal = temporal_sides(
            df,
            spec.temporal_axis,
            cutoff_year=spec.temporal_cutoff_year,
            valid_years=int(spec.options.get("temporal_valid_years", 2)),
        )
        if temporal is not None:
            sides.append(temporal)

        fold = self._combine(df, sides)
        notes = self._notes(df, fold, sides)

        assignment = SplitAssignment(
            fold=fold.reset_index(drop=True),
            spec=spec,
            groups=groups,
            notes=notes,
        )
        logger.info("split %r -> %s", spec.name, assignment.counts())
        if assignment.is_degenerate():
            logger.warning(
                "split %r is degenerate (%s); the axes may be too strict for this dataset",
                spec.name,
                assignment.counts(),
            )
        return assignment

    # --- internals ----------------------------------------------------------

    def _groups_to_sides(self, group_labels: pd.Series, *, salt: str) -> pd.Series:
        assigner = rebalance_by_rows if self.balance_by_rows else assign_groups_to_folds
        return assigner(
            group_labels,
            test_fraction=self.spec.test_fraction,
            valid_fraction=self.spec.valid_fraction,
            salt=salt,
        )

    def _family_sides(self, family_labels: pd.Series) -> pd.Series:
        """Held-out families become test; the rest split into train and valid.

        Validation comes from the training families, because the point of a family
        holdout is to measure transfer to an unseen family -- so the held-out
        family must not inform model selection either.
        """
        holdout = set(self.spec.holdout_families)
        sides = pd.Series("train", index=family_labels.index, dtype=object)
        sides[family_labels.isin(holdout)] = "test"

        if self.spec.valid_fraction > 0:
            train_mask = sides == "train"
            salt = f"{self.spec.salt}|famvalid"
            # Carve validation out by whole family, so no family straddles the
            # train/valid boundary.
            valid_families = {
                label
                for label in family_labels[train_mask].unique()
                if hash_unit(label, salt=salt) < self.spec.valid_fraction
            }
            n_train_families = int(family_labels[train_mask].nunique())
            if valid_families and len(valid_families) < n_train_families:
                sides[train_mask & family_labels.isin(valid_families)] = "valid"
            else:
                # With only one or two training families there is none to spare, so
                # validation rows are drawn within families. Logged because it is a
                # weaker guarantee than the rest of the split provides.
                logger.warning(
                    "family holdout has %d training families, too few to reserve one for "
                    "validation; validation rows are drawn within families instead",
                    n_train_families,
                )
                draws = family_labels[train_mask].index.map(
                    lambda idx: hash_unit(idx, salt=f"{salt}|row")
                )
                sides.loc[train_mask] = [
                    "valid" if draw < self.spec.valid_fraction else "train" for draw in draws
                ]
        return sides

    def _combine(self, df: pd.DataFrame, sides: list[pd.Series]) -> pd.Series:
        if not sides:
            # Every axis is a wildcard: this is the plain random split, kept as the
            # permissive end of the ladder so the paper can quantify the inflation
            # relative to it.
            draws = df[schema.ROW_KEY].map(
                lambda key: hash_unit(key, salt=f"{self.spec.salt}|row")
            )
            return pd.Series(
                [
                    "test"
                    if draw < self.spec.test_fraction
                    else "valid"
                    if draw < self.spec.test_fraction + self.spec.valid_fraction
                    else "train"
                    for draw in draws
                ],
                index=df.index,
                dtype=object,
            )

        stacked = pd.concat([side.reset_index(drop=True) for side in sides], axis=1)
        # A row is placed only where every constrained axis names the same side.
        agree = stacked.nunique(axis=1) == 1
        fold = pd.Series("excluded", index=stacked.index, dtype=object)
        fold[agree] = stacked.iloc[:, 0][agree]
        fold[stacked.isna().any(axis=1)] = "excluded"
        return fold

    def _notes(self, df: pd.DataFrame, fold: pd.Series, sides: list[pd.Series]) -> dict[str, object]:
        n_excluded = int((fold == "excluded").sum())
        notes: dict[str, object] = {
            "n_rows": len(df),
            "n_excluded": n_excluded,
            "excluded_fraction": round(n_excluded / max(len(df), 1), 4),
            "axes_constrained": [side.name for side in sides],
        }
        if self.spec.ligand_axis != WILDCARD or self.spec.protein_axis != WILDCARD:
            notes["why_excluded"] = (
                "Rows whose ligand and protein groups fall on different sides of the "
                "boundary cannot be placed without leaking. This is the cost of a "
                "block-structured, leakage-free split."
            )
        return notes


def build_split(
    df: pd.DataFrame, spec: SplitSpec, *, balance_by_rows: bool = True
) -> SplitAssignment:
    """Convenience wrapper around :class:`ThreeAxisSplitter`."""
    return ThreeAxisSplitter(spec, balance_by_rows=balance_by_rows).split(df)


#: The ladder of splits the paper reports, from the field's usual practice to the
#: setting that closes every channel. Each rung adds one constraint, so the drop
#: between rungs attributes the inflation to a specific channel.
SPLIT_LADDER: tuple[SplitSpec, ...] = (
    SplitSpec(
        name="random",
        ligand_axis="random",
        protein_axis="random",
        temporal_axis="none",
    ),
    SplitSpec(
        name="ligand_scaffold",
        ligand_axis="scaffold",
        protein_axis="random",
        temporal_axis="none",
    ),
    SplitSpec(
        name="ligand_cluster",
        ligand_axis="cluster",
        protein_axis="random",
        temporal_axis="none",
    ),
    SplitSpec(
        name="protein_identity",
        ligand_axis="random",
        protein_axis="identity",
        temporal_axis="none",
    ),
    SplitSpec(
        name="protein_cluster30",
        ligand_axis="random",
        protein_axis="cluster",
        protein_identity=0.30,
        temporal_axis="none",
    ),
    SplitSpec(
        name="protein_cluster50",
        ligand_axis="random",
        protein_axis="cluster",
        protein_identity=0.50,
        temporal_axis="none",
    ),
    SplitSpec(
        name="both_cold",
        ligand_axis="cluster",
        protein_axis="cluster",
        protein_identity=0.30,
        temporal_axis="none",
    ),
    SplitSpec(
        name="temporal",
        ligand_axis="random",
        protein_axis="random",
        temporal_axis="year",
        # Train through 2018, tune on 2019--2020, and evaluate on 2021+.
        # This is explicit rather than the data-dependent 80th percentile so a
        # refreshed ChEMBL release cannot silently move the boundary.
        temporal_cutoff_year=2020,
        options={"temporal_valid_years": 2},
    ),
    SplitSpec(
        name="leakfree",
        ligand_axis="cluster",
        protein_axis="cluster",
        protein_identity=0.30,
        temporal_axis="year",
        temporal_cutoff_year=2020,
        options={"temporal_valid_years": 2},
    ),
)
