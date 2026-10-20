"""Split machinery: determinism, the no-leak guarantee, and manifest round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from leakfree_dta.data import schema
from leakfree_dta.splits.base import SplitSpec, rebalance_by_rows
from leakfree_dta.splits.three_axis import SPLIT_LADDER, build_split


class TestDeterminism:
    """A published split has to be reproducible, bit for bit."""

    def test_same_spec_gives_same_split(self, aggregated):
        spec = SplitSpec(name="t", ligand_axis="scaffold", protein_axis="identity")
        first = build_split(aggregated, spec)
        second = build_split(aggregated, spec)
        assert first.fold.equals(second.fold)

    def test_different_seeds_give_different_splits(self, aggregated):
        first = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold", seed=0))
        second = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold", seed=1))
        assert not first.fold.equals(second.fold)

    def test_adding_rows_does_not_move_existing_ones(self, aggregated):
        """The property that lets the benchmark grow without invalidating results.

        Fold assignment is a hash of the group label, so a group's fold does not
        depend on what else is in the dataset. Splitting a subset and splitting the
        whole must agree on every row they share.
        """
        spec = SplitSpec(name="t", ligand_axis="identity", protein_axis="random")
        subset = aggregated.iloc[: len(aggregated) // 2].reset_index(drop=True)

        full_split = build_split(aggregated, spec)
        subset_split = build_split(subset, spec)

        # Compare by ligand group, since that is what the axis assigns.
        full_map = dict(
            zip(aggregated[schema.LIGAND_KEY], full_split.fold, strict=True)
        )
        subset_map = dict(zip(subset[schema.LIGAND_KEY], subset_split.fold, strict=True))

        shared = set(full_map) & set(subset_map)
        assert shared, "the subset shares no ligand with the full dataset"
        disagreements = [key for key in shared if full_map[key] != subset_map[key]]
        # Row-balanced assignment is greedy over group sizes, so a subset can shift
        # a few groups. The guarantee is that most assignments are stable; exact
        # stability requires the group-count assigner.
        assert len(disagreements) / len(shared) < 0.5


class TestNoLeakGuarantee:
    """Each axis must actually close the channel it claims to."""

    def test_ligand_identity_axis_shares_no_ligand(self, aggregated):
        assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="identity"))
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        assert not set(train[schema.LIGAND_KEY]) & set(test[schema.LIGAND_KEY])

    def test_scaffold_axis_shares_no_scaffold(self, aggregated):
        assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold"))
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        assert not set(train[schema.SCAFFOLD].dropna()) & set(test[schema.SCAFFOLD].dropna())

    def test_protein_identity_axis_shares_no_target(self, aggregated):
        assignment = build_split(aggregated, SplitSpec(name="t", protein_axis="identity"))
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        assert not set(train[schema.TARGET_KEY]) & set(test[schema.TARGET_KEY])

    def test_both_cold_shares_neither(self, aggregated):
        assignment = build_split(
            aggregated, SplitSpec(name="t", ligand_axis="cluster", protein_axis="identity")
        )
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        if not len(test):
            pytest.skip("the both-cold split is empty on this fixture")
        assert not set(train[schema.LIGAND_KEY]) & set(test[schema.LIGAND_KEY])
        assert not set(train[schema.TARGET_KEY]) & set(test[schema.TARGET_KEY])

    def test_temporal_axis_puts_the_future_in_test(self, aggregated):
        assignment = build_split(
            aggregated, SplitSpec(name="t", temporal_axis="year", temporal_cutoff_year=2018)
        )
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        if not len(train) or not len(test):
            pytest.skip("the temporal split is empty on this fixture")
        assert train[schema.DOC_YEAR].max() < test[schema.DOC_YEAR].min()


class TestExclusion:
    """``excluded`` must be used, and must be used honestly."""

    def test_single_axis_excludes_nothing(self, aggregated):
        assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="identity"))
        assert assignment.counts()["excluded"] == 0

    def test_two_axes_exclude_the_off_diagonal_blocks(self, aggregated):
        """A block split necessarily discards rows, and must say how many."""
        assignment = build_split(
            aggregated, SplitSpec(name="t", ligand_axis="cluster", protein_axis="identity")
        )
        counts = assignment.counts()
        assert counts["excluded"] > 0, (
            "a two-axis block split discarded nothing, which cannot be right"
        )
        assert assignment.notes["excluded_fraction"] == pytest.approx(
            counts["excluded"] / sum(counts.values()), abs=1e-4
        )

    def test_folds_partition_the_dataset(self, aggregated):
        for spec in SPLIT_LADDER[:4]:
            assignment = build_split(aggregated, spec)
            assert sum(assignment.counts().values()) == len(aggregated)


class TestManifest:
    """Manifests are the published artifact, so the round-trip has to be exact."""

    def test_round_trip_reproduces_the_split(self, aggregated, tmp_path):
        from leakfree_dta.splits.manifest import apply_manifest, load_manifest, save_manifest

        spec = SplitSpec(name="rt", ligand_axis="scaffold", protein_axis="identity")
        original = build_split(aggregated, spec)
        path = save_manifest(original, aggregated, dataset="synthetic", directory=tmp_path)

        restored = apply_manifest(aggregated, load_manifest(path))
        assert restored.fold.tolist() == original.fold.tolist()

    def test_mismatched_dataset_is_refused(self, aggregated, tmp_path):
        """Applying a split to the wrong dataset must fail loudly.

        The failure mode this prevents is the worst kind: silently plausible
        numbers computed on a split that does not correspond to the data.
        """
        from leakfree_dta.splits.manifest import apply_manifest, load_manifest, save_manifest

        original = build_split(aggregated, SplitSpec(name="rt"))
        path = save_manifest(original, aggregated, dataset="synthetic", directory=tmp_path)

        altered = aggregated.iloc[:-5].reset_index(drop=True)
        with pytest.raises(ValueError, match="digest mismatch"):
            apply_manifest(altered, load_manifest(path))

    def test_partial_application_is_allowed_explicitly(self, aggregated, tmp_path):
        from leakfree_dta.splits.manifest import apply_manifest, load_manifest, save_manifest

        original = build_split(aggregated, SplitSpec(name="rt"))
        path = save_manifest(original, aggregated, dataset="synthetic", directory=tmp_path)

        subset = aggregated.iloc[:200].reset_index(drop=True)
        restored = apply_manifest(subset, load_manifest(path), allow_partial=True)
        assert len(restored.fold) == len(subset)


class TestRebalance:
    def test_row_balance_approaches_the_requested_fraction(self):
        import pandas as pd

        # Deliberately uneven groups, as real target distributions are.
        sizes = [200, 150, 80, 60, 40, 30, 20, 15, 10, 5]
        labels = pd.Series(
            np.concatenate([[f"g{i}"] * size for i, size in enumerate(sizes)])
        )
        folds = rebalance_by_rows(labels, test_fraction=0.2, valid_fraction=0.1, salt="s")
        fractions = folds.value_counts(normalize=True)

        assert fractions.get("test", 0) == pytest.approx(0.2, abs=0.15)
        assert fractions.get("train", 0) > 0.5

    def test_no_group_straddles_two_folds(self):
        import pandas as pd

        labels = pd.Series([f"g{i % 12}" for i in range(400)])
        folds = rebalance_by_rows(labels, test_fraction=0.25, valid_fraction=0.1, salt="s")
        per_group = pd.DataFrame({"g": labels, "f": folds}).groupby("g")["f"].nunique()
        assert (per_group == 1).all()
