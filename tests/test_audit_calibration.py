"""Calibration tests: does the audit recover what was planted?

These are the most important tests in the repository. Every number the paper
reports comes out of the audit, so the audit itself has to be validated against a
dataset whose leakage is known by construction. A check that cannot recover a
planted quantity is not a measurement, and a result built on it would not be
either.

Each test states the planted quantity, the recovered quantity, and the tolerance,
so a failure says which part of the instrument drifted.
"""

from __future__ import annotations

import numpy as np
import pytest

from leakfree_dta.audit import across, shortcut, within
from leakfree_dta.data import schema


class TestLabelNoise:
    """The conflicting-label check against planted disagreements."""

    def test_recovers_planted_conflicts(self, synthetic_df, synthetic_truth):
        result = within.check_conflicting_labels(synthetic_df, threshold=1.0)
        recovered = result.statistics["n_conflicting_pairs"]
        planted = len(synthetic_truth.conflicting_pair_keys)

        # Replicates were planted with a 1.6 log-unit offset, so every one of them
        # must exceed a 1.0 threshold. The check may find a few more where ordinary
        # replicate noise happens to cross it, so this is a lower bound.
        assert recovered >= planted, (
            f"planted {planted} conflicting pairs, the check found only {recovered}"
        )
        assert recovered <= planted * 1.5, (
            f"the check found {recovered} conflicts for {planted} planted; "
            "it is over-reporting"
        )

    def test_noise_floor_is_positive_and_bounded(self, synthetic_df):
        result = within.check_conflicting_labels(synthetic_df)
        floor = result.statistics["pooled_within_pair_std"]
        assert 0.0 < floor < 2.0, f"implausible noise floor {floor}"

    def test_duplicate_count_matches(self, synthetic_df, synthetic_truth):
        result = within.check_duplicate_measurements(synthetic_df)
        assert result.statistics["n_duplicated_pairs"] == len(
            set(synthetic_truth.duplicate_pair_keys)
        )


class TestShortcutProbe:
    """The shortcut probe against the generator's analytic ceiling.

    The synthetic affinities are ``global + target + ligand + interaction + noise``
    with independent terms, so the correlation an additive model can reach is known
    in closed form. The probe must land near it: too low means the probe is weak,
    too high means it is picking up something it should not be able to see.
    """

    def test_additive_probe_approaches_the_ceiling(self, synthetic_df, synthetic_truth):
        # A random split, which is the setting the ceiling was derived for.
        rng = np.random.default_rng(0)
        mask = rng.random(len(synthetic_df)) < 0.75
        train, test = synthetic_df[mask], synthetic_df[~mask]

        scores = shortcut.shortcut_scores(train, test)
        achieved = scores["additive_bias"].pearson
        ceiling = synthetic_truth.shortcut_ceiling

        assert achieved == pytest.approx(ceiling, abs=0.12), (
            f"additive probe reached Pearson {achieved:.3f} against an analytic "
            f"ceiling of {ceiling:.3f}"
        )
        # It must not exceed the ceiling by a meaningful margin: an additive model
        # cannot explain interaction variance, so doing so would mean the probe is
        # seeing the test labels somehow.
        assert achieved <= ceiling + 0.08

    def test_global_mean_carries_no_ranking_signal(self, synthetic_df):
        rng = np.random.default_rng(0)
        mask = rng.random(len(synthetic_df)) < 0.75
        scores = shortcut.shortcut_scores(synthetic_df[mask], synthetic_df[~mask])
        assert scores["global_mean"].pearson == 0.0

    def test_target_mean_loses_coverage_on_a_protein_cold_split(self, aggregated):
        """A protein-cold split must destroy the target-mean shortcut entirely.

        This is the check that proves the probe is measuring the right thing: if
        no test target appears in training, a per-target lookup has nothing to
        look up, and its coverage must be zero.
        """
        from leakfree_dta.splits.base import SplitSpec
        from leakfree_dta.splits.three_axis import build_split

        assignment = build_split(
            aggregated, SplitSpec(name="protein_cold", protein_axis="identity")
        )
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        scores = shortcut.shortcut_scores(train, test)

        assert scores["target_mean"].coverage == 0.0, (
            "a protein-cold split left some test target in training"
        )


class TestNearDuplicateChannel:
    """The channel the planted homologous series exists to exercise."""

    def test_homologues_are_detected_as_near_duplicates(self, synthetic_df, synthetic_truth):
        """Planted analogues must register at or above the near-duplicate threshold.

        The analogues are homologues -- same core, linker one or three carbons
        longer. They have distinct InChIKeys, so InChIKey-based deduplication does
        not see them, which is exactly why fingerprint similarity has to.
        """
        from leakfree_dta.chem.fingerprints import ecfp_bitvects
        from leakfree_dta.chem.similarity import nearest_neighbor

        ligands = synthetic_df.drop_duplicates(subset=[schema.LIGAND_KEY])
        analogues = ligands[ligands[schema.LIGAND_KEY].isin(synthetic_truth.analog_ligand_keys)]
        bases = ligands[~ligands[schema.LIGAND_KEY].isin(synthetic_truth.analog_ligand_keys)]
        assert len(analogues) > 0, "the fixture planted no analogues"

        neighbors = nearest_neighbor(
            ecfp_bitvects(analogues[schema.SMILES]), ecfp_bitvects(bases[schema.SMILES])
        )
        similarity = neighbors.similarity[np.isfinite(neighbors.similarity)]
        above = (similarity >= across.NEAR_DUPLICATE_TANIMOTO).mean()

        assert above >= 0.9, (
            f"only {above:.0%} of planted analogues reach Tanimoto "
            f"{across.NEAR_DUPLICATE_TANIMOTO}; the fixture is not exercising the "
            "near-duplicate channel"
        )

    def test_homologues_have_distinct_keys_and_scaffolds(self, synthetic_df, synthetic_truth):
        """The property that makes scaffold splitting insufficient.

        If homologues shared a scaffold, a scaffold split would keep them together
        and the channel would close by accident. They do not, which is the point.
        """
        ligands = synthetic_df.drop_duplicates(subset=[schema.LIGAND_KEY])
        analogues = ligands[ligands[schema.LIGAND_KEY].isin(synthetic_truth.analog_ligand_keys)]

        assert analogues[schema.LIGAND_KEY].nunique() == len(analogues), (
            "planted analogues collapsed onto existing InChIKeys"
        )
        # Each analogue's scaffold differs from its parent's, because linker length
        # is part of the Bemis-Murcko framework.
        assert analogues[schema.SCAFFOLD].notna().all()


class TestLadderOrdering:
    """The ordering the paper's argument depends on.

    These assertions are the paper's claim in executable form. If tightening a
    split ever fails to reduce recoverable leakage, either the split machinery or
    the audit has broken, and every downstream number is suspect.
    """

    @pytest.fixture(scope="class")
    def ladder_leakage(self, aggregated) -> dict[str, float]:
        from leakfree_dta.audit.runner import audit_split
        from leakfree_dta.splits.base import SplitSpec
        from leakfree_dta.splits.three_axis import build_split

        specs = {
            "random": SplitSpec(name="random"),
            "scaffold": SplitSpec(name="scaffold", ligand_axis="scaffold"),
            "cluster": SplitSpec(name="cluster", ligand_axis="cluster"),
        }
        out: dict[str, float] = {}
        for name, spec in specs.items():
            assignment = build_split(aggregated, spec)
            report = audit_split(
                assignment.train(aggregated),
                assignment.test(aggregated),
                dataset="synthetic",
                split=name,
                # The pairwise alignment is slow and irrelevant to the ligand axis.
                skip=("protein_identity_overlap",),
            )
            out[name] = report.recoverable_fraction()
        return out

    def test_ligand_clustering_closes_what_scaffolding_leaves_open(self, ladder_leakage):
        """Scaffold splitting is close to no constraint; clustering is not.

        The headline result, reproduced on calibrated data: Bemis-Murcko splitting
        barely reduces recoverable leakage, because homologues get different
        scaffolds, while fingerprint clustering removes nearly all of it.
        """
        assert ladder_leakage["cluster"] < ladder_leakage["scaffold"], (
            f"clustering ({ladder_leakage['cluster']:.4f}) did not improve on "
            f"scaffolding ({ladder_leakage['scaffold']:.4f})"
        )
        assert ladder_leakage["cluster"] < 0.05, (
            f"the ligand-cluster split still leaves {ladder_leakage['cluster']:.2%} "
            "of test rows recoverable"
        )

    def test_random_split_leaks_most(self, ladder_leakage):
        assert ladder_leakage["random"] >= ladder_leakage["cluster"]
        assert ladder_leakage["random"] > 0.05, (
            "the random split shows almost no recoverable leakage, so the fixture "
            "is not exercising the audit"
        )
