"""Models: the interface contract, and the behaviour the paper relies on."""

from __future__ import annotations

import numpy as np
import pytest

from leakfree_dta.data import schema
from leakfree_dta.models.registry import build_model
from leakfree_dta.splits.base import SplitSpec
from leakfree_dta.splits.three_axis import build_split

CHEAP_MODELS = ("global_mean", "target_mean", "ligand_mean", "additive_bias", "ridge_ecfp")


@pytest.fixture(scope="module")
def folds(aggregated):
    assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold"))
    return (
        assignment.train(aggregated),
        assignment.valid(aggregated),
        assignment.test(aggregated),
    )


class TestInterface:
    @pytest.mark.parametrize("name", CHEAP_MODELS)
    def test_fit_predict_round_trip(self, name, folds):
        train, valid, test = folds
        model = build_model(name)
        result = model.fit(train, valid if len(valid) else None)

        assert result.n_train == len(train)
        prediction = model.predict(test)
        assert prediction.mean.shape == (len(test),)
        assert np.isfinite(prediction.mean).all(), f"{name} produced non-finite predictions"

    @pytest.mark.parametrize("name", CHEAP_MODELS)
    def test_predicting_before_fitting_is_refused(self, name, folds):
        _, _, test = folds
        with pytest.raises(RuntimeError, match="must be fitted"):
            build_model(name).predict(test)

    def test_unknown_model_names_the_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            build_model("no_such_model")

    def test_prediction_rejects_mismatched_uncertainty(self):
        from leakfree_dta.models.base import Prediction

        with pytest.raises(ValueError, match="shape"):
            Prediction(mean=np.zeros(10), std=np.zeros(5))


class TestDegenerateBaselines:
    """These models have to behave exactly as their names claim."""

    def test_global_mean_predicts_one_value(self, folds):
        train, _, test = folds
        model = build_model("global_mean")
        model.fit(train)
        prediction = model.predict(test)
        assert np.allclose(prediction.mean, prediction.mean[0])
        assert prediction.mean[0] == pytest.approx(float(np.mean(model.targets(train))), abs=1e-4)

    def test_target_mean_ignores_the_ligand(self, folds):
        """Two different ligands against one target must get the same prediction.

        If this ever fails, the model is using ligand information and is no longer
        the degenerate reference the paper compares against.
        """
        train, _, test = folds
        model = build_model("target_mean")
        model.fit(train)
        prediction = model.predict(test)

        frame = test.assign(pred=prediction.mean)
        spreads = frame.groupby(schema.TARGET_KEY, observed=True)["pred"].nunique()
        assert (spreads == 1).all()

    def test_ligand_mean_ignores_the_target(self, folds):
        train, _, test = folds
        model = build_model("ligand_mean")
        model.fit(train)
        frame = test.assign(pred=model.predict(test).mean)
        spreads = frame.groupby(schema.LIGAND_KEY, observed=True)["pred"].nunique()
        assert (spreads == 1).all()

    def test_additive_bias_beats_its_components(self, folds):
        """Main effects together must explain at least as much as either alone."""
        train, valid, test = folds
        scores = {}
        for name in ("target_mean", "ligand_mean", "additive_bias"):
            model = build_model(name)
            model.fit(train, valid if len(valid) else None)
            predictions = model.predict(test).mean
            truth = model.targets(test)
            scores[name] = float(np.sqrt(np.mean(np.square(truth - predictions))))

        assert scores["additive_bias"] <= min(scores["target_mean"], scores["ligand_mean"]) + 0.05


class TestConformalIntegration:
    """The shift-robust model's intervals, end to end."""

    @pytest.mark.torch
    def test_intervals_are_produced_and_ordered(self, aggregated):
        pytest.importorskip("torch")
        from leakfree_dta.models.robust.group_dro import GroupDROConfig
        from leakfree_dta.models.robust.shift_robust import ShiftRobustDTA

        assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold"))
        train = assignment.train(aggregated)
        valid = assignment.valid(aggregated)
        test = assignment.test(aggregated)
        if len(valid) < 40:
            pytest.skip("validation fold too small to calibrate on")

        model = ShiftRobustDTA(
            alpha=0.1,
            config=GroupDROConfig(max_epochs=4, hidden_sizes=(32,), patience=4, min_group_size=5),
        )
        model.fit(train, valid)
        prediction = model.predict(test)

        assert prediction.lower is not None and prediction.upper is not None
        assert (prediction.lower <= prediction.mean).all()
        assert (prediction.mean <= prediction.upper).all()
        assert prediction.metadata["target_coverage"] == pytest.approx(0.9)

    @pytest.mark.torch
    def test_calibration_needs_a_validation_fold(self, aggregated):
        pytest.importorskip("torch")
        from leakfree_dta.models.robust.shift_robust import ShiftRobustDTA

        with pytest.raises(ValueError, match="validation fold"):
            ShiftRobustDTA().fit(aggregated, None)


class TestFeatures:
    def test_identical_ligands_get_identical_features(self, aggregated):
        from leakfree_dta.models.features import FeatureSpec, ligand_features

        spec = FeatureSpec(ligand="ecfp4", n_bits=256)
        features = ligand_features(aggregated, spec)
        keys = aggregated[schema.LIGAND_KEY].to_numpy()

        repeated = keys[0]
        rows = np.flatnonzero(keys == repeated)
        if len(rows) < 2:
            pytest.skip("no ligand appears twice in this fixture")
        assert np.array_equal(features[rows[0]], features[rows[1]])

    def test_homologues_share_a_fingerprint(self, synthetic_df, synthetic_truth):
        """The property the whole fixture is built on, asserted directly.

        Homologues differ in InChIKey and in Murcko scaffold but not in ECFP4. That
        is why a scaffold split does not protect against them, and it is the
        mechanism behind the paper's headline comparison.
        """
        from leakfree_dta.chem.fingerprints import ecfp_matrix

        ligands = synthetic_df.drop_duplicates(subset=[schema.LIGAND_KEY])
        analogues = ligands[ligands[schema.LIGAND_KEY].isin(synthetic_truth.analog_ligand_keys)]
        if analogues.empty:
            pytest.skip("no analogues in this fixture")

        bases = ligands[~ligands[schema.LIGAND_KEY].isin(synthetic_truth.analog_ligand_keys)]
        base_features = ecfp_matrix(bases[schema.SMILES])
        analogue_features = ecfp_matrix(analogues[schema.SMILES])

        # At least one analogue must have a bit-identical counterpart in the bases.
        identical = sum(
            1
            for row in analogue_features
            if np.any(np.all(base_features == row, axis=1))
        )
        assert identical > 0, (
            "no planted analogue is fingerprint-identical to a base ligand; the "
            "near-duplicate channel is not being exercised"
        )
