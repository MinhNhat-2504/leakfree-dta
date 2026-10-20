"""Metrics, bootstrap intervals, and conformal coverage.

The conformal tests check a mathematical guarantee rather than a plausible number,
which makes them the sharpest tests here: split conformal prediction must cover at
the nominal rate on exchangeable data, so a coverage of 0.75 at a nominal 0.90 is
a bug, not a modelling outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from leakfree_dta.eval import bootstrap, metrics


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.linspace(4, 10, 50)
        result = metrics.evaluate(y, y.copy())
        assert result["rmse"] == pytest.approx(0.0)
        assert result["pearson"] == pytest.approx(1.0)
        assert result["r2"] == pytest.approx(1.0)
        assert result["ci"] == pytest.approx(1.0)

    def test_constant_prediction_has_no_correlation(self):
        y = np.linspace(4, 10, 50)
        result = metrics.evaluate(y, np.full_like(y, y.mean()))
        assert result["pearson"] == 0.0
        # R^2 against the test mean is exactly zero for a mean predictor, which is
        # the property that makes it a more honest metric than squared Pearson.
        assert result["r2"] == pytest.approx(0.0, abs=1e-9)

    def test_r2_differs_from_squared_pearson_under_bias(self):
        """The distinction the metrics module is explicit about.

        A prediction that is perfectly correlated but offset has squared Pearson 1
        and negative R^2. Reporting the former as R^2 would hide systematic bias --
        exactly the bias a leakage-free split induces.
        """
        y = np.linspace(4, 10, 50)
        biased = y + 2.0
        result = metrics.evaluate(y, biased)
        assert result["pearson"] == pytest.approx(1.0)
        assert result["r2"] < 0.0

    def test_concordance_index_of_reversed_ranking(self):
        y = np.linspace(4, 10, 40)
        assert metrics.concordance_index(y, -y) == pytest.approx(0.0)

    def test_enrichment_factor_bounds(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(4, 10, 500)
        # A perfect ranking achieves the maximum attainable enrichment.
        perfect = metrics.enrichment_factor(y, y, fraction=0.1, threshold=7.0)
        random_order = metrics.enrichment_factor(y, rng.permutation(y), fraction=0.1, threshold=7.0)
        assert perfect > random_order
        assert perfect <= 1 / 0.1 + 1e-9

    def test_interval_coverage(self):
        y = np.zeros(100)
        assert metrics.interval_coverage(y, y - 1, y + 1) == 1.0
        assert metrics.interval_coverage(y, y + 1, y + 2) == 0.0


class TestBootstrap:
    def test_cluster_bootstrap_is_wider_than_row_bootstrap(self):
        """The paper's methodological point, as a test.

        With strong within-group correlation, resampling rows understates the
        uncertainty. The cluster interval must come out wider -- if it does not,
        the estimator is not doing what the paper claims.
        """
        rng = np.random.default_rng(0)
        n_groups, per_group = 12, 40
        groups = np.repeat(np.arange(n_groups), per_group)
        # A large per-group offset makes rows within a group highly dependent.
        group_offset = rng.normal(0, 1.5, n_groups)[groups]
        y_true = group_offset + rng.normal(0, 0.2, n_groups * per_group)
        y_pred = group_offset + rng.normal(0, 0.4, n_groups * per_group)

        row = bootstrap.bootstrap_metric(y_true, y_pred, metrics.rmse, n_bootstrap=400)
        cluster = bootstrap.bootstrap_metric(
            y_true, y_pred, metrics.rmse, groups=groups, n_bootstrap=400
        )

        assert cluster.unit == "group"
        assert (cluster.upper - cluster.lower) > (row.upper - row.lower), (
            f"cluster interval {cluster.upper - cluster.lower:.4f} is not wider than "
            f"row interval {row.upper - row.lower:.4f}"
        )

    def test_interval_contains_the_estimate(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(7, 1, 300)
        y_pred = y_true + rng.normal(0, 0.5, 300)
        result = bootstrap.bootstrap_metric(y_true, y_pred, metrics.rmse, n_bootstrap=300)
        assert result.lower <= result.estimate <= result.upper

    def test_identical_models_are_not_significantly_different(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(7, 1, 300)
        y_pred = y_true + rng.normal(0, 0.5, 300)
        result = bootstrap.compare_models(
            y_true, y_pred, y_pred.copy(), metrics.rmse, n_bootstrap=300
        )
        assert result["difference"] == pytest.approx(0.0)
        assert not result["significant"]

    def test_clearly_better_model_is_detected(self):
        rng = np.random.default_rng(3)
        y_true = rng.normal(7, 1, 400)
        good = y_true + rng.normal(0, 0.2, 400)
        bad = y_true + rng.normal(0, 1.2, 400)
        result = bootstrap.compare_models(
            y_true, good, bad, metrics.rmse, n_bootstrap=400, higher_is_better=False
        )
        assert result["significant"]
        assert result["difference"] < 0  # lower RMSE for the good model


class TestConformal:
    """The coverage guarantee, which is a theorem and therefore testable exactly."""

    @staticmethod
    def _data(n: int = 2000, n_groups: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        groups = rng.integers(0, n_groups, n)
        # Group-dependent noise scale: the situation global calibration gets wrong.
        scale = (0.2 + 0.5 * groups).astype(float)
        y_pred = rng.normal(7, 1, n)
        y_true = y_pred + rng.normal(0, scale)
        return y_true, y_pred, groups.astype(str), scale

    def test_marginal_coverage_meets_the_guarantee(self):
        """Coverage must be at least nominal, and not wastefully above it.

        Split conformal prediction guarantees coverage *at least* ``1 - alpha``;
        the guarantee is one-sided, so mild over-coverage is correct behaviour and
        not a bug. The finite-sample correction and the conservative quantile
        method both push slightly in that direction. What would be a bug is
        under-coverage, or over-coverage so large that the intervals are useless.
        """
        from leakfree_dta.models.robust.conformal import MondrianConformal

        y_true, y_pred, groups, _ = self._data()
        half = len(y_true) // 2
        conformal = MondrianConformal(alpha=0.1, normalized=False)
        conformal.calibrate(y_true[:half], y_pred[:half], groups[:half])

        lower, upper, _ = conformal.intervals(y_pred[half:], groups[half:])
        coverage = float(((y_true[half:] >= lower) & (y_true[half:] <= upper)).mean())

        assert coverage >= 0.88, (
            f"nominal coverage 0.90, achieved only {coverage:.3f} -- the guarantee is violated"
        )
        assert coverage <= 0.97, (
            f"coverage {coverage:.3f} is far above nominal; the intervals are too wide to be useful"
        )

    def test_group_conditional_coverage_beats_global(self):
        """The reason the Mondrian variant is used at all.

        With group-dependent noise, a globally calibrated interval is far too
        narrow for the noisy groups and too wide for the quiet ones. Both achieve
        the same marginal coverage; only the Mondrian one achieves it per group.
        """
        from leakfree_dta.models.robust.conformal import MondrianConformal

        y_true, y_pred, groups, _ = self._data(n=4000)
        half = len(y_true) // 2

        def worst_group_gap(min_per_group: int) -> float:
            conformal = MondrianConformal(
                alpha=0.1, normalized=False, min_per_group=min_per_group
            )
            conformal.calibrate(y_true[:half], y_pred[:half], groups[:half])
            coverage = conformal.evaluate_coverage(
                y_true[half:], y_pred[half:], groups[half:]
            )
            per_group = coverage[coverage["group"] != "__overall__"]
            return float((per_group["coverage"] - 0.9).abs().max())

        mondrian_gap = worst_group_gap(min_per_group=20)
        # An impossibly high floor forces every group onto the global quantile,
        # which is exactly global calibration.
        global_gap = worst_group_gap(min_per_group=10**9)

        assert mondrian_gap < global_gap, (
            f"group-conditional worst-group gap {mondrian_gap:.3f} is no better than "
            f"global {global_gap:.3f}"
        )

    def test_finite_sample_correction_is_applied(self):
        """The corrected quantile must be at least the naive one.

        Using ``np.quantile(scores, 1 - alpha)`` undercovers on small calibration
        sets. The correction is what makes the guarantee exact.
        """
        from leakfree_dta.models.robust.conformal import _conformal_quantile

        rng = np.random.default_rng(0)
        scores = np.abs(rng.normal(0, 1, 30))
        corrected = _conformal_quantile(scores, 0.1)
        naive = float(np.quantile(scores, 0.9))
        assert corrected >= naive

    def test_too_few_calibration_points_is_refused(self):
        from leakfree_dta.models.robust.conformal import MondrianConformal

        conformal = MondrianConformal(alpha=0.1, normalized=False)
        with pytest.raises(ValueError, match="calibration points"):
            conformal.calibrate(np.zeros(5), np.zeros(5), np.array(["a"] * 5))


class TestRiskCoverage:
    def test_informative_uncertainty_lowers_risk_at_low_coverage(self):
        from leakfree_dta.eval.risk_coverage import selective_prediction_summary

        rng = np.random.default_rng(0)
        n = 1000
        scale = rng.uniform(0.1, 2.0, n)
        y_pred = rng.normal(7, 1, n)
        y_true = y_pred + rng.normal(0, scale)

        # Confidence is the true (negated) noise scale, so selection must help.
        summary = selective_prediction_summary(y_true, y_pred, -scale)
        assert summary["risk_at_10pct"] < summary["risk_at_100pct"]
        assert summary["selectivity_gain"] > 1.0

    def test_uninformative_uncertainty_gives_no_gain(self):
        from leakfree_dta.eval.risk_coverage import selective_prediction_summary

        rng = np.random.default_rng(0)
        n = 1000
        y_pred = rng.normal(7, 1, n)
        y_true = y_pred + rng.normal(0, 0.5, n)
        summary = selective_prediction_summary(y_true, y_pred, rng.normal(0, 1, n))
        assert summary["selectivity_gain"] == pytest.approx(1.0, abs=0.35)
