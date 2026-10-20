"""Shortcut probes: how much of a benchmark is solvable without modelling interaction.

This is the audit's sharpest instrument, and it needs no deep model at all.

A drug-target affinity model is supposed to learn how a *particular* ligand
interacts with a *particular* protein. So fit predictors that are structurally
incapable of learning that, and see how far they get:

``global_mean``
    One number for everything. The floor.
``target_mean``
    The mean training affinity of the target. Ignores the ligand completely.
``ligand_mean``
    The mean training affinity of the ligand. Ignores the protein completely.
``additive_bias``
    Target effect plus ligand effect, no interaction term -- a two-way ANOVA main-
    effects model.

None of these can distinguish two ligands against the same target, or rank one
ligand across targets. If ``target_mean`` reaches Pearson 0.70 on a benchmark
where published models report 0.85, then the modelling contribution is the gap
between them, not the headline number. On several widely used splits that gap is
small, and the field has been reporting the shortcut as progress.

The probe is the quantitative version of the point raised in arXiv 2311.12670
about transductive DTI evaluation, and it generalises it: it measures the shortcut
directly rather than inferring it from the split's construction, which means it
applies to any split, including ones whose construction is undocumented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from leakfree_dta.audit.report import CheckResult, Severity
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ShortcutScore:
    """One degenerate predictor's performance on the test split."""

    name: str
    rmse: float
    pearson: float
    spearman: float
    n_predicted: int
    n_total: int

    @property
    def coverage(self) -> float:
        """Fraction of test rows the predictor could score at all.

        A target-mean predictor cannot score a test row whose target is absent
        from training. Low coverage is itself informative: it means the split
        genuinely holds out proteins.
        """
        return self.n_predicted / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "name": self.name,
            "rmse": round(self.rmse, 4),
            "pearson": round(self.pearson, 4),
            "spearman": round(self.spearman, 4),
            "coverage": round(self.coverage, 4),
            "n_predicted": self.n_predicted,
        }


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray, n_total: int) -> ShortcutScore:
    """Score a predictor on the rows where it produced a prediction."""
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    if finite.sum() < 3:
        return ShortcutScore(name, float("nan"), float("nan"), float("nan"), int(finite.sum()), n_total)

    true, pred = y_true[finite], y_pred[finite]
    rmse = float(np.sqrt(np.mean(np.square(true - pred))))

    # A constant prediction has zero variance, so correlation is undefined rather
    # than zero. Reporting 0.0 is the honest reading: it carries no ranking signal.
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return ShortcutScore(name, rmse, 0.0, 0.0, int(finite.sum()), n_total)

    pearson = float(stats.pearsonr(true, pred).statistic)
    spearman = float(stats.spearmanr(true, pred).statistic)
    return ShortcutScore(name, rmse, pearson, spearman, int(finite.sum()), n_total)


def shortcut_scores(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, ShortcutScore]:
    """Fit and score every degenerate predictor.

    Predictors are fitted on ``train`` only. Rows whose target or ligand is unseen
    get ``NaN`` rather than a fallback, so that coverage is measured honestly;
    :func:`check_shortcut_predictability` reports coverage alongside the score.
    """
    y_train = pd.to_numeric(train[schema.Y], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[schema.Y], errors="coerce").to_numpy(dtype=float)
    n_total = len(test)
    global_mean = float(np.nanmean(y_train)) if np.isfinite(y_train).any() else float("nan")

    target_means = train.groupby(schema.TARGET_KEY, observed=True)[schema.Y].mean()
    ligand_means = train.groupby(schema.LIGAND_KEY, observed=True)[schema.Y].mean()

    scores: dict[str, ShortcutScore] = {}

    scores["global_mean"] = _score(
        "global_mean", y_test, np.full(n_total, global_mean, dtype=float), n_total
    )
    scores["target_mean"] = _score(
        "target_mean",
        y_test,
        test[schema.TARGET_KEY].map(target_means).to_numpy(dtype=float),
        n_total,
    )
    scores["ligand_mean"] = _score(
        "ligand_mean",
        y_test,
        test[schema.LIGAND_KEY].map(ligand_means).to_numpy(dtype=float),
        n_total,
    )

    # Additive main effects: global + (target effect) + (ligand effect). Fitted by
    # one alternating pass, which is enough for a diagnostic and avoids depending
    # on a solver.
    target_effect = target_means - global_mean
    residual = train[schema.Y] - global_mean - train[schema.TARGET_KEY].map(target_effect).fillna(0.0)
    ligand_effect = residual.groupby(train[schema.LIGAND_KEY], observed=True).mean()
    additive = (
        global_mean
        + test[schema.TARGET_KEY].map(target_effect).fillna(0.0)
        + test[schema.LIGAND_KEY].map(ligand_effect).fillna(0.0)
    ).to_numpy(dtype=float)
    # Only count rows where at least one effect was actually known.
    known = (
        test[schema.TARGET_KEY].isin(target_effect.index) | test[schema.LIGAND_KEY].isin(ligand_effect.index)
    ).to_numpy()
    additive = np.where(known, additive, np.nan)
    scores["additive_bias"] = _score("additive_bias", y_test, additive, n_total)

    return scores


def check_shortcut_predictability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    reference_pearson: float | None = None,
) -> CheckResult:
    """Report what the degenerate predictors achieve, and what that implies.

    ``reference_pearson`` is the performance a published model reports on this
    split. When given, the check converts the shortcut into the number that
    matters: the share of the reported correlation that needed no interaction
    modelling at all.
    """
    scores = shortcut_scores(train, test)
    best_name, best = max(
        ((name, score) for name, score in scores.items() if np.isfinite(score.pearson)),
        key=lambda item: item[1].pearson,
        default=("none", ShortcutScore("none", float("nan"), float("nan"), float("nan"), 0, len(test))),
    )
    floor = scores["global_mean"].pearson

    statistics: dict[str, object] = {
        "scores": {name: score.to_dict() for name, score in scores.items()},
        "best_shortcut": best_name,
        "best_shortcut_pearson": None if not np.isfinite(best.pearson) else round(best.pearson, 4),
        "best_shortcut_coverage": round(best.coverage, 4),
    }

    summary_parts = [
        f"Predictors that cannot model ligand-protein interaction reach Pearson "
        f"{best.pearson:.3f} on this split (`{best_name}`, covering {best.coverage:.1%} of test rows). "
        f"The constant-prediction floor is {floor:.3f}."
    ]

    severity = Severity.INFO
    if reference_pearson is not None and np.isfinite(best.pearson):
        denominator = reference_pearson - floor
        share = (best.pearson - floor) / denominator if abs(denominator) > 1e-9 else float("nan")
        statistics["reference_pearson"] = reference_pearson
        statistics["shortcut_share_of_reported"] = None if not np.isfinite(share) else round(share, 4)
        summary_parts.append(
            f"Against a reported Pearson of {reference_pearson:.3f}, "
            f"{share:.1%} of the gain over the floor is reachable without any interaction model."
        )
        if np.isfinite(share) and share > 0.5:
            severity = Severity.CRITICAL
        elif np.isfinite(share) and share > 0.3:
            severity = Severity.HIGH
    elif np.isfinite(best.pearson) and best.pearson > 0.5:
        severity = Severity.HIGH
        summary_parts.append(
            "A shortcut above 0.5 means most of the apparent signal is per-target or "
            "per-ligand bias, not interaction."
        )

    details = pd.DataFrame([score.to_dict() for score in scores.values()])

    # n_affected is the count of test rows a shortcut can already score, which is
    # the population over which the shortcut operates.
    return CheckResult(
        name="shortcut_predictability",
        axis="shortcut",
        severity=severity,
        n_affected=best.n_predicted,
        n_total=len(test),
        summary=" ".join(summary_parts),
        statistics=statistics,
        details=details,
        tolerance=1.0 if reference_pearson is None else 0.0,
    )


def check_target_memorization(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Per-target shortcut strength, to find which targets carry the leak.

    An aggregate shortcut number hides that it is usually driven by a handful of
    heavily measured targets. Reporting it per target tells a benchmark maintainer
    exactly which targets to hold out.
    """
    target_means = train.groupby(schema.TARGET_KEY, observed=True)[schema.Y].mean()
    test = test.copy()
    test["_pred"] = test[schema.TARGET_KEY].map(target_means)

    records: list[dict[str, object]] = []
    for target, group in test.groupby(schema.TARGET_KEY, observed=True):
        y_true = pd.to_numeric(group[schema.Y], errors="coerce").to_numpy(dtype=float)
        y_pred = group["_pred"].to_numpy(dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.sum() < 5:
            continue
        rmse = float(np.sqrt(np.mean(np.square(y_true[finite] - y_pred[finite]))))
        records.append(
            {
                schema.TARGET_KEY: target,
                "n_test_rows": int(finite.sum()),
                "target_mean_rmse": round(rmse, 4),
                "test_y_std": round(float(np.std(y_true[finite], ddof=1)), 4) if finite.sum() > 1 else 0.0,
            }
        )

    if not records:
        return CheckResult(
            name="target_memorization",
            axis="shortcut",
            severity=Severity.INFO,
            n_affected=0,
            n_total=len(test),
            summary="No target had enough test rows to assess per-target shortcut strength.",
            statistics={"skipped": True},
            tolerance=1.0,
        )

    per_target = pd.DataFrame.from_records(records)
    # A target whose spread is smaller than the overall noise is effectively a
    # constant, and predicting its mean is "correct" for reasons unrelated to skill.
    per_target["shortcut_strength"] = (
        1.0 - per_target["target_mean_rmse"] / per_target["test_y_std"].clip(lower=1e-6)
    ).clip(lower=0.0)
    strong = per_target[per_target["shortcut_strength"] > 0.5]
    n_affected = int(per_target.loc[per_target.index.isin(strong.index), "n_test_rows"].sum())

    return CheckResult(
        name="target_memorization",
        axis="shortcut",
        severity=Severity.MEDIUM if len(strong) else Severity.INFO,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"For {len(strong):,} of {len(per_target):,} assessable targets, predicting the "
            f"training mean already explains over half the test spread "
            f"({n_affected:,} test rows). These targets contribute score without "
            "requiring any ligand information."
        ),
        statistics={
            "n_assessable_targets": len(per_target),
            "n_strong_shortcut_targets": len(strong),
            "median_shortcut_strength": round(float(per_target["shortcut_strength"].median()), 4),
        },
        details=per_target.sort_values("shortcut_strength", ascending=False),
        tolerance=1.0,
    )


#: Every shortcut probe.
SHORTCUT_CHECKS = (check_shortcut_predictability, check_target_memorization)
