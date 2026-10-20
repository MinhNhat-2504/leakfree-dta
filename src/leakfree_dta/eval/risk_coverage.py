"""Selective prediction: the risk-coverage curve.

A screening group does not need a prediction for every compound. It needs to know
which predictions to trust. So the useful question is not "what is the RMSE over
the whole test set" but "if I act on only the most confident 20%, what error do I
face?"

The risk-coverage curve answers that directly, and its area is a single number
that rewards a model for knowing what it does not know. A model with worse overall
RMSE but honest uncertainty can be strictly more useful than one with better RMSE
and no uncertainty at all -- which is the case the shift-robust baseline makes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from leakfree_dta.eval.metrics import rmse


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    risk: Callable[[np.ndarray, np.ndarray], float] = rmse,
    n_points: int = 50,
) -> pd.DataFrame:
    """Risk as a function of the fraction of predictions retained.

    ``confidence`` is any score where higher means more trustworthy -- the negated
    predictive standard deviation, the negated conformal interval width, or an
    ensemble agreement. Predictions are retained in descending confidence order.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    confidence = np.asarray(confidence, float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(confidence)
    y_true, y_pred, confidence = y_true[finite], y_pred[finite], confidence[finite]

    n = len(y_true)
    if n < 2:
        return pd.DataFrame(columns=["coverage", "n_retained", "risk"])

    order = np.argsort(-confidence)
    y_true, y_pred = y_true[order], y_pred[order]

    coverages = np.linspace(1.0 / n, 1.0, min(n_points, n))
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        keep = max(2, round(coverage * n))
        rows.append(
            {
                "coverage": round(keep / n, 4),
                "n_retained": keep,
                "risk": risk(y_true[:keep], y_pred[:keep]),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["n_retained"]).reset_index(drop=True)


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    """Area under the risk-coverage curve. Lower is better.

    Integrated with the trapezoid rule over coverage, so it is comparable between
    models evaluated on the same test set regardless of how many points each curve
    was sampled at.
    """
    # NumPy renamed trapz to trapezoid in 2.0 and kept the old name deprecated;
    # resolving it here keeps the package working on both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    if curve.empty:
        return float("nan")
    coverage = curve["coverage"].to_numpy(dtype=float)
    risk = curve["risk"].to_numpy(dtype=float)
    finite = np.isfinite(coverage) & np.isfinite(risk)
    if finite.sum() < 2:
        return float("nan")
    return float(trapezoid(risk[finite], coverage[finite]))


def selective_risk_at(curve: pd.DataFrame, coverage: float) -> float:
    """Risk at a given coverage, interpolated.

    The number a screening group asks for: "if I follow up the top 10%, what error
    should I expect?"
    """
    if curve.empty:
        return float("nan")
    coverages = curve["coverage"].to_numpy(dtype=float)
    risks = curve["risk"].to_numpy(dtype=float)
    return float(np.interp(coverage, coverages, risks))


def selective_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.1, 0.2, 0.5, 0.8, 1.0),
) -> dict[str, float]:
    """Risk at a handful of coverages, plus the curve's area.

    This is the table a reader needs to decide whether a model's uncertainty is
    worth anything: if risk at 20% coverage is no better than at 100%, the
    uncertainty estimate carries no information.
    """
    curve = risk_coverage_curve(y_true, y_pred, confidence)
    summary = {f"risk_at_{int(100 * coverage)}pct": round(selective_risk_at(curve, coverage), 5) for coverage in coverages}
    summary["aurc"] = round(area_under_risk_coverage(curve), 5)
    full = selective_risk_at(curve, 1.0)
    tenth = selective_risk_at(curve, 0.1)
    # The ratio is the headline: how much error a group avoids by screening
    # selectively. 1.0 means the uncertainty estimate is useless.
    summary["selectivity_gain"] = round(full / tenth, 4) if tenth and np.isfinite(tenth) else float("nan")
    return summary
