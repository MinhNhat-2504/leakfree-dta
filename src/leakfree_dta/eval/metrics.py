"""Metrics.

Both regression and ranking metrics are reported, because they answer different
questions and can disagree. A model can have low RMSE and be useless for
screening, if its errors happen to be ordered wrongly at the top of the ranking.
Virtual screening cares about the top of the list; lead optimisation cares about
absolute accuracy. The paper reports both and says which is which.

The concordance index is included because it is the metric most DTA papers lead
with, so the re-evaluation has to be comparable to them -- not because it is the
best choice.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class MetricSet:
    """A metric bundle for one model on one split."""

    values: dict[str, float] = field(default_factory=dict)
    n: int = 0

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.values.get(key, default)

    def to_dict(self) -> dict[str, float | int]:
        return {**{key: round(value, 5) for key, value in self.values.items()}, "n": self.n}


# --- regression -------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred).statistic)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(stats.spearmanr(y_true, y_pred).statistic)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the *test* mean.

    Not the squared Pearson correlation. The two differ whenever predictions are
    biased or mis-scaled, and reporting the squared correlation as R^2 flatters a
    model that systematically over- or under-predicts -- which is exactly what
    happens under the distribution shift a leakage-free split creates.
    """
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    if total < 1e-12:
        return 0.0
    residual = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - residual / total


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, *, max_pairs: int = 5_000_000) -> float:
    """The fraction of comparable pairs ranked in the right order.

    Computed exactly for small inputs and by subsampling pairs above
    ``max_pairs``, since the exact form is quadratic. Ties in the prediction count
    as half credit, following the standard definition used in the DTA literature.
    """
    n = len(y_true)
    if n < 2:
        return float("nan")

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        # Vectorised over the upper triangle.
        true_diff = y_true[:, None] - y_true[None, :]
        pred_diff = y_pred[:, None] - y_pred[None, :]
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        comparable = upper & (true_diff != 0)
        if not comparable.any():
            return float("nan")
        concordant = np.sign(true_diff[comparable]) == np.sign(pred_diff[comparable])
        tied = pred_diff[comparable] == 0
        return float((concordant.sum() + 0.5 * tied.sum()) / comparable.sum())

    rng = np.random.default_rng(0)
    left = rng.integers(0, n, size=max_pairs)
    right = rng.integers(0, n, size=max_pairs)
    keep = y_true[left] != y_true[right]
    if not keep.any():
        return float("nan")
    true_diff = y_true[left][keep] - y_true[right][keep]
    pred_diff = y_pred[left][keep] - y_pred[right][keep]
    concordant = np.sign(true_diff) == np.sign(pred_diff)
    tied = pred_diff == 0
    return float((concordant.sum() + 0.5 * tied.sum()) / keep.sum())


# --- screening --------------------------------------------------------------


def enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, *, fraction: float = 0.01, threshold: float = 7.0) -> float:
    """Enrichment of actives in the top ``fraction`` of the predicted ranking.

    ``threshold`` defines an active; 7.0 on the p-scale is 100 nM, a common
    screening cut-off. EF of 1.0 is random, and the maximum attainable value is
    ``1 / fraction`` capped by the number of actives -- so an EF should always be
    read next to the active rate, which :func:`evaluate` reports.
    """
    n = len(y_true)
    n_top = max(1, round(fraction * n))
    is_active = y_true >= threshold
    n_active = int(is_active.sum())
    if n_active == 0 or n == 0:
        return float("nan")

    top = np.argsort(-y_pred)[:n_top]
    hit_rate_top = float(is_active[top].mean())
    hit_rate_all = n_active / n
    return hit_rate_top / hit_rate_all


def bedroc(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float = 20.0, threshold: float = 7.0) -> float:
    """BEDROC: early-recognition-weighted ranking quality.

    ``alpha=20`` puts about 80% of the weight on the top 8% of the ranking, the
    conventional setting. Included because enrichment factor at a single cut-off
    is noisy on small test sets, while BEDROC uses the whole ranking.
    """
    n = len(y_true)
    is_active = y_true >= threshold
    n_active = int(is_active.sum())
    if n_active == 0 or n_active == n:
        return float("nan")

    order = np.argsort(-y_pred)
    ranks = np.flatnonzero(is_active[order]) + 1
    ratio = n_active / n

    sum_exp = float(np.sum(np.exp(-alpha * ranks / n)))
    random_sum = ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    scale = (
        ratio
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ratio))
    )
    offset = 1.0 / (1 - np.exp(alpha * (1 - ratio)))
    return float(sum_exp / random_sum * scale + offset)


# --- calibration ------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, *, n_bins: int = 10
) -> float:
    """Calibration error for a model that reports a predictive standard deviation.

    Bins predictions by their claimed uncertainty and compares the claimed
    standard deviation to the observed root-mean-square error in each bin. A model
    whose claimed uncertainty tracks its real error has an ECE near zero; one that
    is confidently wrong does not.
    """
    finite = np.isfinite(y_std) & (y_std > 0)
    if finite.sum() < n_bins:
        return float("nan")

    true, pred, std = y_true[finite], y_pred[finite], y_std[finite]
    edges = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    error = 0.0
    total = 0
    for low, high in itertools.pairwise(edges):
        mask = (std >= low) & (std <= high)
        if mask.sum() < 2:
            continue
        observed = float(np.sqrt(np.mean(np.square(true[mask] - pred[mask]))))
        claimed = float(np.mean(std[mask]))
        error += mask.sum() * abs(observed - claimed)
        total += int(mask.sum())
    return error / total if total else float("nan")


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values inside the predicted interval."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width. Coverage without width is meaningless -- an interval
    of ``[-inf, inf]`` has perfect coverage and no use."""
    return float(np.mean(upper - lower))


#: Metrics computed for every model on every split.
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": rmse,
    "mae": mae,
    "pearson": pearson,
    "spearman": spearman,
    "r2": r2,
    "ci": concordance_index,
}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_std: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    active_threshold: float = 7.0,
) -> MetricSet:
    """Compute every applicable metric, skipping ones the inputs cannot support."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]

    values: dict[str, float] = {}
    if len(y_true) >= 2:
        for name, function in METRICS.items():
            values[name] = function(y_true, y_pred)
        values["ef_1pct"] = enrichment_factor(y_true, y_pred, fraction=0.01, threshold=active_threshold)
        values["ef_5pct"] = enrichment_factor(y_true, y_pred, fraction=0.05, threshold=active_threshold)
        values["bedroc_20"] = bedroc(y_true, y_pred, alpha=20.0, threshold=active_threshold)
        values["active_rate"] = float(np.mean(y_true >= active_threshold))

    if y_std is not None:
        values["ece"] = expected_calibration_error(y_true, y_pred, np.asarray(y_std, float)[finite])
    if lower is not None and upper is not None:
        lower_f = np.asarray(lower, float)[finite]
        upper_f = np.asarray(upper, float)[finite]
        values["coverage"] = interval_coverage(y_true, lower_f, upper_f)
        values["interval_width"] = interval_width(lower_f, upper_f)

    return MetricSet(values=values, n=len(y_true))
