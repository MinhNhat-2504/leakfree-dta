"""Bootstrap confidence intervals, resampled over groups.

This module implements the paper's most consequential methodological point.

The standard bootstrap resamples rows. That assumes rows are independent. They are
not: a DTA test set contains dozens of measurements of near-identical compounds
against the same target, so resampling rows underestimates the uncertainty --
sometimes by a factor of two or three. Published confidence intervals are
therefore too narrow, and differences reported as significant are often not.

The **cluster bootstrap** resamples whole groups (ligand clusters, or protein
clusters, or both) rather than rows. It is the correct estimator for data with
this dependence structure, and switching to it is enough on its own to make
several published improvements statistically indistinguishable from their
baselines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

DEFAULT_N_BOOTSTRAP = 1000


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a bootstrap interval."""

    metric: str
    estimate: float
    lower: float
    upper: float
    std_error: float
    n_bootstrap: int
    unit: str  # "row" or "group"
    n_units: int

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "estimate": round(self.estimate, 5),
            "ci_lower": round(self.lower, 5),
            "ci_upper": round(self.upper, 5),
            "std_error": round(self.std_error, 5),
            "n_bootstrap": self.n_bootstrap,
            "resample_unit": self.unit,
            "n_units": self.n_units,
        }

    def overlaps(self, other: BootstrapResult) -> bool:
        """Whether two intervals overlap.

        A crude test, and reported as such: non-overlapping intervals imply a
        difference, but overlapping ones do not imply its absence. For a real
        comparison use :func:`compare_models`, which bootstraps the difference.
        """
        return not (self.upper < other.lower or other.upper < self.lower)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str = "metric",
    groups: Sequence | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap a metric, over groups when ``groups`` is given and rows otherwise.

    Passing ``groups`` is strongly preferred. The row bootstrap is retained only so
    the paper can report both and quantify how much narrower the naive interval is.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]

    if groups is not None:
        group_array = np.asarray(groups)[finite]
        return _group_bootstrap(
            y_true, y_pred, metric, metric_name, group_array, n_bootstrap, alpha, seed
        )
    return _row_bootstrap(y_true, y_pred, metric, metric_name, n_bootstrap, alpha, seed)


def _row_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimate = metric(y_true, y_pred) if n >= 2 else float("nan")

    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.integers(0, n, size=n)
        samples[index] = metric(y_true[draw], y_pred[draw])

    return _summarize(metric_name, estimate, samples, alpha, "row", n, n_bootstrap)


def _group_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str,
    groups: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    # Precompute each group's row indices so the resampling loop is cheap.
    indices = {value: np.flatnonzero(groups == value) for value in unique}
    estimate = metric(y_true, y_pred) if len(y_true) >= 2 else float("nan")

    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        drawn = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([indices[unique[position]] for position in drawn])
        samples[index] = metric(y_true[rows], y_pred[rows]) if len(rows) >= 2 else np.nan

    if len(unique) < 20:
        logger.warning(
            "cluster bootstrap over only %d groups; the interval will be unstable. "
            "Report it, but do not draw conclusions from small differences.",
            len(unique),
        )
    return _summarize(metric_name, estimate, samples, alpha, "group", len(unique), n_bootstrap)


def _summarize(
    metric_name: str,
    estimate: float,
    samples: np.ndarray,
    alpha: float,
    unit: str,
    n_units: int,
    n_bootstrap: int,
) -> BootstrapResult:
    finite = samples[np.isfinite(samples)]
    if len(finite) < 2:
        return BootstrapResult(
            metric_name, estimate, float("nan"), float("nan"), float("nan"), n_bootstrap, unit, n_units
        )
    lower, upper = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        metric=metric_name,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        std_error=float(np.std(finite, ddof=1)),
        n_bootstrap=n_bootstrap,
        unit=unit,
        n_units=n_units,
    )


def compare_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str = "metric",
    groups: Sequence | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
    higher_is_better: bool = True,
) -> dict[str, float | bool | str]:
    """Bootstrap the *difference* between two models on the same test set.

    This is the right test for "is model A better than model B", and it is not the
    same as checking whether their individual intervals overlap. Because both
    models are evaluated on the same resample, the shared test-set variance
    cancels, and the paired interval is substantially tighter -- so this test is
    both more correct and more powerful than the comparison it replaces.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred_a) & np.isfinite(y_pred_b)
    y_true, y_pred_a, y_pred_b = y_true[finite], y_pred_a[finite], y_pred_b[finite]

    observed = metric(y_true, y_pred_a) - metric(y_true, y_pred_b)
    rng = np.random.default_rng(seed)

    if groups is not None:
        group_array = np.asarray(groups)[finite]
        unique = np.unique(group_array)
        indices = {value: np.flatnonzero(group_array == value) for value in unique}

        def draw() -> np.ndarray:
            drawn = rng.integers(0, len(unique), size=len(unique))
            return np.concatenate([indices[unique[position]] for position in drawn])

        n_units, unit = len(unique), "group"
    else:
        n = len(y_true)

        def draw() -> np.ndarray:
            return rng.integers(0, n, size=n)

        n_units, unit = n, "row"

    differences = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        rows = draw()
        if len(rows) < 2:
            differences[index] = np.nan
            continue
        differences[index] = metric(y_true[rows], y_pred_a[rows]) - metric(y_true[rows], y_pred_b[rows])

    valid = differences[np.isfinite(differences)]
    lower, upper = (np.quantile(valid, [alpha / 2, 1 - alpha / 2]) if len(valid) > 1 else (np.nan, np.nan))
    # Two-sided bootstrap p-value: how often the resampled difference crosses zero.
    p_value = (
        float(2 * min((valid <= 0).mean(), (valid >= 0).mean())) if len(valid) else float("nan")
    )
    significant = bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0))

    return {
        "metric": metric_name,
        "difference": round(float(observed), 5),
        "ci_lower": round(float(lower), 5),
        "ci_upper": round(float(upper), 5),
        "p_value": round(p_value, 5),
        "significant": significant,
        "a_is_better": bool(observed > 0) == higher_is_better if significant else False,
        "resample_unit": unit,
        "n_units": n_units,
    }


def bootstrap_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    groups: Sequence | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """A model-by-metric table of point estimates with intervals.

    The shape of the paper's main results table: every cell is an estimate and an
    interval, never a bare number.
    """
    rows: list[dict[str, object]] = []
    for model_name, y_pred in predictions.items():
        for metric_name, function in metrics.items():
            result = bootstrap_metric(
                y_true,
                y_pred,
                function,
                metric_name=metric_name,
                groups=groups,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append({"model": model_name, **result.to_dict()})
    return pd.DataFrame(rows)
