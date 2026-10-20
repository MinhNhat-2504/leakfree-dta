"""Mondrian conformal regression.

Split conformal prediction turns any point predictor into an interval predictor
with a finite-sample coverage guarantee, and it needs nothing from the underlying
model -- no retraining, no assumption about its form. The guarantee is: given
exchangeable calibration and test data, the interval covers the truth at least
``1 - alpha`` of the time.

The **Mondrian** variant conditions that guarantee on a group label. This matters
more here than in most applications. A DTA test set is dominated by a few
heavily measured targets, so a globally calibrated interval is tuned to them and
systematically wrong for everything else -- too wide where there is plenty of
data and far too narrow where there is little, which is exactly where a screening
group needs the warning.

Two limitations are stated rather than glossed, because a reviewer will raise them:

* **Exchangeability fails under a temporal or protein-cluster split.** That is the
  point of the split, so the marginal guarantee is not formally valid there. What
  the paper reports is the *empirical* coverage on each split, which is the
  honest claim: conformal intervals degrade gracefully under shift where a
  Gaussian error model degrades catastrophically, and the degradation is measured.
* **Groups need enough calibration points.** With ``n`` calibration points a group
  can only express quantiles at resolution ``1/(n+1)``, so 90% coverage needs at
  least 9. Groups below the floor fall back to the global quantile, and the
  fallback is counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: A group needs at least this many calibration points to get its own quantile.
#: ``ceil(1/alpha) - 1`` is the mathematical floor; 20 is a practical one that
#: keeps the quantile from being set by a single outlier.
MIN_CALIBRATION_PER_GROUP = 20


@dataclass
class ConformalCalibration:
    """The fitted quantiles, per group and globally."""

    alpha: float
    global_quantile: float
    group_quantiles: dict[str, float] = field(default_factory=dict)
    group_sizes: dict[str, int] = field(default_factory=dict)
    n_calibration: int = 0
    normalized: bool = False

    @property
    def n_groups_calibrated(self) -> int:
        return len(self.group_quantiles)

    def quantile_for(self, group: str) -> tuple[float, bool]:
        """The quantile for ``group``, and whether it fell back to the global one."""
        if group in self.group_quantiles:
            return self.group_quantiles[group], False
        return self.global_quantile, True

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "target_coverage": round(1 - self.alpha, 4),
            "global_quantile": round(self.global_quantile, 5),
            "n_calibration": self.n_calibration,
            "n_groups_calibrated": self.n_groups_calibrated,
            "normalized": self.normalized,
            "median_group_quantile": (
                round(float(np.median(list(self.group_quantiles.values()))), 5)
                if self.group_quantiles
                else None
            ),
        }


class MondrianConformal:
    """Group-conditional split conformal regression.

    Usage is deliberately two-step and mirrors the data the pipeline has: calibrate
    on the validation fold, then predict intervals on test. The calibration fold
    must not have been used to fit the underlying model, or the intervals will be
    too narrow -- the most common way conformal prediction is got wrong.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        normalized: bool = True,
        min_per_group: int = MIN_CALIBRATION_PER_GROUP,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.normalized = normalized
        self.min_per_group = min_per_group
        self.calibration: ConformalCalibration | None = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray | pd.Series,
        *,
        difficulty: np.ndarray | None = None,
    ) -> ConformalCalibration:
        """Fit the quantiles on held-out calibration data.

        ``difficulty`` enables **normalised** conformal prediction: the
        nonconformity score becomes ``|residual| / difficulty`` instead of
        ``|residual|``, which lets interval width vary within a group. Any
        higher-means-harder signal works -- the model's own predictive standard
        deviation, the distance to the nearest training ligand, or the size of the
        protein cluster. Without it, every member of a group gets the same width,
        which wastes most of what conformal prediction can do.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        groups = np.asarray(groups)

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        residuals = np.abs(y_true - y_pred)

        use_normalized = self.normalized and difficulty is not None
        if use_normalized:
            difficulty = np.asarray(difficulty, dtype=float)
            # Guard against zero or non-finite difficulty, which would produce
            # infinite scores and swallow the quantile.
            floor = np.nanpercentile(difficulty[np.isfinite(difficulty) & (difficulty > 0)], 5) if np.isfinite(difficulty).any() else 1.0
            difficulty = np.where(np.isfinite(difficulty) & (difficulty > 0), difficulty, floor)
            difficulty = np.maximum(difficulty, max(floor, 1e-6))
            scores = residuals / difficulty
            finite &= np.isfinite(scores)
        else:
            scores = residuals

        scores, groups = scores[finite], groups[finite]
        if len(scores) < int(np.ceil(1 / self.alpha)):
            raise ValueError(
                f"need at least {int(np.ceil(1 / self.alpha))} calibration points for "
                f"alpha={self.alpha}, got {len(scores)}"
            )

        global_quantile = _conformal_quantile(scores, self.alpha)

        group_quantiles: dict[str, float] = {}
        group_sizes: dict[str, int] = {}
        for group in np.unique(groups):
            mask = groups == group
            size = int(mask.sum())
            group_sizes[str(group)] = size
            if size >= max(self.min_per_group, int(np.ceil(1 / self.alpha))):
                group_quantiles[str(group)] = _conformal_quantile(scores[mask], self.alpha)

        n_fallback = len(group_sizes) - len(group_quantiles)
        if n_fallback:
            logger.info(
                "%d of %d groups have under %d calibration points and will use the global "
                "quantile; their coverage guarantee is marginal, not group-conditional",
                n_fallback,
                len(group_sizes),
                self.min_per_group,
            )

        self.calibration = ConformalCalibration(
            alpha=self.alpha,
            global_quantile=global_quantile,
            group_quantiles=group_quantiles,
            group_sizes=group_sizes,
            n_calibration=len(scores),
            normalized=use_normalized,
        )
        logger.info("conformal calibration: %s", self.calibration.to_dict())
        return self.calibration

    def intervals(
        self,
        y_pred: np.ndarray,
        groups: np.ndarray | pd.Series,
        *,
        difficulty: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lower bound, upper bound, and a per-row flag for global-quantile fallback."""
        if self.calibration is None:
            raise RuntimeError("calibrate() must be called before intervals()")

        y_pred = np.asarray(y_pred, dtype=float)
        groups = np.asarray(groups)

        widths = np.empty(len(y_pred), dtype=float)
        fell_back = np.zeros(len(y_pred), dtype=bool)
        for index, group in enumerate(groups):
            quantile, fallback = self.calibration.quantile_for(str(group))
            widths[index] = quantile
            fell_back[index] = fallback

        if self.calibration.normalized:
            if difficulty is None:
                raise ValueError(
                    "this calibration is normalised; intervals() needs the same difficulty signal"
                )
            difficulty = np.asarray(difficulty, dtype=float)
            difficulty = np.where(np.isfinite(difficulty) & (difficulty > 0), difficulty, 1.0)
            widths = widths * difficulty

        return y_pred - widths, y_pred + widths, fell_back

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray | pd.Series,
        *,
        difficulty: np.ndarray | None = None,
        min_group_size: int = 10,
    ) -> pd.DataFrame:
        """Empirical coverage and mean width, overall and per group.

        This table is the paper's evidence that group-conditional calibration was
        worth doing: the overall row looks the same for global and Mondrian
        calibration, and the per-group rows do not.
        """
        lower, upper, fell_back = self.intervals(y_pred, groups, difficulty=difficulty)
        y_true = np.asarray(y_true, dtype=float)
        groups = np.asarray(groups)

        frame = pd.DataFrame(
            {
                "group": groups,
                "covered": (y_true >= lower) & (y_true <= upper),
                "width": upper - lower,
                "fell_back": fell_back,
            }
        )

        rows: list[dict[str, object]] = [
            {
                "group": "__overall__",
                "n": len(frame),
                "coverage": round(float(frame["covered"].mean()), 4),
                "mean_width": round(float(frame["width"].mean()), 4),
                "fallback_fraction": round(float(frame["fell_back"].mean()), 4),
            }
        ]
        for group, chunk in frame.groupby("group", observed=True):
            if len(chunk) < min_group_size:
                continue
            rows.append(
                {
                    "group": group,
                    "n": len(chunk),
                    "coverage": round(float(chunk["covered"].mean()), 4),
                    "mean_width": round(float(chunk["width"].mean()), 4),
                    "fallback_fraction": round(float(chunk["fell_back"].mean()), 4),
                }
            )

        out = pd.DataFrame(rows)
        target = 1 - self.alpha
        groups_only = out[out["group"] != "__overall__"]
        if len(groups_only):
            # The number that distinguishes Mondrian from global calibration.
            worst = float(groups_only["coverage"].min())
            logger.info(
                "coverage: overall %.3f (target %.3f), worst group %.3f over %d groups",
                float(out.iloc[0]["coverage"]),
                target,
                worst,
                len(groups_only),
            )
        return out


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected ``1 - alpha`` quantile of the scores.

    The correction ``ceil((n + 1)(1 - alpha)) / n`` is what makes the coverage
    guarantee exact rather than asymptotic. Using ``np.quantile(scores, 1 - alpha)``
    instead -- the common mistake -- undercovers, by an amount that matters most
    exactly where calibration sets are small.
    """
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))
