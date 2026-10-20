"""The proposed baseline: group DRO plus a Mondrian conformal head.

The paper's claim about this model is narrow and should stay narrow, because an
overclaim here is what a reviewer will attack:

**It is not claimed to be the most accurate model.** On an average-RMSE
leaderboard it will often lose to a well-tuned ERM model. That is expected: the
worst-group objective trades average accuracy for uniformity, and the trade is
the point.

**It is claimed to be the most *usable* model under shift**, on three measurable
counts:

1. Its worst-group error is lower, so it does not collapse on rare targets.
2. Its intervals hold their stated coverage within each group, not merely on
   average, so a screening group can act on them.
3. Its risk-coverage curve is better, so selecting the most confident predictions
   actually reduces error -- which is what a screening campaign does.

All three are reported with cluster-bootstrap intervals, and the ablation
separates the contribution of the objective from that of the conformal head.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.models.base import DTAModel, FitResult, Prediction
from leakfree_dta.models.features import FeatureSpec
from leakfree_dta.models.robust.conformal import MondrianConformal
from leakfree_dta.models.robust.group_dro import GroupDROConfig, GroupDROModel

logger = get_logger(__name__)


class ShiftRobustDTA(DTAModel):
    """Group-DRO backbone with group-conditional conformal intervals.

    The conformal head calibrates on the validation fold, which the backbone used
    only for early stopping. That reuse is a real caveat -- strict split conformal
    wants a fold the model has not touched at all -- so ``calibration_fraction``
    splits the validation fold in two by default: one part for early stopping, one
    held back purely for calibration. The split is deterministic and recorded.
    """

    name = "shift_robust"
    reports_uncertainty = True

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        group_column: str = "protein_group",
        calibration_fraction: float = 0.5,
        backbone: DTAModel | None = None,
        config: GroupDROConfig | None = None,
        feature_spec: FeatureSpec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            alpha=alpha,
            group_column=group_column,
            calibration_fraction=calibration_fraction,
            **kwargs,
        )
        self.alpha = alpha
        self.group_column = group_column
        self.calibration_fraction = calibration_fraction
        self.backbone = backbone or GroupDROModel(
            config=config, group_column=group_column, feature_spec=feature_spec
        )
        self.conformal = MondrianConformal(alpha=alpha, normalized=True)
        self._calibration_coverage: pd.DataFrame | None = None

    # --- helpers ------------------------------------------------------------

    def _groups(self, df: pd.DataFrame) -> np.ndarray:
        if self.group_column in df.columns:
            return df[self.group_column].astype(str).to_numpy()
        if schema.TARGET_FAMILY in df.columns:
            return df[schema.TARGET_FAMILY].astype(str).to_numpy()
        return df[schema.TARGET_KEY].astype(str).to_numpy()

    def _split_validation(self, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Deterministically halve the validation fold into tuning and calibration.

        Split by *group* rather than by row, for the same reason the main splits
        are: rows within a protein group are not independent, and a row-level split
        would put near-identical measurements on both sides, making the calibration
        set look easier than it is.
        """
        from leakfree_dta.utils.hashing import hash_unit

        groups = self._groups(valid)
        unique = np.unique(groups)
        calibration_groups = {
            group for group in unique if hash_unit(group, salt="shiftrobust|calib") < self.calibration_fraction
        }
        # With very few groups, a group-level halving can leave one side empty.
        if not calibration_groups or len(calibration_groups) == len(unique):
            logger.warning(
                "validation fold has %d groups; splitting calibration by row instead of by group",
                len(unique),
            )
            mask = np.array(
                [hash_unit(key, salt="shiftrobust|calibrow") < self.calibration_fraction for key in valid[schema.ROW_KEY]]
            )
        else:
            mask = np.isin(groups, list(calibration_groups))

        return valid.loc[~mask].reset_index(drop=True), valid.loc[mask].reset_index(drop=True)

    @staticmethod
    def _difficulty(prediction: Prediction, n_rows: int) -> np.ndarray:
        """The signal conformal normalises interval width by.

        The backbone's own predictive standard deviation, which is inflated for
        unseen groups. Conformal prediction does not require this signal to be
        *correct* -- the coverage guarantee holds regardless -- only for it to be
        informative, which is what makes it safe to use a heuristic here.
        """
        if prediction.std is not None:
            return np.asarray(prediction.std, dtype=float)
        return np.ones(n_rows, dtype=float)

    # --- interface ----------------------------------------------------------

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        if valid is None or len(valid) < int(np.ceil(1 / self.alpha)) * 2:
            raise ValueError(
                f"ShiftRobustDTA needs a validation fold of at least "
                f"{int(np.ceil(1 / self.alpha)) * 2} rows for early stopping and calibration; "
                f"got {0 if valid is None else len(valid)}"
            )

        tuning, calibration = self._split_validation(valid)
        logger.info(
            "validation fold split into %d tuning rows and %d calibration rows",
            len(tuning),
            len(calibration),
        )

        backbone_result = self.backbone.fit(train, tuning if len(tuning) >= 10 else valid, **kwargs)

        prediction = self.backbone.predict(calibration)
        self.conformal.calibrate(
            self.targets(calibration),
            prediction.mean,
            self._groups(calibration),
            difficulty=self._difficulty(prediction, len(calibration)),
        )
        # Coverage on the calibration fold itself is optimistic by construction;
        # it is kept only as a sanity check that calibration ran, never reported.
        self._calibration_coverage = self.conformal.evaluate_coverage(
            self.targets(calibration),
            prediction.mean,
            self._groups(calibration),
            difficulty=self._difficulty(prediction, len(calibration)),
        )

        self._mark_fitted()
        return FitResult(
            n_train=backbone_result.n_train,
            n_valid=len(valid),
            history=backbone_result.history,
            best_epoch=backbone_result.best_epoch,
            notes={
                **backbone_result.notes,
                "n_tuning_rows": len(tuning),
                "n_calibration_rows": len(calibration),
                "conformal": self.conformal.calibration.to_dict() if self.conformal.calibration else {},
            },
        )

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        prediction = self.backbone.predict(test, **kwargs)
        difficulty = self._difficulty(prediction, len(test))
        lower, upper, fell_back = self.conformal.intervals(
            prediction.mean, self._groups(test), difficulty=difficulty
        )

        return Prediction(
            mean=prediction.mean,
            # Half the interval width as a standard-deviation-like scale, so the
            # risk-coverage curve can rank predictions by conformal width.
            std=(upper - lower) / 2.0,
            lower=lower,
            upper=upper,
            metadata={
                **prediction.metadata,
                "target_coverage": 1 - self.alpha,
                "fallback_fraction": round(float(fell_back.mean()), 4),
                "uncertainty_source": "mondrian_conformal",
            },
        )

    def coverage_report(self, test: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Per-group empirical coverage on a held-out set.

        The table that supports claim 2. Call it on test data with known labels;
        the result is what the paper reports, unlike the calibration-fold coverage
        which is optimistic by construction.
        """
        prediction = self.backbone.predict(test, **kwargs)
        return self.conformal.evaluate_coverage(
            self.targets(test),
            prediction.mean,
            self._groups(test),
            difficulty=self._difficulty(prediction, len(test)),
        )
