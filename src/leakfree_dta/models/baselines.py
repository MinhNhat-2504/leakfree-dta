"""Baselines.

The ladder runs from "cannot possibly work" to "a competent classical model", and
reading the whole ladder is how the paper establishes what a benchmark is worth:

``GlobalMeanModel``
    One constant. The floor.
``TargetMeanModel``, ``LigandMeanModel``, ``AdditiveBiasModel``
    The shortcut probes from the audit, promoted to first-class models so they
    appear in the main results table rather than only in a diagnostic. Putting
    them in the same table as the deep models is the single most informative
    editorial choice in the paper.
``RidgeBaseline``, ``GradientBoostingBaseline``
    Classical learners on ECFP4 and protein k-mer features. These are the numbers a
    new architecture has to beat to have contributed anything, and on strict splits
    they are often not beaten.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.models.base import DTAModel, FitResult, Prediction
from leakfree_dta.models.features import FeatureSpec, build_features

logger = get_logger(__name__)


class GlobalMeanModel(DTAModel):
    """Predicts the training mean for everything."""

    name = "global_mean"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        y = self.targets(train)
        self._mean = float(np.nanmean(y))
        self._std = float(np.nanstd(y))
        self._mark_fitted()
        return FitResult(n_train=len(train), n_valid=len(valid) if valid is not None else 0)

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        return Prediction(mean=np.full(len(test), self._mean))


class TargetMeanModel(DTAModel):
    """Predicts the target's mean training affinity, ignoring the ligand entirely.

    Unseen targets fall back to the global mean, which is the charitable choice --
    it keeps the model's coverage at 100% so its score is directly comparable to
    the deep models'. The audit's shortcut probe reports the uncharitable version
    (``NaN`` for unseen targets) alongside, so neither reading can mislead.
    """

    name = "target_mean"
    reports_uncertainty = True

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        grouped = train.groupby(schema.TARGET_KEY, observed=True)[schema.Y]
        self._means = grouped.mean()
        self._stds = grouped.std().fillna(0.0)
        self._global_mean = float(np.nanmean(self.targets(train)))
        self._global_std = float(np.nanstd(self.targets(train)))
        self._mark_fitted()
        return FitResult(
            n_train=len(train),
            notes={"n_targets_seen": len(self._means)},
        )

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        mean = test[schema.TARGET_KEY].map(self._means).fillna(self._global_mean).to_numpy(dtype=float)
        # An unseen target gets the global spread, which is wider: the model is
        # genuinely less certain there, and saying so is the whole point.
        std = test[schema.TARGET_KEY].map(self._stds).fillna(self._global_std).to_numpy(dtype=float)
        n_unseen = int((~test[schema.TARGET_KEY].isin(self._means.index)).sum())
        return Prediction(
            mean=mean,
            std=std,
            metadata={"n_unseen_targets": n_unseen, "coverage": 1 - n_unseen / max(len(test), 1)},
        )


class LigandMeanModel(DTAModel):
    """Predicts the ligand's mean training affinity, ignoring the protein entirely."""

    name = "ligand_mean"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        self._means = train.groupby(schema.LIGAND_KEY, observed=True)[schema.Y].mean()
        self._global_mean = float(np.nanmean(self.targets(train)))
        self._mark_fitted()
        return FitResult(n_train=len(train), notes={"n_ligands_seen": len(self._means)})

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        mean = test[schema.LIGAND_KEY].map(self._means).fillna(self._global_mean).to_numpy(dtype=float)
        n_unseen = int((~test[schema.LIGAND_KEY].isin(self._means.index)).sum())
        return Prediction(mean=mean, metadata={"n_unseen_ligands": n_unseen})


class AdditiveBiasModel(DTAModel):
    """Target effect plus ligand effect, with no interaction term.

    A two-way main-effects model, fitted by alternating least squares. It is the
    strongest model that is *structurally incapable* of learning which ligand suits
    which protein, so the gap between it and any real model is the clearest
    available measure of how much interaction modelling the benchmark rewards.
    """

    name = "additive_bias"

    def __init__(self, n_iterations: int = 10, **kwargs: Any) -> None:
        super().__init__(n_iterations=n_iterations, **kwargs)
        self.n_iterations = n_iterations

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        y = self.targets(train)
        self._global = float(np.nanmean(y))
        target_keys = train[schema.TARGET_KEY]
        ligand_keys = train[schema.LIGAND_KEY]

        target_effect = pd.Series(0.0, index=pd.Index(target_keys.unique()))
        ligand_effect = pd.Series(0.0, index=pd.Index(ligand_keys.unique()))

        history: list[dict[str, float]] = []
        for iteration in range(self.n_iterations):
            residual = y - self._global - ligand_keys.map(ligand_effect).to_numpy()
            target_effect = pd.Series(residual, index=train.index).groupby(target_keys.to_numpy()).mean()

            residual = y - self._global - target_keys.map(target_effect).to_numpy()
            ligand_effect = pd.Series(residual, index=train.index).groupby(ligand_keys.to_numpy()).mean()

            fitted = (
                self._global
                + target_keys.map(target_effect).to_numpy()
                + ligand_keys.map(ligand_effect).to_numpy()
            )
            rmse = float(np.sqrt(np.nanmean(np.square(y - fitted))))
            history.append({"iteration": iteration, "train_rmse": rmse})

        self._target_effect = target_effect
        self._ligand_effect = ligand_effect
        self._mark_fitted()
        return FitResult(n_train=len(train), history=history)

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        mean = (
            self._global
            + test[schema.TARGET_KEY].map(self._target_effect).fillna(0.0)
            + test[schema.LIGAND_KEY].map(self._ligand_effect).fillna(0.0)
        ).to_numpy(dtype=float)
        return Prediction(mean=mean)


class RidgeBaseline(DTAModel):
    """Ridge regression on concatenated ligand and protein features.

    The simplest model in the comparison that can use chemistry at all. Its
    regularisation strength is chosen on the validation fold, never on test.
    """

    name = "ridge_ecfp"

    def __init__(
        self,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
        feature_spec: FeatureSpec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(alphas=alphas, **kwargs)
        self.alphas = alphas
        self.spec = feature_spec or FeatureSpec()

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        from sklearn.linear_model import Ridge

        x_train = build_features(train, self.spec)
        y_train = self.targets(train)

        best_alpha, best_score, history = self.alphas[0], np.inf, []
        if valid is not None and len(valid) >= 10:
            x_valid = build_features(valid, self.spec)
            y_valid = self.targets(valid)
            for alpha in self.alphas:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                score = float(np.sqrt(np.mean(np.square(y_valid - model.predict(x_valid)))))
                history.append({"alpha": alpha, "valid_rmse": score})
                if score < best_score:
                    best_alpha, best_score = alpha, score
            logger.info("ridge: selected alpha=%g (valid RMSE %.4f)", best_alpha, best_score)
        else:
            logger.warning("no usable validation fold; using alpha=%g unselected", best_alpha)

        self._model = Ridge(alpha=best_alpha).fit(x_train, y_train)
        self._alpha = best_alpha
        self._mark_fitted()
        return FitResult(
            n_train=len(train),
            n_valid=len(valid) if valid is not None else 0,
            history=history,
            notes={"alpha": best_alpha, "n_features": int(x_train.shape[1])},
        )

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        return Prediction(mean=self._model.predict(build_features(test, self.spec)))


class GradientBoostingBaseline(DTAModel):
    """Gradient-boosted trees on the same features.

    The strongest classical baseline, and in practice the hardest for a deep model
    to beat on tabular molecular features. Included so that "our architecture beats
    DeepDTA" cannot stand in for "our architecture is useful".
    """

    name = "xgboost_ecfp"

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.6,
        feature_spec: FeatureSpec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )
        self.spec = feature_spec or FeatureSpec()

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GradientBoostingBaseline needs xgboost; install the 'gbm' extra"
            ) from exc

        x_train = build_features(train, self.spec)
        y_train = self.targets(train)

        params = dict(self.hyperparameters)
        params.setdefault("tree_method", "hist")
        params.setdefault("random_state", 0)
        model = XGBRegressor(**params, early_stopping_rounds=50 if valid is not None else None)

        eval_set = None
        if valid is not None and len(valid) >= 10:
            eval_set = [(build_features(valid, self.spec), self.targets(valid))]
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

        self._model = model
        self._mark_fitted()
        return FitResult(
            n_train=len(train),
            n_valid=len(valid) if valid is not None else 0,
            best_epoch=getattr(model, "best_iteration", None),
            notes={"n_features": int(x_train.shape[1])},
        )

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        return Prediction(mean=self._model.predict(build_features(test, self.spec)))
