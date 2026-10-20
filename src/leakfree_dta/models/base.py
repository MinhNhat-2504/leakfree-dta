"""The model interface.

Deliberately narrow. Everything a model needs comes in as a curated DataFrame, and
predictions come out as an array aligned to it. That keeps every model in the
comparison -- a one-line target-mean lookup and a published graph network alike --
behind the same contract, so the evaluation code cannot accidentally treat them
differently.

Uncertainty is part of the contract rather than an extension. A model that cannot
report it returns ``None``, and the evaluation records that as a limitation instead
of substituting a default, because "no uncertainty estimate" and "an uncertainty
estimate of zero" are very different claims.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema


@dataclass
class Prediction:
    """A model's output on one test set."""

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        for name in ("std", "lower", "upper"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != self.mean.shape:
                raise ValueError(
                    f"{name} has shape {array.shape} but mean has {self.mean.shape}"
                )
            setattr(self, name, array)

    @property
    def has_uncertainty(self) -> bool:
        return self.std is not None or (self.lower is not None and self.upper is not None)

    @property
    def confidence(self) -> np.ndarray | None:
        """A higher-is-better trust score, for the risk-coverage curve."""
        if self.std is not None:
            return -self.std
        if self.lower is not None and self.upper is not None:
            return -(self.upper - self.lower)
        return None


@dataclass
class FitResult:
    """What fitting produced, beyond the fitted model itself."""

    n_train: int
    n_valid: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class DTAModel(ABC):
    """Base class for every model in the comparison."""

    #: A short identifier used in result tables and filenames.
    name: str = "unnamed"

    #: Whether this model reports calibrated uncertainty. Recorded in the results
    #: table so a reader can see which rows support a selective-prediction claim.
    reports_uncertainty: bool = False

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = hyperparameters
        self._fitted = False

    @abstractmethod
    def fit(
        self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any
    ) -> FitResult:
        """Fit on ``train``, optionally using ``valid`` for model selection.

        A model that uses ``valid`` must use only ``valid`` -- never the test set,
        not even for early stopping. The split machinery builds a validation fold
        for exactly this purpose, and using it is what makes the reported test
        number an honest held-out estimate.
        """

    @abstractmethod
    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        """Predict affinities for ``test``, in row order."""

    def save(self, path: str | Path) -> Path:  # pragma: no cover - model-specific
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    @classmethod
    def load(cls, path: str | Path) -> DTAModel:  # pragma: no cover - model-specific
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    # --- helpers shared by subclasses --------------------------------------

    @staticmethod
    def targets(df: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(df[schema.Y], errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def check_fitted(model: DTAModel) -> None:
        if not model._fitted:
            raise RuntimeError(f"{type(model).__name__} must be fitted before predicting")

    def _mark_fitted(self) -> None:
        self._fitted = True

    def describe(self) -> dict[str, Any]:
        """Everything needed to reproduce this model's configuration."""
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "reports_uncertainty": self.reports_uncertainty,
            "hyperparameters": self.hyperparameters,
        }
