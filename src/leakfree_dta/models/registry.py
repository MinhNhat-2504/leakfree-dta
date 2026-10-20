"""Model registry.

The CLI resolves ``--model`` through this mapping, which means the set of models in
the paper's results table is defined in one visible place rather than scattered
across experiment scripts. Adding a model to the comparison is one entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from leakfree_dta.models.base import DTAModel
from leakfree_dta.models.baselines import (
    AdditiveBiasModel,
    GlobalMeanModel,
    GradientBoostingBaseline,
    LigandMeanModel,
    RidgeBaseline,
    TargetMeanModel,
)

#: Name -> constructor. Names appear verbatim in result filenames and tables.
MODELS: dict[str, Callable[..., DTAModel]] = {
    # Degenerate predictors. In the main table on purpose: they are the reference
    # against which any claim of progress has to be read.
    "global_mean": GlobalMeanModel,
    "target_mean": TargetMeanModel,
    "ligand_mean": LigandMeanModel,
    "additive_bias": AdditiveBiasModel,
    # Classical learners.
    "ridge_ecfp": RidgeBaseline,
    "xgboost_ecfp": GradientBoostingBaseline,
}


def _register_deep_models() -> None:
    """Add the torch-dependent models when torch is available.

    Registered lazily so that ``leakfree audit`` and ``leakfree split`` work on a
    machine without PyTorch. A missing deep model is reported as an absent row in
    the results table rather than an import error at startup.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return

    from leakfree_dta.models.deepdta import DeepDTAModel
    from leakfree_dta.models.robust.group_dro import ERMModel, GroupDROModel
    from leakfree_dta.models.robust.shift_robust import ShiftRobustDTA

    MODELS["erm_mlp"] = ERMModel
    MODELS["group_dro_mlp"] = GroupDROModel
    MODELS["shift_robust"] = ShiftRobustDTA
    # A reimplementation, not the published code. The name carries the suffix so
    # that no results table can present it as a re-run of the original.
    MODELS["deepdta_reimpl"] = DeepDTAModel


_register_deep_models()


def build_model(name: str, **kwargs: Any) -> DTAModel:
    """Construct a registered model by name."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS))
        extra = (
            ""
            if "shift_robust" in MODELS
            else " (PyTorch is not installed, so the deep models are unavailable)"
        )
        raise KeyError(f"unknown model {name!r}; available: {available}{extra}")
    return MODELS[name](**kwargs)


def available_models() -> list[str]:
    return sorted(MODELS)
