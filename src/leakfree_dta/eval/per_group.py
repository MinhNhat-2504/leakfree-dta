"""Per-group metrics and worst-group reporting.

An overall metric on an imbalanced test set is a weighted average dominated by a
few heavily measured targets. Two models with the same overall RMSE can differ
completely in whether they work on the targets a screening campaign actually
cares about, which are usually the ones with little data.

So every result in the paper is reported three ways: overall, macro-averaged over
groups, and worst group. The third is the one that matters for the robustness
claim, and it is the objective the group-DRO baseline optimises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from leakfree_dta.eval.metrics import METRICS, evaluate
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)


def per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence,
    *,
    min_group_size: int = 10,
    active_threshold: float = 7.0,
) -> pd.DataFrame:
    """One row of metrics per group.

    Groups smaller than ``min_group_size`` are reported but flagged, because a
    Pearson correlation over four points is not an estimate of anything.
    """
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true, float), "y_pred": np.asarray(y_pred, float), "group": np.asarray(groups)}
    )
    rows: list[dict[str, object]] = []
    for group, chunk in frame.groupby("group", observed=True):
        metrics = evaluate(
            chunk["y_true"].to_numpy(), chunk["y_pred"].to_numpy(), active_threshold=active_threshold
        )
        rows.append(
            {
                "group": group,
                "n": metrics.n,
                "reliable": metrics.n >= min_group_size,
                **{key: round(value, 5) for key, value in metrics.values.items()},
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True) if len(out) else out


def aggregate_group_metrics(
    per_group: pd.DataFrame,
    *,
    metric: str = "rmse",
    higher_is_better: bool = False,
    reliable_only: bool = True,
) -> dict[str, float]:
    """Macro average, worst group, and spread across groups."""
    frame = per_group[per_group["reliable"]] if reliable_only and "reliable" in per_group else per_group
    if frame.empty or metric not in frame.columns:
        return {"macro": float("nan"), "worst": float("nan"), "spread": float("nan"), "n_groups": 0}

    values = frame[metric].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"macro": float("nan"), "worst": float("nan"), "spread": float("nan"), "n_groups": 0}

    worst = float(values.min() if higher_is_better else values.max())
    return {
        "macro": round(float(values.mean()), 5),
        "worst": round(worst, 5),
        "p10": round(float(np.percentile(values, 10 if higher_is_better else 90)), 5),
        "spread": round(float(values.max() - values.min()), 5),
        "n_groups": int(values.size),
    }


def group_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_sets: dict[str, Sequence],
    *,
    metrics: Sequence[str] = ("rmse", "pearson", "spearman"),
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Aggregate group statistics for several groupings at once.

    The paper groups by target family, by protein cluster and by ligand cluster,
    because a model can be uniform across families and wildly uneven within one.
    """
    rows: list[dict[str, object]] = []
    for grouping_name, groups in group_sets.items():
        per_group = per_group_metrics(y_true, y_pred, groups, min_group_size=min_group_size)
        for metric in metrics:
            higher_is_better = metric not in {"rmse", "mae", "ece"}
            summary = aggregate_group_metrics(
                per_group, metric=metric, higher_is_better=higher_is_better
            )
            rows.append({"grouping": grouping_name, "metric": metric, **summary})
    return pd.DataFrame(rows)


def worst_group_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    min_group_size: int = 10,
) -> float:
    """The worst group's loss -- the quantity group DRO minimises.

    Exposed as a function so that the training objective and the reported metric
    are computed by the same code. Where they diverge, a paper reports an
    improvement on a quantity it did not optimise, and the difference is usually
    where the improvement came from.
    """
    metric = metric or METRICS["rmse"]
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true, float), "y_pred": np.asarray(y_pred, float), "group": np.asarray(groups)}
    )
    losses: list[float] = []
    for _, chunk in frame.groupby("group", observed=True):
        if len(chunk) < min_group_size:
            continue
        losses.append(metric(chunk["y_true"].to_numpy(), chunk["y_pred"].to_numpy()))
    return float(max(losses)) if losses else float("nan")
