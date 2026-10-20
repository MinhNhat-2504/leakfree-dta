"""The progress curve: the paper's headline figure.

Two lines on one set of axes, with publication year on the x-axis:

1. **Reported performance.** What each model's own paper claimed, taken from the
   literature.
2. **Re-measured performance.** The same models, re-run by us on a split that
   closes every leakage channel.

If the field has made real progress, the two lines rise together. If much of the
reported gain came from increasingly transductive evaluation on increasingly
leaky benchmarks, the second line is flat while the first climbs. Whichever it is,
the figure is the answer to the title's question, and it is a measurement rather
than an argument.

The honest version of this figure needs three things the code enforces:

* Every re-measured point comes from the *same* split, or the comparison is
  meaningless. :func:`validate_rerun_table` checks that.
* The reported points must be quoted with their original metric and benchmark,
  because a Davis CI and a ChEMBL Pearson are not comparable numbers.
* Models that could not be re-run are shown as gaps, not dropped. Silently
  omitting the ones that failed to reproduce would bias the second line upward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReportedResult:
    """A number quoted from a published paper, with its provenance."""

    model: str
    year: int
    benchmark: str
    metric: str
    value: float
    split: str
    citation: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "year": self.year,
            "benchmark": self.benchmark,
            "metric": self.metric,
            "reported_value": self.value,
            "reported_split": self.split,
            "citation": self.citation,
            "notes": self.notes,
        }


@dataclass
class RerunResult:
    """A number we measured ourselves, with the reason if we could not."""

    model: str
    year: int
    split: str
    metric: str
    value: float | None
    ci_lower: float | None = None
    ci_upper: float | None = None
    status: str = "ok"  # "ok" | "failed" | "not_attempted"
    reason: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "year": self.year,
            "split": self.split,
            "metric": self.metric,
            "rerun_value": self.value,
            "rerun_ci_lower": self.ci_lower,
            "rerun_ci_upper": self.ci_upper,
            "status": self.status,
            "reason": self.reason,
            **{f"provenance_{key}": value for key, value in self.provenance.items()},
        }


def build_progress_table(
    reported: list[ReportedResult],
    rerun: list[RerunResult],
    *,
    metric: str = "pearson",
) -> pd.DataFrame:
    """Join reported and re-measured results into the figure's data table."""
    reported_frame = pd.DataFrame([result.to_dict() for result in reported])
    rerun_frame = pd.DataFrame([result.to_dict() for result in rerun])

    if reported_frame.empty or rerun_frame.empty:
        logger.warning("progress table needs both reported and re-run results")
        return pd.DataFrame()

    reported_frame = reported_frame[reported_frame["metric"] == metric]
    rerun_frame = rerun_frame[rerun_frame["metric"] == metric]

    merged = reported_frame.merge(
        rerun_frame.drop(columns=["year"]), on="model", how="outer", suffixes=("", "_rerun")
    )
    merged["year"] = merged["year"].fillna(
        merged["model"].map(dict(zip(rerun_frame["model"], rerun_frame["year"], strict=True)))
    )
    merged["delta"] = merged["rerun_value"] - merged["reported_value"]
    return merged.sort_values("year").reset_index(drop=True)


def validate_rerun_table(rerun: list[RerunResult]) -> list[str]:
    """Check that the re-run results are actually comparable to each other.

    Returns the problems found. An empty list is a precondition for plotting the
    figure, because a curve whose points come from different splits measures
    nothing.
    """
    problems: list[str] = []
    attempted = [result for result in rerun if result.status == "ok"]
    if not attempted:
        return ["no model was successfully re-run"]

    splits = {result.split for result in attempted}
    if len(splits) > 1:
        problems.append(f"re-run results span several splits {sorted(splits)}; they are not comparable")

    metrics = {result.metric for result in attempted}
    if len(metrics) > 1:
        problems.append(f"re-run results mix metrics {sorted(metrics)}")

    missing_ci = [result.model for result in attempted if result.ci_lower is None]
    if missing_ci:
        problems.append(f"no confidence interval for {missing_ci}")

    failed = [result for result in rerun if result.status != "ok"]
    unexplained = [result.model for result in failed if not result.reason]
    if unexplained:
        problems.append(f"models excluded without a stated reason: {unexplained}")

    return problems


def trend_slope(table: pd.DataFrame, *, value_column: str, year_column: str = "year") -> dict[str, float]:
    """Least-squares slope of performance against year, with its interval.

    The number the paper's abstract quotes: the per-year improvement in reported
    performance, beside the per-year improvement that survives re-measurement.
    """
    frame = table[[year_column, value_column]].dropna()
    if len(frame) < 3:
        return {"slope": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": len(frame)}

    from scipy import stats

    years = frame[year_column].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    fit = stats.linregress(years, values)
    # 95% interval on the slope from its standard error, t-distributed.
    critical = stats.t.ppf(0.975, df=len(frame) - 2)
    return {
        "slope": round(float(fit.slope), 5),
        "ci_lower": round(float(fit.slope - critical * fit.stderr), 5),
        "ci_upper": round(float(fit.slope + critical * fit.stderr), 5),
        "r_squared": round(float(fit.rvalue**2), 4),
        "p_value": float(fit.pvalue),
        "n": len(frame),
    }


def progress_summary(table: pd.DataFrame) -> dict[str, Any]:
    """The figure reduced to the sentences the abstract needs."""
    reported_trend = trend_slope(table, value_column="reported_value")
    rerun_trend = trend_slope(table, value_column="rerun_value")

    deltas = table["delta"].dropna().to_numpy(dtype=float)
    retained = (
        float(rerun_trend["slope"] / reported_trend["slope"])
        if np.isfinite(reported_trend["slope"]) and abs(reported_trend["slope"]) > 1e-9
        else float("nan")
    )

    return {
        "reported_trend_per_year": reported_trend,
        "rerun_trend_per_year": rerun_trend,
        "fraction_of_trend_retained": None if not np.isfinite(retained) else round(retained, 4),
        "mean_delta": round(float(deltas.mean()), 5) if deltas.size else None,
        "median_delta": round(float(np.median(deltas)), 5) if deltas.size else None,
        "n_models_rerun": int(table["rerun_value"].notna().sum()),
        "n_models_failed": int((table.get("status", pd.Series(dtype=object)) == "failed").sum()),
    }
