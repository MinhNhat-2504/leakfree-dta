"""Assembling the paper's tables from the artifacts on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from leakfree_dta import paths
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import read_json

logger = get_logger(__name__)


def leakage_ladder(dataset: str) -> pd.DataFrame:
    """Table 1: leakage by channel, one row per split.

    The paper's first table and its central evidence. Each row is a rung of the
    split ladder; each column is a channel. Reading down a column shows which
    constraint closes which channel, and the last column shows what the shortcut
    probe can still reach once they are all closed.
    """
    rows: list[dict[str, Any]] = []
    audit_root = paths.AUDIT / dataset
    if not audit_root.exists():
        logger.warning("no audit results under %s", audit_root)
        return pd.DataFrame()

    for split_dir in sorted(audit_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "dataset":
            continue
        report_file = split_dir / "report.json" if (split_dir / "report.json").exists() else split_dir / "overall" / "report.json"
        if not report_file.exists():
            continue
        payload = read_json(report_file)
        row: dict[str, Any] = {
            "split": payload.get("split", split_dir.name),
            "n_train": payload.get("metadata", {}).get("n_train"),
            "n_test": payload.get("metadata", {}).get("n_test"),
            "recoverable_fraction": payload.get("recoverable_fraction"),
            "worst_severity": payload.get("worst_severity"),
        }
        for result in payload.get("results", []):
            row[result["name"]] = result["fraction"]
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Order the splits as the ladder does, permissive first, so the table reads as
    # a progression rather than alphabetically.
    from leakfree_dta.splits.three_axis import SPLIT_LADDER

    order = {spec.name: index for index, spec in enumerate(SPLIT_LADDER)}
    table["_order"] = table["split"].map(order).fillna(len(order))
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def model_comparison(dataset: str, split: str) -> pd.DataFrame:
    """Table 2: every model's metrics with cluster-bootstrap intervals."""
    path = paths.TABLES / f"{dataset}.{split}.metrics.csv"
    if not path.exists():
        logger.warning("no metrics table at %s; run `leakfree evaluate`", path)
        return pd.DataFrame()

    table = pd.read_csv(path)
    table["formatted"] = table.apply(
        lambda row: f"{row['estimate']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1
    )
    return table.pivot(index="model", columns="metric", values="formatted").reset_index()


def inflation_table(dataset: str, *, metric: str = "pearson") -> pd.DataFrame:
    """Table 3: how much each model's score falls as the split tightens.

    The quantitative answer to the title. A model whose score collapses from the
    random split to the leakage-free one was being scored on leakage; one whose
    score holds was doing something real. Both outcomes are findings.
    """
    rows: list[dict[str, Any]] = []
    runs_root = paths.RUNS / dataset
    if not runs_root.exists():
        return pd.DataFrame()

    for split_dir in sorted(runs_root.iterdir()):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            metrics_file = model_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            metrics = read_json(metrics_file)
            rows.append(
                {
                    "model": model_dir.name,
                    "split": split_dir.name,
                    metric: metrics.get(metric),
                    "rmse": metrics.get("rmse"),
                    "n": metrics.get("n"),
                }
            )

    long = pd.DataFrame(rows)
    if long.empty:
        return long

    wide = long.pivot(index="model", columns="split", values=metric)
    if "random" in wide.columns:
        for column in wide.columns:
            if column == "random":
                continue
            wide[f"drop_vs_random_{column}"] = wide["random"] - wide[column]
    return wide.reset_index()


def curation_summary(dataset: str) -> pd.DataFrame:
    """Supplementary table: the curation log, as published."""
    path = paths.TABLES / f"{dataset}.curation_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def noise_floor(dataset: str) -> dict[str, Any]:
    """The label noise floor, extracted from the dataset audit.

    Quoted in the results section next to every RMSE, because an RMSE below the
    noise floor is not an achievement -- it is a sign the model is fitting assay
    artifacts shared between the training and test labels.
    """
    report_file = paths.AUDIT / dataset / "dataset" / "report.json"
    if not report_file.exists():
        return {}
    payload = read_json(report_file)
    for result in payload.get("results", []):
        if result["name"] == "conflicting_labels":
            statistics = result.get("statistics", {})
            return {
                "noise_floor_rmse": statistics.get("noise_floor_rmse"),
                "n_replicated_pairs": statistics.get("n_replicated_pairs"),
                "fraction_conflicting": statistics.get("fraction_of_replicated_pairs"),
            }
    return {}


def write_markdown(table: pd.DataFrame, path: str | Path, *, title: str, note: str = "") -> Path:
    """Write a table as Markdown with its caption, ready to paste into the paper."""
    from leakfree_dta.utils.io import write_text

    if table.empty:
        return write_text(f"## {title}\n\n_No data available._\n", path)

    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    lines.append(table.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return write_text("\n".join(lines), path)
