"""Regenerating every paper artifact in one call.

``leakfree report --dataset X`` runs this. It reads only what is already on disk,
so it is fast, repeatable, and cannot quietly change a number between drafts.
Anything missing becomes a note in the output rather than an error, because the
report is assembled many times while the slow stages are still running.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.eval.risk_coverage import risk_coverage_curve
from leakfree_dta.logging import get_logger
from leakfree_dta.report import figures, tables
from leakfree_dta.utils.io import read_table, write_table, write_text

logger = get_logger(__name__)


def build_report(dataset: str, *, splits: list[str] | None = None) -> list[Path]:
    """Write every table and figure the paper needs, and an index of them."""
    written: list[Path] = []
    out_tables = paths.TABLES / dataset
    out_figures = paths.FIGURES / dataset
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []

    # Table 1 / Figure 1 -- the leakage ladder.
    ladder = tables.leakage_ladder(dataset)
    if ladder.empty:
        notes.append("No audit results found. Run `leakfree audit` for each split.")
    else:
        written.append(write_table(ladder, out_tables / "table1_leakage_ladder.csv"))
        written.append(
            tables.write_markdown(
                ladder,
                out_tables / "table1_leakage_ladder.md",
                title="Table 1. Leakage by channel across the split ladder",
                note=(
                    "Each value is the fraction of test rows affected by that channel. "
                    "Rows are ordered from the field's usual practice (`random`) to the "
                    "setting that closes every channel (`leakfree`)."
                ),
            )
        )
        figure = figures.plot_leakage_ladder(ladder, out_figures / "fig1_leakage_ladder.png")
        if figure:
            written.append(figure)

    # Table 3 / Figure 2 -- inflation across the ladder.
    inflation = tables.inflation_table(dataset)
    if inflation.empty:
        notes.append("No model runs found. Run `leakfree train` for each model and split.")
    else:
        written.append(write_table(inflation, out_tables / "table3_inflation.csv"))
        written.append(
            tables.write_markdown(
                inflation,
                out_tables / "table3_inflation.md",
                title="Table 3. Pearson correlation by model and split",
                note=(
                    "The `drop_vs_random_*` columns are the inflation attributable to "
                    "whichever channels that split closes."
                ),
            )
        )
        figure = figures.plot_inflation(inflation, out_figures / "fig2_inflation.png")
        if figure:
            written.append(figure)

    # Table 2 -- per-split model comparison with intervals.
    discovered = splits or _discover_splits(dataset)
    for split in discovered:
        comparison = tables.model_comparison(dataset, split)
        if comparison.empty:
            continue
        written.append(write_table(comparison, out_tables / f"table2_models_{split}.csv"))
        written.append(
            tables.write_markdown(
                comparison,
                out_tables / f"table2_models_{split}.md",
                title=f"Table 2. Model comparison on the `{split}` split",
                note=(
                    "Point estimate with a 95% cluster-bootstrap interval, resampling "
                    "protein groups rather than rows. Degenerate predictors "
                    "(`global_mean`, `target_mean`, `ligand_mean`, `additive_bias`) are "
                    "included on purpose: they bound what the benchmark rewards without "
                    "interaction modelling."
                ),
            )
        )

    robustness = tables.robustness_summary(dataset)
    if not robustness.empty:
        written.append(write_table(robustness, out_tables / "table4_robustness.csv"))
        written.append(
            tables.write_markdown(
                robustness,
                out_tables / "table4_robustness.md",
                title="Table 4. Robustness and uncertainty summary on the leakage-free split",
                note=(
                    "Worst-family RMSE is the maximum family-level RMSE. Coverage and "
                    "interval width are reported only for models that emit prediction "
                    "intervals; AURC is the area under the selective risk-coverage curve."
                ),
            )
        )
        coverage_figure = _coverage_figure(dataset, out_figures)
        if coverage_figure:
            written.append(coverage_figure)
        risk_figure = _risk_coverage_figure(dataset, out_figures)
        if risk_figure:
            written.append(risk_figure)

    # The noise floor, quoted beside every RMSE in the text.
    floor = tables.noise_floor(dataset)
    if floor:
        written.append(
            write_text(
                "## Label noise floor\n\n"
                + "\n".join(f"- `{key}`: {value}" for key, value in floor.items())
                + "\n\nAn RMSE below the noise floor indicates the model is fitting assay "
                "artifacts shared between the training and test labels, not affinity.\n",
                out_tables / "noise_floor.md",
            )
        )
    else:
        notes.append("No noise floor available. Run `leakfree audit` without `--split`.")

    # The curation log, published as the supplement.
    curation = tables.curation_summary(dataset)
    if not curation.empty:
        written.append(
            tables.write_markdown(
                curation,
                out_tables / "supp_curation_log.md",
                title="Supplementary Table. Curation log",
                note="Rows removed at each curation step, with the reason.",
            )
        )

    # The re-run attempt log and the pre-registered inclusion criteria.
    from leakfree_dta.models.external import criteria_markdown, registry_table

    written.append(write_table(registry_table(), out_tables / "supp_external_models.csv"))
    written.append(write_text(criteria_markdown(), out_tables / "supp_inclusion_criteria.md"))

    written.append(_write_index(dataset, written, notes, out_tables))
    return written


def _coverage_figure(dataset: str, out_figures: Path) -> Path | None:
    """Generate family-level empirical coverage for the interval model."""
    path = paths.RUNS / dataset / "leakfree" / "shift_robust" / "predictions.parquet"
    if not path.exists():
        return None
    frame = read_table(path)
    required = {"y_true", "y_lower", "y_upper", schema.TARGET_FAMILY}
    if not required.issubset(frame.columns):
        return None
    frame["covered"] = (frame["y_true"] >= frame["y_lower"]) & (
        frame["y_true"] <= frame["y_upper"]
    )
    rows = [
        {
            "group": group,
            "coverage": float(chunk["covered"].mean()),
            "n": len(chunk),
        }
        for group, chunk in frame.groupby(schema.TARGET_FAMILY, observed=True)
    ]
    rows.append({"group": "__overall__", "coverage": float(frame["covered"].mean()), "n": len(frame)})
    coverage = pd.DataFrame(rows)
    table_path = paths.TABLES / dataset / "family_coverage.csv"
    write_table(coverage, table_path)
    return figures.plot_group_coverage(coverage, out_figures / "fig3_coverage_by_family.png")


def _risk_coverage_figure(dataset: str, out_figures: Path) -> Path | None:
    """Generate risk-coverage curves for models that report uncertainty."""
    root = paths.RUNS / dataset / "leakfree"
    curves: dict[str, pd.DataFrame] = {}
    for model in ("erm_mlp", "group_dro_mlp", "shift_robust"):
        path = root / model / "predictions.parquet"
        if not path.exists():
            continue
        frame = read_table(path)
        if not {"y_true", "y_pred", "y_std"}.issubset(frame.columns):
            continue
        curves[model] = risk_coverage_curve(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            -frame["y_std"].to_numpy(),
        )
    if not curves:
        return None
    return figures.plot_risk_coverage(curves, out_figures / "fig4_risk_coverage.png")


def _discover_splits(dataset: str) -> list[str]:
    runs = paths.RUNS / dataset
    if not runs.exists():
        return []
    return sorted(path.name for path in runs.iterdir() if path.is_dir())


def _write_index(dataset: str, written: list[Path], notes: list[str], out_dir: Path) -> Path:
    lines = [
        f"# Report artifacts for `{dataset}`",
        "",
        "Generated by `leakfree report`. Every number here was computed by an earlier "
        "pipeline stage and read from disk; regenerating this report does not recompute "
        "anything.",
        "",
        "## Files",
        "",
    ]
    for path in sorted(written):
        try:
            relative = path.relative_to(paths.ROOT)
        except ValueError:
            relative = path
        lines.append(f"- `{relative.as_posix()}`")

    if notes:
        lines += ["", "## Missing pieces", ""]
        lines += [f"- {note}" for note in notes]

    lines += [
        "",
        "## Suggested reading order for a reviewer",
        "",
        "1. `table1_leakage_ladder` -- what leaks, and which constraint closes it.",
        "2. `noise_floor` -- the error below which an RMSE is not meaningful.",
        "3. `table2_models_leakfree` -- what the models achieve once nothing leaks, "
        "beside the predictors that model no interaction at all.",
        "4. `table3_inflation` / `fig2_inflation` -- the difference between those two "
        "worlds, which is the paper's answer to its title.",
        "",
    ]
    return write_text("\n".join(lines), out_dir / "INDEX.md")
