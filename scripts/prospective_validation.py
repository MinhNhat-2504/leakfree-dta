"""Retrospective temporal-holdout evaluation and prospective-study protocol.

Evaluates models trained through 2018 on records published in 2021--2025. Because
the analysis was designed after those measurements existed, it is a retrospective
temporal simulation, not prospective or experimental validation. The script also
writes a protocol that can be frozen before genuinely future measurements arrive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.eval.metrics import evaluate
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.splits.manifest import apply_manifest, load_manifest
from leakfree_dta.utils.io import read_table, write_table, write_text

logger = get_logger(__name__)


def run_prospective_analysis(dataset: str = "chembl_3fam") -> dict:
    tables_dir = paths.TABLES / dataset
    figures_dir = paths.FIGURES / dataset
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading aggregated dataset for metadata...")
    agg_path = paths.PROCESSED / f"{dataset}.aggregated.parquet"
    full_df = read_table(agg_path)
    manifest_path = paths.SPLITS / dataset / "temporal.json"
    assignment = apply_manifest(full_df, load_manifest(manifest_path), allow_partial=True)
    train_year_max = int(pd.to_numeric(assignment.train(full_df)[schema.DOC_YEAR]).max())
    valid_years_observed = sorted(
        pd.to_numeric(assignment.valid(full_df)[schema.DOC_YEAR]).dropna().astype(int).unique()
    )
    test_year_min = int(pd.to_numeric(assignment.test(full_df)[schema.DOC_YEAR]).min())
    if (train_year_max, valid_years_observed, test_year_min) != (2018, [2019, 2020], 2021):
        raise RuntimeError(
            "temporal manifest does not implement train<=2018, valid=2019--2020, test>=2021: "
            f"got train max {train_year_max}, valid {valid_years_observed}, test min {test_year_min}"
        )
    meta = full_df[[schema.ROW_KEY, schema.DOC_YEAR, schema.TARGET_FAMILY, schema.SMILES, schema.TARGET_KEY, schema.LIGAND_KEY]].set_index(schema.ROW_KEY)

    runs_root = paths.RUNS / dataset / "temporal"
    if not runs_root.exists():
        raise SystemExit(f"No temporal runs found at {runs_root}")

    models = sorted([p.name for p in runs_root.iterdir() if p.is_dir() and (p / "predictions.parquet").exists()])
    logger.info("Found %d models evaluated on temporal split: %s", len(models), models)

    model_preds = {}
    for model in models:
        preds = read_table(runs_root / model / "predictions.parquet")
        model_preds[model] = preds.set_index(schema.ROW_KEY)

    ref_model = "xgboost_ecfp" if "xgboost_ecfp" in model_preds else models[0]
    ref_df = model_preds[ref_model].copy()
    ref_df["doc_year"] = meta.loc[ref_df.index, schema.DOC_YEAR]
    ref_df["target_family"] = meta.loc[ref_df.index, schema.TARGET_FAMILY]
    ref_df["smiles"] = meta.loc[ref_df.index, schema.SMILES]
    ref_df["target_key"] = meta.loc[ref_df.index, schema.TARGET_KEY]

    # Post-cutoff years. They are future relative to training but historical
    # relative to this analysis, hence "temporal holdout", not "prospective".
    years = sorted([int(y) for y in ref_df["doc_year"].dropna().unique() if y >= 2021])
    logger.info("Temporal-holdout years: %s (total rows: %d)", years, len(ref_df))

    # 1. Performance over post-cutoff publication years
    year_rows = []
    for year in years:
        mask = ref_df["doc_year"] == year
        n_pairs = int(mask.sum())
        for model in models:
            sub = model_preds[model].loc[mask]
            y_t = sub["y_true"].to_numpy()
            y_p = sub["y_pred"].to_numpy()
            m = evaluate(y_t, y_p)
            year_rows.append({
                "year": year,
                "model": model,
                "n_pairs": n_pairs,
                "rmse": float(m.get("rmse")),
                "pearson": float(m.get("pearson")),
                "spearman": float(m.get("spearman")),
                "mae": float(m.get("mae")),
            })
    year_df = pd.DataFrame(year_rows)
    write_table(year_df, tables_dir / "table_prospective_by_year.csv")

    # 2. Performance by target family across all held-out years (2021--2025)
    fam_rows = []
    for fam in ("kinase", "gpcr", "protease"):
        mask = ref_df["target_family"] == fam
        n_pairs = int(mask.sum())
        for model in models:
            sub = model_preds[model].loc[mask]
            y_t = sub["y_true"].to_numpy()
            y_p = sub["y_pred"].to_numpy()
            m = evaluate(y_t, y_p)
            fam_rows.append({
                "target_family": fam,
                "model": model,
                "n_pairs": n_pairs,
                "rmse": float(m.get("rmse")),
                "pearson": float(m.get("pearson")),
                "spearman": float(m.get("spearman")),
            })
    fam_df = pd.DataFrame(fam_rows)
    write_table(fam_df, tables_dir / "table_prospective_by_family.csv")

    # 3. Retrospective reference examples. These labels were already known when
    # this script was written and must never be described as assay validation.
    case_study_targets = {
        "P00533": "EGFR (Kinase, Non-small cell lung cancer)",
        "P15056": "BRAF (Kinase, Melanoma)",
        "P14416": "DRD2 (GPCR, Neuropsychiatric disorders)",
        "P28223": "HTR2A (GPCR, CNS/Serotonergic target)",
        "P0C6U8": "SARS-CoV-2 3CLpro / Mpro (Protease, COVID-19 antiviral)",
        "P56817": "BACE1 (Protease, Alzheimer's disease)",
    }

    case_rows = []
    top_candidates = []
    for target_id, target_name in case_study_targets.items():
        t_mask = ref_df["target_key"] == target_id
        if not t_mask.any():
            continue
        sub = ref_df[t_mask].copy()
        for model in ("xgboost_ecfp", "deepdta_reimpl", "erm_mlp", "additive_bias"):
            if model not in model_preds:
                continue
            m_sub = model_preds[model].loc[sub.index]
            y_t = m_sub["y_true"].to_numpy()
            y_p = m_sub["y_pred"].to_numpy()
            if len(y_t) > 3:
                m = evaluate(y_t, y_p)
                case_rows.append({
                    "uniprot": target_id,
                    "target_description": target_name,
                    "model": model,
                    "n_compounds": len(y_t),
                    "pearson": float(m.get("pearson")),
                    "rmse": float(m.get("rmse")),
                })

        # Identify high-ranked held-out examples for error inspection.
        xgb_preds = model_preds["xgboost_ecfp"].loc[sub.index]
        sub["y_pred_xgb"] = xgb_preds["y_pred"]
        sub["y_true"] = xgb_preds["y_true"]
        sub["error"] = np.abs(sub["y_pred_xgb"] - sub["y_true"])
        # Top 2 potent compounds
        top = sub.sort_values("y_pred_xgb", ascending=False).head(2)
        for _, r in top.iterrows():
            top_candidates.append({
                "target": target_name,
                "uniprot": target_id,
                "doc_year": int(r["doc_year"]),
                "smiles": r["smiles"],
                "measured_pY": float(r["y_true"]),
                "predicted_pY": float(r["y_pred_xgb"]),
                "abs_error": float(r["error"]),
                "potency_nM": float(10 ** (9 - r["y_true"])),
            })

    case_df = pd.DataFrame(case_rows)
    cand_df = pd.DataFrame(top_candidates)
    write_table(case_df, tables_dir / "table_prospective_case_targets.csv")
    write_table(cand_df, tables_dir / "table_prospective_experimental_candidates.csv")

    # 4. Generate Figures
    # Fig 1: Temporal Decay of Pearson correlation over 2021-2025
    plt.figure(figsize=(8, 5))
    palette = {
        "xgboost_ecfp": "#1f77b4",
        "erm_mlp": "#ff7f0e",
        "deepdta_reimpl": "#2ca02c",
        "additive_bias": "#d62728",
        "shift_robust": "#9467bd",
        "ridge_ecfp": "#8c564b",
    }
    for model in ("xgboost_ecfp", "erm_mlp", "deepdta_reimpl", "shift_robust", "additive_bias"):
        m_data = year_df[year_df["model"] == model].sort_values("year")
        if not m_data.empty:
            plt.plot(m_data["year"], m_data["pearson"], marker="o", linewidth=2.2, label=model, color=palette.get(model))

    plt.xlabel("Held-out publication year", fontsize=11)
    plt.ylabel("Temporal-holdout Pearson $r$", fontsize=11)
    plt.title("Retrospective temporal holdout: train <=2018, evaluate 2021-2025", fontsize=12)
    plt.xticks(years)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig1_path = figures_dir / "fig_prospective_temporal_drift.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig1_path)

    # Fig 2: temporal-holdout scatter with selected reference examples
    _fig, ax = plt.subplots(figsize=(7, 5))
    xgb_full = model_preds["xgboost_ecfp"]
    sample_sub = xgb_full.sample(n=min(3000, len(xgb_full)), random_state=42)
    ax.scatter(sample_sub["y_true"], sample_sub["y_pred"], alpha=0.2, c="#7f7f7f", s=12, label="Temporal-holdout pairs (2021-2025 sample)")
    if not cand_df.empty:
        ax.scatter(cand_df["measured_pY"], cand_df["predicted_pY"], color="crimson", s=70, marker="*", edgecolors="black", zorder=5, label="Selected retrospective examples")
    min_val = min(sample_sub["y_true"].min(), sample_sub["y_pred"].min())
    max_val = max(sample_sub["y_true"].max(), sample_sub["y_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="Ideal ($y=x$)")
    ax.set_xlabel("Measured held-out pAffinity ($-\\log_{10}$ M)", fontsize=11)
    ax.set_ylabel("Predicted pAffinity (XGBoost)", fontsize=11)
    ax.set_title("Retrospective validation on post-cutoff publications", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig2_path = figures_dir / "fig_prospective_assay_validation.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig2_path)

    # Generate Markdown Summary
    pivot_year = year_df.pivot(index="model", columns="year", values="pearson").reset_index()
    md_lines = [
        "## Retrospective Temporal-Holdout Evaluation",
        "",
        "**Status:** this is not prospective or experimental validation. The 2021--2025 labels",
        "already existed when the analysis was designed. Models trained on ChEMBL records through",
        "2018 are evaluated on later publications as a temporal-holdout simulation.",
        "",
        "### 1. Temporal generalization across held-out years (2021--2025)",
        "",
        pivot_year.to_markdown(index=False),
        "",
        "### 2. Temporal-holdout performance by target family (Pearson $r$)",
        "",
        fam_df.pivot(index="model", columns="target_family", values="pearson").reset_index().to_markdown(index=False),
        "",
        "### 3. Selected retrospective examples",
        "These compounds were selected after their measurements were available and are shown only for error inspection:",
        "",
        cand_df[["target", "doc_year", "potency_nM", "measured_pY", "predicted_pY", "abs_error"]].to_markdown(index=False),
        "",
    ]
    summary_path = tables_dir / "prospective_validation_summary.md"
    write_text("\n".join(md_lines), summary_path)
    logger.info("Saved temporal-holdout summary to %s", summary_path)

    protocol_lines = [
        "# Prospective validation protocol (not yet executed)",
        "",
        "This protocol becomes prospective only after the candidate list, model weights,",
        "software environment, endpoints, and analysis plan are timestamped before labels are obtained.",
        "",
        "1. Refit the selected model once on records published through the declared lock date.",
        "2. Save model weights and a SHA-256 digest; the current training pipeline saves predictions only,",
        "   so weight serialization must be added before enrollment.",
        "3. Register targets, compounds, primary affinity endpoint, replicate count, exclusion rules,",
        "   and primary metrics before any new labels are inspected.",
        "4. Obtain future database records or commission assays from an independent laboratory.",
        "5. Keep analysts blinded to labels until structures pass QC and predictions are frozen.",
        "6. Report every enrolled pair, failures included, with confidence intervals and deviations.",
        "",
        "A physical assay cannot be completed by this repository alone; it requires compounds,",
        "a laboratory, assay conditions, controls, and author approval.",
    ]
    protocol_path = tables_dir / "prospective_validation_protocol.md"
    write_text("\n".join(protocol_lines), protocol_path)

    return {
        "years": years,
        "n_models": len(models),
        "summary_path": str(summary_path),
        "protocol_path": str(protocol_path),
    }


def main():
    setup_logging("INFO")
    run_prospective_analysis()


if __name__ == "__main__":
    main()
