"""BindingDB cross-source concordance analysis.

Compares ChEMBL 37 and BindingDB measurements for the same ligand, UniProt target,
and affinity endpoint. This is a database-concordance check, not an independent
experimental validation: BindingDB imports records from ChEMBL and the sources can
therefore contain the same publication or underlying assay.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.chem.standardize import standardize_smiles
from leakfree_dta.data import schema
from leakfree_dta.data.curate import TYPE_TO_P_TYPE
from leakfree_dta.data.sources import bindingdb
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.io import read_table, write_json, write_table, write_text

logger = get_logger(__name__)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cross_source_validation(
    bindingdb_path: Path,
    chembl_path: Path,
    output_dir: Path,
    max_bindingdb_rows: int | None = None,
    chembl_raw_path: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = paths.FIGURES / "chembl_3fam"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ChEMBL 3-family dataset from %s", chembl_path)
    chembl = read_table(chembl_path)
    logger.info("Loaded %d ChEMBL pairs", len(chembl))

    logger.info("Loading BindingDB extract from %s", bindingdb_path)
    # Load BindingDB chunk
    bdb_raw = bindingdb.load(bindingdb_path, chunksize=100_000)
    if max_bindingdb_rows and len(bdb_raw) > max_bindingdb_rows:
        bdb_raw = bdb_raw.iloc[:max_bindingdb_rows]

    logger.info("Checking InChIKeys in BindingDB...")
    if schema.LIGAND_KEY not in bdb_raw.columns or bdb_raw[schema.LIGAND_KEY].isna().all():
        logger.info("Standardizing BindingDB SMILES to canonical InChIKeys...")
        unique_smiles = bdb_raw[schema.SMILES].drop_duplicates()
        smi_to_key = {}
        for smi in unique_smiles:
            res = standardize_smiles(smi)
            if res.ok and res.inchikey:
                smi_to_key[smi] = res.inchikey
        bdb_raw[schema.LIGAND_KEY] = bdb_raw[schema.SMILES].map(smi_to_key)

    bdb = bdb_raw[bdb_raw[schema.LIGAND_KEY].notna() & (bdb_raw[schema.LIGAND_KEY] != "")].copy()
    bdb[schema.PAIR_KEY] = schema.make_pair_key(bdb[schema.LIGAND_KEY], bdb[schema.TARGET_KEY])

    # Filter uncensored
    bdb = bdb[bdb[schema.Y_RELATION] == "="].copy()
    logger.info("Usable BindingDB records with InChIKeys: %d", len(bdb))

    # Endpoint identity is part of the key: comparing a Ki with an IC50 would
    # manufacture disagreement unrelated to database curation.
    joined = bindingdb.cross_source_disagreement(chembl, bdb)
    logger.info("Shared pairs found: %d", len(joined))

    if len(joined) == 0:
        logger.warning("No shared pairs found between ChEMBL and BindingDB.")
        return {}

    # Merge family information from chembl
    chembl_meta = chembl[
        [schema.PAIR_KEY, schema.Y_TYPE, schema.TARGET_FAMILY, schema.TARGET_KEY]
    ].drop_duplicates(subset=[schema.PAIR_KEY, schema.Y_TYPE])
    joined = joined.merge(chembl_meta, on=[schema.PAIR_KEY, schema.Y_TYPE], how="left")

    shared_keys = joined[[schema.PAIR_KEY, schema.Y_TYPE]].drop_duplicates()
    contributing = bdb.merge(shared_keys, on=[schema.PAIR_KEY, schema.Y_TYPE], how="inner")
    if "curation_source" in contributing.columns:
        source_df = (
            contributing["curation_source"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .rename_axis("bindingdb_curation_source")
            .reset_index(name="n_records")
        )
        source_df["fraction"] = source_df["n_records"] / source_df["n_records"].sum()
    else:
        source_df = pd.DataFrame(
            [{"bindingdb_curation_source": "not available", "n_records": len(contributing), "fraction": 1.0}]
        )

    publication_stats: dict[str, float | int] = {}
    if chembl_raw_path is not None and chembl_raw_path.exists() and "pmid" in contributing:
        raw = read_table(chembl_raw_path)[
            ["canonical_smiles", "target_uniprot", "standard_type", "pubmed_id"]
        ].rename(
            columns={"canonical_smiles": schema.SMILES, "target_uniprot": schema.TARGET_KEY}
        )
        raw[schema.Y_TYPE] = raw["standard_type"].map(TYPE_TO_P_TYPE)
        raw["pubmed_id"] = pd.to_numeric(raw["pubmed_id"], errors="coerce").astype("Int64")

        # Recover publication identifiers for the current processed table through
        # exact canonical-SMILES matches. Coverage is reported because structure
        # standardisation can change the string without changing the molecule.
        processed_shared = chembl[
            [schema.PAIR_KEY, schema.SMILES, schema.TARGET_KEY, schema.Y_TYPE]
        ].merge(shared_keys, on=[schema.PAIR_KEY, schema.Y_TYPE], how="inner")
        chembl_pubs = raw.merge(
            processed_shared,
            on=[schema.SMILES, schema.TARGET_KEY, schema.Y_TYPE],
            how="inner",
        ).dropna(subset=["pubmed_id"])
        bdb_pubs = contributing.copy()
        bdb_pubs["pmid"] = pd.to_numeric(bdb_pubs["pmid"], errors="coerce").astype("Int64")
        bdb_pubs = bdb_pubs.dropna(subset=["pmid"])

        keys = [schema.PAIR_KEY, schema.Y_TYPE]
        left_pubs = chembl_pubs.groupby(keys, observed=True)["pubmed_id"].agg(
            lambda values: frozenset(map(int, values))
        )
        right_pubs = bdb_pubs.groupby(keys, observed=True)["pmid"].agg(
            lambda values: frozenset(map(int, values))
        )
        pub_compare = left_pubs.rename("chembl_pmids").to_frame().join(
            right_pubs.rename("bindingdb_pmids"), how="inner"
        )
        pub_compare["same_publication"] = [
            bool(a & b) for a, b in zip(pub_compare["chembl_pmids"], pub_compare["bindingdb_pmids"], strict=True)
        ]
        publication_stats = {
            "n_pairs_with_pmids_both_sources": len(pub_compare),
            "fraction_same_publication": float(pub_compare["same_publication"].mean()),
            "pmid_coverage_of_shared_pairs": float(len(pub_compare) / len(joined)),
        }

    # Metrics
    deltas = joined["abs_delta"]
    signed_delta = joined["chembl_y"] - joined["bindingdb_y"]
    rmse = np.sqrt(np.mean(signed_delta**2))
    mae = np.mean(deltas)
    median_delta = np.median(deltas)
    mean_offset = np.mean(signed_delta)
    pct_gt_05 = 100.0 * np.mean(deltas > 0.5)
    pct_gt_10 = 100.0 * np.mean(deltas > 1.0)
    pct_gt_20 = 100.0 * np.mean(deltas > 2.0)
    pearson = np.corrcoef(joined["chembl_y"], joined["bindingdb_y"])[0, 1]

    logger.info(
        "Cross-source consistency summary: N=%d, Pearson=%.3f, RMSE=%.3f, MAE=%.3f, Median |Δ|=%.3f, Offset=%.3f, >1.0 log: %.1f%%",
        len(joined), pearson, rmse, mae, median_delta, mean_offset, pct_gt_10
    )

    # Breakdown by family
    family_rows = []
    for fam, grp in joined.groupby(schema.TARGET_FAMILY):
        d = grp["abs_delta"]
        sd = grp["chembl_y"] - grp["bindingdb_y"]
        family_rows.append({
            "target_family": fam,
            "n_shared_pairs": len(grp),
            "pearson": float(np.corrcoef(grp["chembl_y"], grp["bindingdb_y"])[0, 1]) if len(grp) > 2 else float("nan"),
            "rmse": float(np.sqrt(np.mean(sd**2))),
            "mae": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "fraction_gt_1log": float(np.mean(d > 1.0)),
        })
    family_df = pd.DataFrame(family_rows)

    # Save summary tables
    summary_data = {
        "metric": [
            "n_shared_pairs",
            "pearson_r",
            "rmse_log_units",
            "mae_log_units",
            "median_abs_delta",
            "mean_signed_offset",
            "fraction_gt_0_5_log",
            "fraction_gt_1_0_log (10-fold)",
            "fraction_gt_2_0_log (100-fold)",
        ],
        "value": [
            len(joined),
            round(float(pearson), 4),
            round(float(rmse), 4),
            round(float(mae), 4),
            round(float(median_delta), 4),
            round(float(mean_offset), 4),
            round(float(pct_gt_05 / 100.0), 4),
            round(float(pct_gt_10 / 100.0), 4),
            round(float(pct_gt_20 / 100.0), 4),
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    if publication_stats:
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    {"metric": list(publication_stats), "value": list(publication_stats.values())}
                ),
            ],
            ignore_index=True,
        )
    
    write_table(summary_df, output_dir / "table_cross_source_bindingdb.csv")
    write_table(family_df, output_dir / "table_cross_source_by_family.csv")
    write_table(source_df, output_dir / "table_cross_source_provenance.csv")
    write_table(joined.head(1000), output_dir / "sample_cross_source_pairs.csv")
    write_json(
        {
            "analysis": "endpoint-matched database concordance",
            "bindingdb_file": bindingdb_path.name,
            "bindingdb_bytes": bindingdb_path.stat().st_size,
            "bindingdb_sha256": sha256_file(bindingdb_path),
            "chembl_file": chembl_path.name,
            "chembl_bytes": chembl_path.stat().st_size,
            "chembl_sha256": sha256_file(chembl_path),
            "chembl_raw_file": chembl_raw_path.name if chembl_raw_path else None,
            "match_key": [schema.LIGAND_KEY, schema.TARGET_KEY, schema.Y_TYPE],
            "single_chain_targets_only": True,
            "uncensored_only": True,
            "n_shared": len(joined),
            **publication_stats,
        },
        output_dir / "cross_source_provenance.json",
    )

    # Render Markdown table
    md_lines = [
        "## Cross-Source Consistency: ChEMBL 37 vs. BindingDB",
        "",
        "Endpoint-matched concordance for identical ligand-target pairs in ChEMBL 37 and BindingDB.",
        "This is not independent experimental validation: BindingDB imports records from ChEMBL,",
        "so this measures database-level concordance and provenance overlap, not a physical noise floor.",
        "",
        summary_df.to_markdown(index=False),
        "",
        "### Breakdown by Target Family",
        "",
        family_df.to_markdown(index=False),
        "",
        "### BindingDB provenance among contributing records",
        "",
        source_df.to_markdown(index=False),
        "",
    ]
    write_text("\n".join(md_lines), output_dir / "cross_source_summary.md")

    # Generate Figure
    _fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax1, ax2 = axes

    # Scatter
    ax1.scatter(joined["chembl_y"], joined["bindingdb_y"], alpha=0.35, edgecolors="none", s=18, c="#1f77b4")
    min_v = min(joined["chembl_y"].min(), joined["bindingdb_y"].min())
    max_v = max(joined["chembl_y"].max(), joined["bindingdb_y"].max())
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", alpha=0.8, label="Ideal agreement ($y=x$)")
    ax1.plot([min_v, max_v], [min_v + 1.0, max_v + 1.0], ":", color="gray", label="$\\pm 1$ log unit (10-fold)")
    ax1.plot([min_v, max_v], [min_v - 1.0, max_v - 1.0], ":", color="gray")
    ax1.set_xlabel("ChEMBL 37 pAffinity ($-\\log_{10}$ M)", fontsize=11)
    ax1.set_ylabel("BindingDB pAffinity ($-\\log_{10}$ M)", fontsize=11)
    ax1.set_title(f"Cross-Source Concordance ($N={len(joined):,}$, Pearson $r={pearson:.3f}$)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Histogram of residuals
    ax2.hist(signed_delta, bins=40, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    ax2.axvline(0, color="red", linestyle="--", label="Zero offset")
    ax2.axvline(mean_offset, color="blue", linestyle="-", label=f"Mean offset ({mean_offset:+.2f})")
    ax2.axvline(-1.0, color="gray", linestyle=":")
    ax2.axvline(1.0, color="gray", linestyle=":", label="$\\pm 1$ log bounds")
    ax2.set_xlabel("Residual: ChEMBL $-$ BindingDB (log units)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"Disagreement Distribution (RMSE = {rmse:.3f}, MAE = {mae:.3f})", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = figures_dir / "fig_cross_source_bindingdb.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info("Saved figure to %s", fig_path)

    return {
        "n_shared": len(joined),
        "pearson": float(pearson),
        "rmse": float(rmse),
        "mae": float(mae),
        "pct_gt_10": float(pct_gt_10),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb", required=True, help="Path to a BindingDB TSV release")
    parser.add_argument(
        "--chembl",
        default=str(paths.PROCESSED / "chembl_3fam.aggregated.parquet"),
    )
    parser.add_argument(
        "--chembl-raw",
        default=str(paths.RAW / "chembl_3fam.parquet"),
        help="Raw ChEMBL extract carrying PMID provenance",
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.TABLES / "chembl_3fam"),
    )
    args = parser.parse_args()

    setup_logging("INFO")
    run_cross_source_validation(
        Path(args.bindingdb),
        Path(args.chembl),
        Path(args.output_dir),
        chembl_raw_path=Path(args.chembl_raw),
    )


if __name__ == "__main__":
    main()
