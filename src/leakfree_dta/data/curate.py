"""Curation: raw source tables into the canonical schema.

Every decision made here changes what the audit measures, so each one is a named,
configurable step rather than an inline transformation, and each is counted in the
curation log. The log is published alongside the benchmark: a reader who disagrees
with a choice can see exactly how many rows it affected.

The steps, in order:

1. **Unit harmonisation.** Affinities to a single p-scale.
2. **Relation filtering.** Censored measurements (``>``, ``<``) dropped or kept.
3. **Structure standardisation.** Salts, charges, tautomers; InChIKey assignment.
4. **Target resolution.** Accessions and sequences; family assignment.
5. **Scope.** Restrict to the three families and to small molecules.
6. **Aggregation.** Replicate measurements of a pair collapsed, or kept.

Aggregation is last on purpose. Collapsing replicates before measuring their
disagreement would destroy the evidence for the label-inconsistency finding, so
the audit runs on the pre-aggregation table and the models train on the
post-aggregation one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Multipliers converting a concentration unit to molar.
UNIT_TO_MOLAR: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "μM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

#: Raw measurement types mapped to their p-scale name.
TYPE_TO_P_TYPE: dict[str, str] = {
    "Ki": "pKi",
    "Kd": "pKd",
    "IC50": "pIC50",
    "EC50": "pEC50",
    "KI": "pKi",
    "KD": "pKd",
    "pKi": "pKi",
    "pKd": "pKd",
    "pIC50": "pIC50",
    "pEC50": "pEC50",
}


@dataclass
class CurationLog:
    """Row counts before and after each step, published with the benchmark."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, before: int, after: int, **detail: Any) -> None:
        self.steps.append(
            {
                "step": step,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "fraction_removed": round((before - after) / before, 5) if before else 0.0,
                **detail,
            }
        )
        logger.info(
            "%-28s %8d -> %8d (%+d)", step, before, after, after - before
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def to_p_scale(
    values: pd.Series,
    units: pd.Series,
    *,
    already_p: pd.Series | None = None,
) -> pd.Series:
    """Convert concentration measurements to a p-scale.

    ``pX = -log10(concentration in M)``. Values already on a p-scale are passed
    through unchanged when flagged by ``already_p``.

    Non-positive concentrations cannot be log-transformed and become ``NaN``; they
    are counted by the curation log rather than silently clipped, because a zero
    in an affinity column is a data error worth reporting.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    multipliers = units.astype(str).str.strip().map(UNIT_TO_MOLAR)

    unknown_units = units[multipliers.isna() & numeric.notna()].astype(str).value_counts()
    if len(unknown_units):
        logger.warning("unrecognised concentration units, rows dropped: %s", unknown_units.to_dict())

    molar = numeric * multipliers
    with np.errstate(divide="ignore", invalid="ignore"):
        converted = -np.log10(molar.where(molar > 0))

    if already_p is not None:
        converted = converted.where(~already_p.fillna(False), numeric)
    return converted.astype("float32")


def curate(
    raw: pd.DataFrame,
    *,
    drop_censored: bool = True,
    aggregate: str | None = "median",
    scope_families: bool = True,
    druglike_only: bool = True,
    min_year: int | None = None,
    standardize: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, CurationLog]:
    """Curate a raw table that already carries the pre-canonical column names.

    Source loaders are responsible for renaming their columns; this function owns
    the decisions that affect what the benchmark contains.

    ``n_jobs`` is passed to structure standardisation, which dominates the cost of
    this function: it is roughly fourteen single-threaded hours on the ChEMBL
    three-family extract.

    ``aggregate=None`` keeps replicate measurements, which is what the audit needs.
    ``aggregate="median"`` collapses them, which is what training needs. The
    pipeline calls this function twice with different settings and keeps both
    tables, so that the noise floor reported by the audit is measured on the data
    the models are actually trained on.
    """
    log = CurationLog()
    df = raw.copy()
    log.record("input", len(df), len(df))

    # --- affinity -----------------------------------------------------------
    before = len(df)
    df = df[pd.to_numeric(df[schema.Y], errors="coerce").notna()]
    log.record("drop_missing_affinity", before, len(df))

    before = len(df)
    y = pd.to_numeric(df[schema.Y], errors="coerce")
    df = df[(y >= 0) & (y <= 14)]
    log.record("drop_implausible_affinity", before, len(df), bounds="[0, 14] on the p-scale")

    if schema.Y_TYPE in df.columns:
        df[schema.Y_TYPE] = df[schema.Y_TYPE].map(lambda value: TYPE_TO_P_TYPE.get(str(value), str(value)))
        before = len(df)
        df = df[df[schema.Y_TYPE].isin(schema.AFFINITY_TYPES)]
        log.record("drop_unknown_affinity_type", before, len(df))

    if drop_censored and schema.Y_RELATION in df.columns:
        before = len(df)
        df = df[df[schema.Y_RELATION].astype(str).str.strip().isin(["=", "~", ""])]
        log.record(
            "drop_censored",
            before,
            len(df),
            note="'>' and '<' measurements are bounds, not values; keeping them biases RMSE",
        )

    if min_year is not None and schema.DOC_YEAR in df.columns:
        before = len(df)
        years = pd.to_numeric(df[schema.DOC_YEAR], errors="coerce")
        df = df[years.isna() | (years >= min_year)]
        log.record("drop_before_min_year", before, len(df), min_year=min_year)

    # --- structures ---------------------------------------------------------
    if standardize:
        from leakfree_dta.chem.standardize import standardize_series

        before = len(df)
        standardized = standardize_series(df[schema.SMILES], n_jobs=n_jobs)
        df = df.reset_index(drop=True)
        df[schema.SMILES] = standardized["smiles_std"]
        df[schema.LIGAND_KEY] = standardized["inchikey"]
        df[schema.LIGAND_SKELETON] = standardized["skeleton"]
        reasons = standardized.loc[~standardized["std_ok"], "std_reason"].value_counts().to_dict()
        df = df[standardized["std_ok"].to_numpy()]
        log.record("standardize_structures", before, len(df), failure_reasons=reasons)

    if druglike_only:
        from leakfree_dta.chem.standardize import is_druglike

        before = len(df)
        unique = df[[schema.LIGAND_KEY, schema.SMILES]].drop_duplicates(subset=[schema.LIGAND_KEY])
        keep = {
            key
            for key, smiles in zip(unique[schema.LIGAND_KEY], unique[schema.SMILES], strict=True)
            if is_druglike(smiles)
        }
        df = df[df[schema.LIGAND_KEY].isin(keep)]
        log.record(
            "drop_non_druglike",
            before,
            len(df),
            note="peptides, oligonucleotides and metal complexes; a different prediction task",
        )

    # --- targets ------------------------------------------------------------
    before = len(df)
    df = df[df[schema.TARGET_SEQUENCE].notna() & (df[schema.TARGET_SEQUENCE].astype(str).str.len() > 30)]
    log.record("drop_targets_without_sequence", before, len(df))

    if scope_families and schema.TARGET_FAMILY in df.columns:
        from leakfree_dta.prot.families import scope_to_families

        before = len(df)
        df = scope_to_families(df)
        log.record(
            "scope_to_three_families", before, len(df), families=list(schema.TARGET_FAMILIES)
        )

    df = schema.finalize(df)

    # --- aggregation --------------------------------------------------------
    if aggregate is not None:
        before = len(df)
        df = aggregate_replicates(df, how=aggregate)
        log.record("aggregate_replicates", before, len(df), how=aggregate)

    violations = schema.validate(df)
    if violations:
        logger.warning(
            "curated table still has schema violations: %s",
            "; ".join(str(violation) for violation in violations),
        )

    return df.reset_index(drop=True), log


def aggregate_replicates(df: pd.DataFrame, *, how: str = "median") -> pd.DataFrame:
    """Collapse replicate measurements of a pair into one row.

    The median is used rather than the mean because the conflicting-label check
    shows the replicate distribution has heavy tails: a single mis-keyed
    measurement three log units out would drag a mean but not a median.

    The aggregated row keeps the number of replicates and their spread, which the
    models' evaluation uses to weight rows by label reliability.
    """
    if how not in {"median", "mean", "first", "max"}:
        raise ValueError(f"unknown aggregation {how!r}")

    grouped = df.groupby(schema.PAIR_KEY, observed=True, sort=False)
    aggregated = grouped.agg(
        **{
            schema.Y: (schema.Y, how if how != "first" else "first"),
            "y_n_replicates": (schema.Y, "size"),
            "y_spread": (schema.Y, lambda values: float(values.max() - values.min())),
            "y_std": (schema.Y, "std"),
        }
    )

    # Carry the remaining columns from the first record of each pair. Aggregating
    # them would be meaningless (there is no "median SMILES"), and the first record
    # is well defined because the table is sorted by row key before this point.
    representative = (
        df.sort_values(schema.ROW_KEY)
        .drop_duplicates(subset=[schema.PAIR_KEY], keep="first")
        .set_index(schema.PAIR_KEY)
        .drop(columns=[schema.Y])
    )
    out = representative.join(aggregated, how="inner").reset_index()
    out["y_std"] = out["y_std"].fillna(0.0)
    return schema.finalize(out, compute_scaffold=False)
