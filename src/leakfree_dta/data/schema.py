"""The canonical interaction table.

Every source -- ChEMBL, BindingDB, Davis, KIBA, PDBbind -- is normalised into one
long-format table with the columns below. Fixing the schema early is what makes
the audit meaningful: a leakage check can only compare two datasets if "the same
ligand" and "the same target" mean the same thing in both.

Two choices here are load-bearing for the paper:

``ligand_key`` is an InChIKey, not a SMILES string
    SMILES are not canonical across toolkits or across source databases, so
    matching on them silently misses duplicates -- one of the inconsistencies the
    2026 Chemical Science audit reports. The first block of the InChIKey (the
    skeleton layer) is kept separately as ``ligand_skeleton`` so that
    stereoisomers can be grouped when a check calls for it.

``y`` is always a p-scale affinity, higher meaning tighter binding
    Mixing Ki, Kd and IC50 on a single scale is a real approximation, and the
    paper has to defend it. ``y_type`` keeps the provenance so the audit can
    report results with and without IC50 included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

# --- Column names -----------------------------------------------------------

LIGAND_KEY: Final = "ligand_key"
LIGAND_SKELETON: Final = "ligand_skeleton"
SMILES: Final = "smiles"
SCAFFOLD: Final = "murcko_scaffold"

TARGET_KEY: Final = "target_key"
TARGET_SEQUENCE: Final = "target_sequence"
TARGET_FAMILY: Final = "target_family"
TARGET_SUBFAMILY: Final = "target_subfamily"

Y: Final = "y"
Y_TYPE: Final = "y_type"
Y_RELATION: Final = "y_relation"

SOURCE: Final = "source"
DOC_YEAR: Final = "doc_year"
ASSAY_KEY: Final = "assay_key"
CONFIDENCE: Final = "confidence"

PAIR_KEY: Final = "pair_key"
ROW_KEY: Final = "row_key"

#: Columns every curated table must carry.
REQUIRED_COLUMNS: Final = (
    LIGAND_KEY,
    SMILES,
    TARGET_KEY,
    TARGET_SEQUENCE,
    Y,
    Y_TYPE,
    SOURCE,
)

#: Columns added by :func:`finalize`, derivable from the required ones.
DERIVED_COLUMNS: Final = (LIGAND_SKELETON, SCAFFOLD, PAIR_KEY, ROW_KEY)

#: Columns that may be absent; the pipeline fills a documented default.
OPTIONAL_DEFAULTS: Final = {
    TARGET_FAMILY: "unassigned",
    TARGET_SUBFAMILY: "unassigned",
    Y_RELATION: "=",
    DOC_YEAR: pd.NA,
    ASSAY_KEY: pd.NA,
    CONFIDENCE: pd.NA,
}

#: Affinity types accepted on the shared p-scale.
AFFINITY_TYPES: Final = ("pKi", "pKd", "pIC50", "pEC50", "pKi_app")

#: Target families the paper scopes itself to. See ``docs`` in the README for why
#: these three: they fail in different ways, so together they show that no single
#: splitting rule is sufficient.
TARGET_FAMILIES: Final = ("kinase", "gpcr", "protease")


@dataclass(frozen=True)
class SchemaViolation:
    """One problem found by :func:`validate`."""

    column: str
    problem: str
    n_rows: int

    def __str__(self) -> str:
        return f"[{self.column}] {self.problem} ({self.n_rows} rows)"


def make_pair_key(ligand_key: pd.Series, target_key: pd.Series) -> pd.Series:
    """The identity of a (ligand, target) measurement, independent of source.

    Cross-source duplicate detection and cross-split pair overlap both key on
    this, so it is defined once here rather than inline at each use.
    """
    return ligand_key.astype(str) + "||" + target_key.astype(str)


def make_row_key(df: pd.DataFrame) -> pd.Series:
    """A stable per-row identifier.

    A pair can legitimately appear more than once -- different assays, different
    papers -- so the row key includes the source and assay. Split manifests
    reference rows by this key, which is why it must not depend on row order.
    """
    assay = df[ASSAY_KEY].astype(str) if ASSAY_KEY in df.columns else pd.Series("na", index=df.index)
    return (
        df[LIGAND_KEY].astype(str)
        + "||"
        + df[TARGET_KEY].astype(str)
        + "||"
        + df[SOURCE].astype(str)
        + "||"
        + assay.fillna("na")
    )


def empty_frame() -> pd.DataFrame:
    """An empty table with the full schema, for sources that yield no rows."""
    columns = list(REQUIRED_COLUMNS) + list(OPTIONAL_DEFAULTS) + list(DERIVED_COLUMNS)
    return pd.DataFrame({column: pd.Series(dtype="object") for column in dict.fromkeys(columns)})


def finalize(df: pd.DataFrame, *, compute_scaffold: bool = True) -> pd.DataFrame:
    """Fill optional columns, derive keys, and order columns canonically.

    Called at the end of every source loader so that downstream code can assume
    the schema holds. ``compute_scaffold=False`` skips the RDKit pass, which is
    the slow part, for callers that only need the keys.
    """
    df = df.copy()

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    if LIGAND_SKELETON not in df.columns:
        # InChIKey layout is SKELETON-STEREOPROTO-CHARGE; the first block ignores
        # stereochemistry, which is the right granularity for duplicate hunting.
        df[LIGAND_SKELETON] = df[LIGAND_KEY].astype(str).str.split("-").str[0]

    if compute_scaffold and SCAFFOLD not in df.columns:
        from leakfree_dta.chem.fingerprints import murcko_scaffolds

        df[SCAFFOLD] = murcko_scaffolds(df[SMILES])
    elif SCAFFOLD not in df.columns:
        df[SCAFFOLD] = pd.NA

    df[PAIR_KEY] = make_pair_key(df[LIGAND_KEY], df[TARGET_KEY])
    df[ROW_KEY] = make_row_key(df)

    if DOC_YEAR in df.columns:
        df[DOC_YEAR] = pd.to_numeric(df[DOC_YEAR], errors="coerce").astype("Int16")
    df[Y] = pd.to_numeric(df[Y], errors="coerce").astype("float32")

    ordered = [
        ROW_KEY,
        LIGAND_KEY,
        LIGAND_SKELETON,
        SMILES,
        SCAFFOLD,
        TARGET_KEY,
        TARGET_FAMILY,
        TARGET_SUBFAMILY,
        TARGET_SEQUENCE,
        Y,
        Y_TYPE,
        Y_RELATION,
        SOURCE,
        DOC_YEAR,
        ASSAY_KEY,
        CONFIDENCE,
        PAIR_KEY,
    ]
    rest = [column for column in df.columns if column not in ordered]
    return df[[column for column in ordered if column in df.columns] + rest]


def validate(df: pd.DataFrame, *, strict: bool = False) -> list[SchemaViolation]:
    """Check a curated table against the schema.

    Returns the violations rather than raising, because the audit reports them as
    findings: a source that fails validation is itself a result worth tabulating.
    ``strict=True`` raises instead, for use in tests and at pipeline boundaries.
    """
    violations: list[SchemaViolation] = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            violations.append(SchemaViolation(column, "required column missing", len(df)))

    if violations and strict:
        raise ValueError("Schema validation failed:\n  " + "\n  ".join(map(str, violations)))
    if violations:
        return violations

    for column in (LIGAND_KEY, TARGET_KEY, SMILES, TARGET_SEQUENCE):
        n_null = int(df[column].isna().sum())
        if n_null:
            violations.append(SchemaViolation(column, "null values", n_null))

    y = pd.to_numeric(df[Y], errors="coerce")
    n_nan = int(y.isna().sum())
    if n_nan:
        violations.append(SchemaViolation(Y, "non-numeric or missing affinity", n_nan))

    # pAffinity outside [0, 14] is almost always a unit conversion error upstream:
    # pKd 0 is 1 M and pKd 14 is 10 fM, neither of which is a real measurement.
    finite = y.dropna()
    n_out = int(((finite < 0) | (finite > 14)).sum())
    if n_out:
        violations.append(SchemaViolation(Y, "affinity outside plausible range [0, 14]", n_out))

    bad_types = set(df[Y_TYPE].dropna().unique()) - set(AFFINITY_TYPES)
    if bad_types:
        n_rows = int(df[Y_TYPE].isin(bad_types).sum())
        violations.append(
            SchemaViolation(Y_TYPE, f"unrecognised affinity types {sorted(bad_types)}", n_rows)
        )

    if TARGET_FAMILY in df.columns:
        bad_families = set(df[TARGET_FAMILY].dropna().unique()) - set(TARGET_FAMILIES) - {
            "unassigned",
            "other",
        }
        if bad_families:
            n_rows = int(df[TARGET_FAMILY].isin(bad_families).sum())
            violations.append(
                SchemaViolation(
                    TARGET_FAMILY, f"families outside project scope {sorted(bad_families)}", n_rows
                )
            )

    # A sequence that is not protein-like means the target table was joined wrong.
    if TARGET_SEQUENCE in df.columns:
        sequences = df[TARGET_SEQUENCE].dropna().astype(str)
        suspicious = sequences[~sequences.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZUO]+")]
        if len(suspicious):
            violations.append(
                SchemaViolation(TARGET_SEQUENCE, "non-amino-acid characters", len(suspicious))
            )

    if strict and violations:
        raise ValueError("Schema validation failed:\n  " + "\n  ".join(map(str, violations)))
    return violations


def summarize(df: pd.DataFrame) -> dict[str, object]:
    """Counts used in the paper's dataset table and printed by ``leakfree curate``."""
    summary: dict[str, object] = {
        "n_measurements": len(df),
        "n_unique_pairs": int(df[PAIR_KEY].nunique()) if PAIR_KEY in df.columns else None,
        "n_ligands": int(df[LIGAND_KEY].nunique()),
        "n_targets": int(df[TARGET_KEY].nunique()),
    }
    if SCAFFOLD in df.columns:
        summary["n_scaffolds"] = int(df[SCAFFOLD].nunique(dropna=True))
    if TARGET_FAMILY in df.columns:
        summary["by_family"] = {
            str(family): int(count) for family, count in df[TARGET_FAMILY].value_counts().items()
        }
    if Y_TYPE in df.columns:
        summary["by_y_type"] = {
            str(kind): int(count) for kind, count in df[Y_TYPE].value_counts().items()
        }
    if SOURCE in df.columns:
        summary["by_source"] = {
            str(source): int(count) for source, count in df[SOURCE].value_counts().items()
        }
    y = pd.to_numeric(df[Y], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(y).any():
        summary["y_mean"] = round(float(np.nanmean(y)), 4)
        summary["y_std"] = round(float(np.nanstd(y)), 4)
        summary["y_min"] = round(float(np.nanmin(y)), 4)
        summary["y_max"] = round(float(np.nanmax(y)), 4)
    if DOC_YEAR in df.columns and df[DOC_YEAR].notna().any():
        years = pd.to_numeric(df[DOC_YEAR], errors="coerce").dropna()
        summary["year_min"] = int(years.min())
        summary["year_max"] = int(years.max())
    return summary
