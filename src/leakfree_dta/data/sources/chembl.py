"""ChEMBL: the reference source for this benchmark.

ChEMBL is used as the primary source rather than one of the packaged benchmarks
because it carries the two fields the paper cannot do without:

* ``document_year``, without which no temporal split is possible. Davis and KIBA
  ship without dates, which is why no honest prospective evaluation has been done
  on them.
* ``assay_id`` and ``confidence_score``, which make the label-inconsistency
  measurement possible -- two measurements of one pair can be traced to the assays
  that disagree.

Expected input
--------------
A table extracted from the ChEMBL release with one activity per row. Obtain it
either from the SQLite/PostgreSQL dump with ``sql/chembl_activities.sql`` in this
repo, or from the web services. Required columns:

``canonical_smiles``, ``target_uniprot``, ``target_sequence``,
``standard_type``, ``standard_relation``, ``standard_value``, ``standard_units``,
``pchembl_value``, ``document_year``, ``assay_id``, ``confidence_score``,
``target_protein_class`` (optional but strongly preferred -- it drives family
assignment, and the sequence-motif fallback is far less reliable).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.data.curate import to_p_scale
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import read_table

logger = get_logger(__name__)

SOURCE_NAME = "chembl"

#: Activities below this ChEMBL confidence score are not reliably assigned to a
#: single protein target. 8 is "homologous single protein"; 9 is "direct single
#: protein". Keeping 8 and above is the usual choice in the DTA literature.
MIN_CONFIDENCE = 8

REQUIRED_RAW_COLUMNS = (
    "canonical_smiles",
    "target_uniprot",
    "standard_type",
    "standard_value",
    "standard_units",
)


def load(
    path: str | Path,
    *,
    min_confidence: int = MIN_CONFIDENCE,
    prefer_pchembl: bool = True,
) -> pd.DataFrame:
    """Load a ChEMBL activity extract into pre-canonical form.

    ``prefer_pchembl`` uses ChEMBL's own ``pchembl_value`` where available and
    falls back to converting ``standard_value`` otherwise. ChEMBL's conversion
    already handles several unit edge cases, so trusting it where present reduces
    the number of decisions this pipeline has to defend.
    """
    raw = read_table(path)
    missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"{Path(path).name} is missing required ChEMBL columns {sorted(missing)}. "
            f"See the module docstring for the expected extract."
        )

    logger.info("loaded %d raw ChEMBL activities from %s", len(raw), Path(path).name)

    if "confidence_score" in raw.columns and min_confidence is not None:
        before = len(raw)
        raw = raw[pd.to_numeric(raw["confidence_score"], errors="coerce") >= min_confidence]
        logger.info(
            "confidence >= %d kept %d/%d activities", min_confidence, len(raw), before
        )

    has_pchembl = prefer_pchembl and "pchembl_value" in raw.columns
    pchembl = pd.to_numeric(raw["pchembl_value"], errors="coerce") if has_pchembl else None
    converted = to_p_scale(raw["standard_value"], raw["standard_units"])
    y = converted if pchembl is None else pchembl.fillna(converted)
    if pchembl is not None:
        logger.info(
            "affinity source: %d rows from pchembl_value, %d converted from standard_value",
            int(pchembl.notna().sum()),
            int((pchembl.isna() & converted.notna()).sum()),
        )

    out = pd.DataFrame(
        {
            schema.SMILES: raw["canonical_smiles"].astype(str),
            schema.TARGET_KEY: raw["target_uniprot"].astype(str).str.strip().str.upper(),
            schema.TARGET_SEQUENCE: raw.get("target_sequence"),
            schema.Y: y,
            schema.Y_TYPE: raw["standard_type"].astype(str),
            schema.Y_RELATION: raw.get("standard_relation", pd.Series("=", index=raw.index)),
            schema.SOURCE: SOURCE_NAME,
            schema.DOC_YEAR: raw.get("document_year"),
            schema.ASSAY_KEY: raw.get("assay_id"),
            schema.CONFIDENCE: raw.get("confidence_score"),
        }
    )
    if "target_protein_class" in raw.columns:
        out["protein_class"] = raw["target_protein_class"]
    if "target_uniprot_keywords" in raw.columns:
        out["uniprot_keywords"] = raw["target_uniprot_keywords"]

    return out.reset_index(drop=True)


def resolve_sequences(df: pd.DataFrame, *, fetch_missing: bool = True) -> pd.DataFrame:
    """Fill ``target_sequence`` from UniProt where the extract did not supply it."""
    if schema.TARGET_SEQUENCE in df.columns and df[schema.TARGET_SEQUENCE].notna().all():
        return df

    accessions = sorted(df[schema.TARGET_KEY].dropna().astype(str).unique())
    if not fetch_missing:
        logger.warning("%d targets lack sequences and fetching is disabled", len(accessions))
        return df

    from leakfree_dta.prot.sequences import fetch_sequences, is_uniprot_accession

    valid = [accession for accession in accessions if is_uniprot_accession(accession)]
    if len(valid) < len(accessions):
        logger.warning(
            "%d of %d target keys are not UniProt accessions and cannot be resolved",
            len(accessions) - len(valid),
            len(accessions),
        )

    resolved = fetch_sequences(valid)
    out = df.copy()
    filled = out[schema.TARGET_KEY].map(resolved)
    out[schema.TARGET_SEQUENCE] = (
        out[schema.TARGET_SEQUENCE].fillna(filled) if schema.TARGET_SEQUENCE in out else filled
    )
    return out
