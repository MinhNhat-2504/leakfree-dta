"""Assigning targets to the three families the paper is scoped to.

The three families are not an arbitrary sample. They were chosen because they fail
differently, which is the paper's actual finding:

``kinase``
    ATP-site conservation is high, so many targets are near-identical in the
    region ligands bind. Leakage here comes mostly through the **protein** axis.
``gpcr``
    Sparser data, and medicinal chemistry programmes explore tight analogue
    series. Leakage here comes mostly through the **ligand** axis.
``protease``
    Mixed substrate-like and allosteric chemotypes measured in heterogeneous
    assays. The dominant problem here is **label inconsistency** rather than
    overlap.

If one splitting rule were enough, all three would behave the same. Showing that
they do not is why the scope is three families rather than one.

Assignment is by provenance, in descending order of trust: a curated override
file, then the source database's own protein classification, then UniProt
keywords, then a sequence-motif heuristic. Every assignment records which rule
fired, and the audit reports the breakdown so a reviewer can discount the
heuristic ones.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

KINASE = "kinase"
GPCR = "gpcr"
PROTEASE = "protease"
OTHER = "other"
UNASSIGNED = "unassigned"

#: Patterns matched against a source database's protein class path, e.g. ChEMBL's
#: ``"Enzyme/Kinase/Protein Kinase/TK protein kinase group"``.
_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (KINASE, re.compile(r"\bprotein kinase\b|\bkinase\b", re.IGNORECASE)),
    (
        GPCR,
        re.compile(
            r"\b7tm\b|g[\s\-]?protein[\s\-]?coupled|\bgpcr\b|family a g protein",
            re.IGNORECASE,
        ),
    ),
    (
        PROTEASE,
        re.compile(
            r"\bprotease\b|\bpeptidase\b|\bproteinase\b|serine protease|"
            r"metallo[\s\-]?protease|aspartic protease|cysteine protease",
            re.IGNORECASE,
        ),
    ),
)

#: Patterns matched against UniProt keyword lists.
_KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (KINASE, re.compile(r"\b(?:transferase|kinase|serine/threonine-protein kinase|tyrosine-protein kinase)\b", re.IGNORECASE)),
    (GPCR, re.compile(r"g-protein coupled receptor|transducer", re.IGNORECASE)),
    (PROTEASE, re.compile(r"\b(?:protease|hydrolase|aspartyl protease|metalloprotease|thiol protease)\b", re.IGNORECASE)),
)

#: Kinase catalytic motifs. The HRD catalytic loop and the DFG magnesium-binding
#: motif are present in essentially every active eukaryotic protein kinase, and
#: the glycine-rich loop anchors the ATP phosphates.
_KINASE_MOTIFS = (
    re.compile(r"HRD[LIVMFYWC]"),
    re.compile(r"D[FLIV]G"),
    re.compile(r"G[ENQ]G[AQS][FY]G[ESKV]V"),
)


@dataclass(frozen=True)
class FamilyAssignment:
    """One target's family, plus how it was decided."""

    family: str
    rule: str
    confidence: str  # "curated" | "database" | "keyword" | "heuristic" | "none"


def assign_family(
    *,
    target_key: str,
    protein_class: str | None = None,
    uniprot_keywords: str | None = None,
    sequence: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> FamilyAssignment:
    """Assign one target, trying each source of evidence in order of trust."""
    if overrides and target_key in overrides:
        return FamilyAssignment(overrides[target_key], "override_file", "curated")

    if protein_class:
        for family, pattern in _CLASS_PATTERNS:
            if pattern.search(protein_class):
                return FamilyAssignment(family, f"protein_class:{pattern.pattern[:24]}", "database")

    if uniprot_keywords:
        for family, pattern in _KEYWORD_PATTERNS:
            if pattern.search(uniprot_keywords):
                return FamilyAssignment(family, f"keyword:{pattern.pattern[:24]}", "keyword")

    if sequence:
        heuristic = _assign_from_sequence(sequence)
        if heuristic is not None:
            return FamilyAssignment(heuristic, "sequence_motif", "heuristic")

    return FamilyAssignment(UNASSIGNED, "no_evidence", "none")


def _assign_from_sequence(sequence: str) -> str | None:
    """Motif heuristic. Only ever used when no annotation is available.

    Deliberately conservative: it claims a kinase only when at least two of the
    three catalytic motifs are present, and it does not attempt GPCRs at all,
    because reliable 7-transmembrane detection needs a topology predictor rather
    than a regular expression. Returning ``None`` is the right answer far more
    often than guessing.
    """
    upper = sequence.upper()
    n_kinase_motifs = sum(1 for pattern in _KINASE_MOTIFS if pattern.search(upper))
    if n_kinase_motifs >= 2:
        return KINASE
    return None


def assign_families(
    targets: pd.DataFrame,
    *,
    key_column: str = "target_key",
    class_column: str | None = "protein_class",
    keyword_column: str | None = "uniprot_keywords",
    sequence_column: str | None = "target_sequence",
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Assign families for a table of targets.

    Returns the input with ``target_family``, ``family_rule`` and
    ``family_confidence`` columns added. The confidence column is what the paper's
    supplementary table reports: a family split is only as trustworthy as the
    annotations it is built on.
    """
    records: list[FamilyAssignment] = []
    for row in targets.itertuples(index=False):
        records.append(
            assign_family(
                target_key=str(getattr(row, key_column)),
                protein_class=_maybe_str(getattr(row, class_column, None)) if class_column else None,
                uniprot_keywords=(
                    _maybe_str(getattr(row, keyword_column, None)) if keyword_column else None
                ),
                sequence=_maybe_str(getattr(row, sequence_column, None)) if sequence_column else None,
                overrides=overrides,
            )
        )

    out = targets.copy()
    out["target_family"] = [record.family for record in records]
    out["family_rule"] = [record.rule for record in records]
    out["family_confidence"] = [record.confidence for record in records]

    breakdown = out.groupby(["target_family", "family_confidence"], observed=True).size()
    logger.info("family assignment:\n%s", breakdown.to_string())

    n_unassigned = int((out["target_family"] == UNASSIGNED).sum())
    if n_unassigned:
        logger.warning(
            "%d/%d targets unassigned; add them to configs/data/target_families.csv "
            "or they will be dropped when the benchmark is scoped to the three families",
            n_unassigned,
            len(out),
        )
    return out


def load_overrides(path: str | None) -> dict[str, str]:
    """Read a curated ``target_key,target_family`` CSV.

    This file is the escape hatch for every target the automated rules get wrong,
    and it is committed to the repo so that the assignment is auditable rather
    than a side effect of whichever annotation snapshot was current.
    """
    if not path:
        return {}
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        logger.info("no family override file at %s", file)
        return {}
    table = pd.read_csv(file, comment="#")
    missing = {"target_key", "target_family"} - set(table.columns)
    if missing:
        raise ValueError(f"{file} must have columns target_key,target_family (missing {missing})")
    overrides = dict(zip(table["target_key"].astype(str), table["target_family"].astype(str), strict=True))
    logger.info("loaded %d curated family overrides from %s", len(overrides), file.name)
    return overrides


def scope_to_families(
    df: pd.DataFrame,
    families: Iterable[str] = (KINASE, GPCR, PROTEASE),
    *,
    family_column: str = "target_family",
) -> pd.DataFrame:
    """Restrict a table to the families in scope, logging what was dropped."""
    wanted = set(families)
    mask = df[family_column].isin(wanted)
    dropped = df.loc[~mask, family_column].value_counts().to_dict()
    if dropped:
        logger.info("scoping to %s dropped %d rows: %s", sorted(wanted), int((~mask).sum()), dropped)
    return df.loc[mask].reset_index(drop=True)


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NA:
        return None
    text = str(value).strip()
    return text or None
