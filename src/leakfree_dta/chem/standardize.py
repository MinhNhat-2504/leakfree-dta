"""Structure standardisation.

The audit's central claim is that published benchmarks contain duplicates they do
not know about. That claim is only as strong as the normalisation used to detect
them, so the rules applied here are stated explicitly and are configurable:

1. Parse and sanitise; drop anything RDKit rejects.
2. Keep the largest organic fragment (strips counterions and solvates).
3. Neutralise charges where a neutral form exists.
4. Canonicalise the tautomer.
5. Emit a canonical SMILES and an InChIKey.

Steps 2-4 each merge records that some sources treat as distinct compounds. The
paper reports duplicate counts at three normalisation levels -- raw SMILES,
InChIKey, and skeleton-only -- so a reader can see how much of the measured
duplication depends on these choices rather than taking them on faith.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

# RDKit logs a warning per unparseable SMILES; at ChEMBL scale that is unusable.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StandardizedMolecule:
    """The result of standardising one input string."""

    smiles: str | None
    inchikey: str | None
    skeleton: str | None
    ok: bool
    reason: str = ""


_FAILED = StandardizedMolecule(None, None, None, False, "parse_error")


@lru_cache(maxsize=1)
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    enumerator = rdMolStandardize.TautomerEnumerator()
    # The default cap of 1000 can take seconds on peptide-like protease ligands.
    enumerator.SetMaxTautomers(128)
    # Stereochemistry must survive tautomer canonicalisation. By default RDKit
    # strips sp3 stereo at atoms it considers potential tautomer centres, which
    # silently merges enantiomers into one compound. For an affinity benchmark that
    # is not an acceptable normalisation: enantiomers routinely differ in potency by
    # orders of magnitude, and collapsing them would both lose real signal and
    # manufacture label conflicts that the audit would then report as assay noise.
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    return enumerator


@lru_cache(maxsize=1)
def _fragment_chooser() -> rdMolStandardize.LargestFragmentChooser:
    return rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def standardize_smiles(
    smiles: str | None,
    *,
    strip_salts: bool = True,
    neutralize: bool = True,
    canonical_tautomer: bool = True,
) -> StandardizedMolecule:
    """Standardise one SMILES string.

    Returns a result object rather than raising: at benchmark scale a few per cent
    of records fail to parse, and the audit wants to count them, not crash on
    them.
    """
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return StandardizedMolecule(None, None, None, False, "empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _FAILED

    try:
        mol = rdMolStandardize.Cleanup(mol)
        if strip_salts:
            mol = _fragment_chooser().choose(mol)
        if neutralize:
            mol = _uncharger().uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer_enumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises a variety of C++-backed exceptions
        return StandardizedMolecule(None, None, None, False, f"standardize_error:{type(exc).__name__}")

    if mol.GetNumHeavyAtoms() == 0:
        return StandardizedMolecule(None, None, None, False, "no_heavy_atoms")

    canonical = Chem.MolToSmiles(mol)
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        key = None
    if not key:
        return StandardizedMolecule(canonical, None, None, False, "inchikey_error")

    return StandardizedMolecule(canonical, key, key.split("-")[0], True)


def standardize_series(
    smiles: Iterable[str | None],
    *,
    strip_salts: bool = True,
    neutralize: bool = True,
    canonical_tautomer: bool = True,
) -> pd.DataFrame:
    """Standardise many SMILES, caching repeated inputs.

    Benchmark tables repeat the same ligand across dozens of targets, so the cache
    does most of the work: on a ChEMBL kinase extract it cuts the pass from tens of
    minutes to a few.
    """
    values: Sequence[str | None] = list(smiles)
    cache: dict[str | None, StandardizedMolecule] = {}
    rows: list[StandardizedMolecule] = []

    for value in values:
        if value not in cache:
            cache[value] = standardize_smiles(
                value,
                strip_salts=strip_salts,
                neutralize=neutralize,
                canonical_tautomer=canonical_tautomer,
            )
        rows.append(cache[value])

    n_failed = sum(not row.ok for row in rows)
    if n_failed:
        logger.warning(
            "standardisation dropped %d/%d records (%.2f%%); reasons: %s",
            n_failed,
            len(rows),
            100 * n_failed / max(len(rows), 1),
            dict(pd.Series([r.reason for r in rows if not r.ok]).value_counts()),
        )

    return pd.DataFrame(
        {
            "smiles_std": [row.smiles for row in rows],
            "inchikey": [row.inchikey for row in rows],
            "skeleton": [row.skeleton for row in rows],
            "std_ok": [row.ok for row in rows],
            "std_reason": [row.reason for row in rows],
        }
    )


def is_druglike(
    smiles: str | None,
    *,
    max_heavy_atoms: int = 100,
    min_heavy_atoms: int = 6,
    allowed_elements: frozenset[str] = frozenset(
        {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si", "Se"}
    ),
) -> bool:
    """A coarse filter for records that are not small molecules.

    Benchmarks derived from ChEMBL carry peptides, oligonucleotides and metal
    complexes alongside drug-like compounds. They are not errors, but a model
    scored on them is being asked a different question, so the paper reports
    results with and without them.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if not min_heavy_atoms <= n_heavy <= max_heavy_atoms:
        return False
    return all(atom.GetSymbol() in allowed_elements for atom in mol.GetAtoms())
