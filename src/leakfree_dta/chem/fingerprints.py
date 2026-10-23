"""Molecular fingerprints and scaffolds.

Fingerprints serve two distinct purposes here and the distinction matters:

* As **features** for the ligand-only baselines and the shortcut probe.
* As a **similarity metric** for near-duplicate detection and Butina clustering.

Both use ECFP4 (Morgan radius 2), the field's default, so that the audit's notion
of "near-duplicate" matches the one readers already have in mind.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def _generator(radius: int, n_bits: int) -> rdFingerprintGenerator.FingerprintGenerator64:
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def ecfp_bitvects(
    smiles: Iterable[str | None],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> list[DataStructs.ExplicitBitVect | None]:
    """RDKit bit vectors, for similarity calculations.

    Kept as native bit vectors rather than NumPy arrays because RDKit's bulk
    Tanimoto is far faster on them than any array-based equivalent.
    """
    generator = _generator(radius, n_bits)
    out: list[DataStructs.ExplicitBitVect | None] = []
    cache: dict[str | None, DataStructs.ExplicitBitVect | None] = {}
    for value in smiles:
        if value not in cache:
            mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
            cache[value] = generator.GetFingerprint(mol) if mol is not None else None
        out.append(cache[value])
    return out


def ecfp_matrix(
    smiles: Iterable[str | None],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    dtype: type = np.float32,
) -> np.ndarray:
    """A dense ``(n_molecules, n_bits)`` feature matrix.

    Unparseable molecules become all-zero rows so that row indices stay aligned
    with the input table; callers that need to exclude them should filter on
    ``std_ok`` from standardisation instead of inspecting the matrix.
    """
    vectors = ecfp_bitvects(smiles, radius=radius, n_bits=n_bits)
    matrix = np.zeros((len(vectors), n_bits), dtype=dtype)
    n_missing = 0
    for row, vector in enumerate(vectors):
        if vector is None:
            n_missing += 1
            continue
        DataStructs.ConvertToNumpyArray(vector, matrix[row])
    if n_missing:
        logger.warning("%d/%d molecules produced no fingerprint", n_missing, len(vectors))
    return matrix


def murcko_scaffolds(smiles: Iterable[str | None], *, generic: bool = False) -> list[str | None]:
    """Bemis-Murcko scaffolds as canonical SMILES.

    ``generic=True`` additionally erases atom and bond types, giving the coarser
    graph-framework scaffold. The audit reports scaffold overlap at both levels,
    because the generic one catches analogue series that the standard scaffold
    splits apart -- a channel most published scaffold splits leave open.
    """
    out: list[str | None] = []
    cache: dict[str | None, str | None] = {}
    for value in smiles:
        if value not in cache:
            cache[value] = _one_scaffold(value, generic=generic)
        out.append(cache[value])
    return out


def _one_scaffold(smiles: str | None, *, generic: bool) -> str | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        result = Chem.MolToSmiles(scaffold)
    except Exception:
        return None
    # Acyclic molecules have an empty Murcko scaffold. Treating them as one giant
    # shared scaffold would be wrong, so they are reported as their own group.
    return result if result else "acyclic"


def descriptor_frame(smiles: Sequence[str | None]) -> pd.DataFrame:
    """A handful of interpretable descriptors.

    Used for the distribution-shift diagnostics -- showing *how* a leakage-free
    test set differs from its training set, in terms a medicinal chemist reads
    directly -- not as model features.
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    records: list[dict[str, float]] = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if mol is None:
            records.append(dict.fromkeys(_DESCRIPTOR_NAMES, float("nan")))
            continue
        records.append(
            {
                "mw": Descriptors.MolWt(mol),
                "clogp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                "heavy_atoms": float(mol.GetNumHeavyAtoms()),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            }
        )
    return pd.DataFrame.from_records(records)


_DESCRIPTOR_NAMES = (
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "stereocenters",
    "heavy_atoms",
    "fraction_csp3",
)
