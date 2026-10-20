"""The packaged benchmarks the field actually reports on: Davis, KIBA, PDBbind.

These are loaded not because they are good, but because they are what a decade of
papers measured themselves against. The paper's re-evaluation needs them in the
same canonical form as the ChEMBL extract so that the audit can be applied to the
*published* benchmarks, not just to a freshly built one.

Each has a known limitation that the audit quantifies rather than asserts:

Davis
    72 kinase inhibitors against 442 kinases, fully measured. Every ligand appears
    against every target, so any random split places every test ligand and every
    test target in training. It is a matrix-completion benchmark described as a
    prediction benchmark.
KIBA
    Larger and sparser, with a derived score that mixes Ki, Kd and IC50 through a
    model-based adjustment -- so its labels are not a measured quantity.
PDBbind
    Has structures, which makes it the basis of most structure-based scoring
    comparisons, and has documented train/test contamination in its standard
    core-set split.

Expected inputs are noted on each loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import read_json, read_table

logger = get_logger(__name__)


def load_davis(
    directory: str | Path,
    *,
    affinity_file: str = "Y",
    ligand_file: str = "ligands_can.txt",
    target_file: str = "proteins.txt",
) -> pd.DataFrame:
    """Load the Davis kinase dataset from the DeepDTA distribution layout.

    The DeepDTA release stores affinities as a pickled matrix plus two JSON
    dictionaries. Kd values in nM are converted to pKd; the widely used
    ``-log10(Kd/1e9)`` convention is applied, matching what DeepDTA and its
    successors report, so the numbers here are comparable to published ones.
    """
    directory = Path(directory)
    ligands = read_json(directory / ligand_file)
    targets = read_json(directory / target_file)
    matrix = _load_affinity_matrix(directory / affinity_file)

    if matrix.shape != (len(ligands), len(targets)):
        raise ValueError(
            f"Davis affinity matrix is {matrix.shape} but there are {len(ligands)} ligands "
            f"and {len(targets)} targets"
        )

    ligand_keys = list(ligands)
    target_keys = list(targets)
    rows, cols = np.nonzero(np.isfinite(matrix))

    out = pd.DataFrame(
        {
            schema.SMILES: [str(ligands[ligand_keys[row]]) for row in rows],
            schema.TARGET_KEY: [str(target_keys[col]) for col in cols],
            schema.TARGET_SEQUENCE: [str(targets[target_keys[col]]) for col in cols],
            schema.Y: -np.log10(matrix[rows, cols] / 1e9),
            schema.Y_TYPE: "pKd",
            schema.Y_RELATION: "=",
            schema.SOURCE: "davis",
            schema.TARGET_FAMILY: "kinase",
        }
    )
    logger.info(
        "Davis: %d measurements over %d ligands and %d targets (density %.1f%%)",
        len(out),
        len(ligand_keys),
        len(target_keys),
        100 * len(out) / (len(ligand_keys) * len(target_keys)),
    )
    return out


def load_kiba(
    directory: str | Path,
    *,
    affinity_file: str = "Y",
    ligand_file: str = "ligands_can.txt",
    target_file: str = "proteins.txt",
) -> pd.DataFrame:
    """Load KIBA from the DeepDTA distribution layout.

    KIBA scores are kept as-is and labelled ``pKi_app`` -- an apparent affinity --
    rather than being passed off as a pKi. The distinction matters for the paper's
    argument: a derived score computed by a model cannot be used to validate
    another model without circularity, and that is worth saying plainly.
    """
    directory = Path(directory)
    ligands = read_json(directory / ligand_file)
    targets = read_json(directory / target_file)
    matrix = _load_affinity_matrix(directory / affinity_file)

    ligand_keys = list(ligands)
    target_keys = list(targets)
    rows, cols = np.nonzero(np.isfinite(matrix))

    out = pd.DataFrame(
        {
            schema.SMILES: [str(ligands[ligand_keys[row]]) for row in rows],
            schema.TARGET_KEY: [str(target_keys[col]) for col in cols],
            schema.TARGET_SEQUENCE: [str(targets[target_keys[col]]) for col in cols],
            schema.Y: matrix[rows, cols],
            schema.Y_TYPE: "pKi_app",
            schema.Y_RELATION: "=",
            schema.SOURCE: "kiba",
            schema.TARGET_FAMILY: "kinase",
        }
    )
    logger.info("KIBA: %d measurements over %d ligands and %d targets", len(out), len(ligand_keys), len(target_keys))
    return out


def _load_affinity_matrix(path: Path) -> np.ndarray:
    """Read the affinity matrix, accepting the formats the releases ship in."""
    if path.suffix in {".npy"}:
        return np.load(path)
    if path.suffix in {".csv", ".tsv"}:
        return read_table(path).to_numpy(dtype=float)
    # The DeepDTA release ships an extensionless pickle.
    import pickle

    with path.open("rb") as handle:
        loaded = pickle.load(handle, encoding="latin1")
    return np.asarray(loaded, dtype=float)


def load_pdbbind(
    index_file: str | Path,
    *,
    sequences: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a PDBbind index file (``INDEX_general_PL_data.*``).

    PDBbind's index gives a PDB code, a resolution, a year, and a ``-logKd/Ki``
    value, but not sequences or SMILES. Those must be supplied separately -- from
    the structure files or from a mapping table -- which is why this loader takes
    ``sequences`` and leaves SMILES to the caller. The ``year`` column is PDBbind's
    deposition year, which makes a genuine temporal split possible here, unlike
    Davis and KIBA.
    """
    index_file = Path(index_file)
    records: list[dict[str, object]] = []
    with index_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                records.append(
                    {
                        "pdb_code": fields[0].lower(),
                        "resolution": fields[1],
                        "year": int(fields[2]),
                        schema.Y: float(fields[3]),
                    }
                )
            except (ValueError, IndexError):
                continue

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError(f"no usable records parsed from {index_file.name}")

    out[schema.TARGET_KEY] = out["pdb_code"]
    out[schema.TARGET_SEQUENCE] = (
        out["pdb_code"].map(sequences) if sequences else pd.Series(pd.NA, index=out.index)
    )
    out[schema.SMILES] = pd.NA
    out[schema.Y_TYPE] = "pKd"
    out[schema.Y_RELATION] = "="
    out[schema.SOURCE] = "pdbbind"
    out[schema.DOC_YEAR] = out["year"]

    logger.info(
        "PDBbind: %d complexes, years %d-%d%s",
        len(out),
        int(out["year"].min()),
        int(out["year"].max()),
        "" if sequences else " (no sequences supplied; pass `sequences=` to make them usable)",
    )
    return out


def density_report(df: pd.DataFrame) -> dict[str, float | int]:
    """Matrix density, the statistic that explains why Davis is easy.

    A fully measured ligand-by-target matrix cannot be split so that either axis
    is held out, because every ligand appears against every target. Reporting
    density next to every benchmark's headline score is the shortest way to show
    which benchmarks can support a generalisation claim at all.
    """
    n_ligands = int(df[schema.LIGAND_KEY].nunique()) if schema.LIGAND_KEY in df else int(df[schema.SMILES].nunique())
    n_targets = int(df[schema.TARGET_KEY].nunique())
    cells = n_ligands * n_targets
    return {
        "n_measurements": len(df),
        "n_ligands": n_ligands,
        "n_targets": n_targets,
        "density": round(len(df) / cells, 5) if cells else 0.0,
        "measurements_per_ligand": round(len(df) / n_ligands, 2) if n_ligands else 0.0,
        "measurements_per_target": round(len(df) / n_targets, 2) if n_targets else 0.0,
    }
