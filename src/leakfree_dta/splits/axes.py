"""The three axes, each reduced to a group labelling.

Every axis works the same way: it partitions rows into groups that must not be
split, and then whole groups go to one side of the boundary. What differs is what
counts as a group.

The axis options are ordered from permissive to strict, and the paper reports the
whole ladder rather than picking one, because the drop from one rung to the next is
the measurement of interest.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Axis settings that impose no constraint. Rows are free to fall either side.
WILDCARD = "random"

LIGAND_AXES = (WILDCARD, "identity", "skeleton", "scaffold", "generic_scaffold", "cluster")
PROTEIN_AXES = (WILDCARD, "identity", "cluster", "family")
TEMPORAL_AXES = ("none", "year")


def ligand_groups(
    df: pd.DataFrame, axis: str, *, cluster_cutoff: float = 0.4
) -> pd.Series | None:
    """Group label per row for the ligand axis, or ``None`` for no constraint.

    ``identity``
        One group per InChIKey. Stops the same compound appearing on both sides.
    ``skeleton``
        One group per InChIKey skeleton, so stereoisomers stay together.
    ``scaffold``
        Bemis-Murcko scaffold -- the field's usual "hard" split. The audit shows
        this leaves the near-duplicate channel open, because a single ring-atom
        change produces a new scaffold.
    ``generic_scaffold``
        Scaffold with atom and bond types erased. Catches analogue series that
        plain scaffold splitting separates.
    ``cluster``
        Butina / sphere-exclusion clusters at a Tanimoto distance cutoff. The
        strictest setting, and the one the leakage-free benchmark uses.
    """
    if axis == WILDCARD:
        return None
    if axis == "identity":
        return df[schema.LIGAND_KEY].astype(str).rename("ligand_group")
    if axis == "skeleton":
        return df[schema.LIGAND_SKELETON].astype(str).rename("ligand_group")
    if axis == "scaffold":
        return df[schema.SCAFFOLD].fillna("unknown").astype(str).rename("ligand_group")
    if axis == "generic_scaffold":
        from leakfree_dta.chem.fingerprints import murcko_scaffolds

        unique = df[[schema.LIGAND_KEY, schema.SMILES]].drop_duplicates(subset=[schema.LIGAND_KEY])
        generic = murcko_scaffolds(unique[schema.SMILES], generic=True)
        mapping = dict(zip(unique[schema.LIGAND_KEY], generic, strict=True))
        return (
            df[schema.LIGAND_KEY].map(mapping).fillna("unknown").astype(str).rename("ligand_group")
        )
    if axis == "cluster":
        return _ligand_cluster_groups(df, cutoff=cluster_cutoff)
    raise ValueError(f"unknown ligand axis {axis!r}; expected one of {LIGAND_AXES}")


def _ligand_cluster_groups(df: pd.DataFrame, *, cutoff: float) -> pd.Series:
    from leakfree_dta.chem.fingerprints import ecfp_bitvects
    from leakfree_dta.chem.similarity import cluster_ligands

    # Sorting by InChIKey makes sphere exclusion order-independent, so the split
    # does not depend on how the dataset table happened to be assembled.
    unique = (
        df[[schema.LIGAND_KEY, schema.SMILES]]
        .drop_duplicates(subset=[schema.LIGAND_KEY])
        .sort_values(schema.LIGAND_KEY)
        .reset_index(drop=True)
    )
    logger.info("clustering %d ligands at Tanimoto distance %.2f", len(unique), cutoff)
    labels = cluster_ligands(ecfp_bitvects(unique[schema.SMILES]), cutoff=cutoff)
    mapping = {
        key: f"lc{label}" for key, label in zip(unique[schema.LIGAND_KEY], labels, strict=True)
    }
    return df[schema.LIGAND_KEY].map(mapping).fillna("unknown").astype(str).rename("ligand_group")


def protein_groups(
    df: pd.DataFrame,
    axis: str,
    *,
    identity: float = 0.30,
    holdout_families: tuple[str, ...] = (),
) -> pd.Series | None:
    """Group label per row for the protein axis, or ``None`` for no constraint.

    ``identity``
        One group per accession. This is what most published "protein-cold"
        splits do, and the audit's central finding is that it is not enough: a
        held-out accession can have a 90%-identical paralogue in training.
    ``cluster``
        One group per sequence-identity cluster at ``identity``. This is the
        setting that closes the channel.
    ``family``
        Hold out whole target families named in ``holdout_families``. The hardest
        setting, used for the cross-family generalisation experiment.
    """
    if axis == WILDCARD:
        return None
    if axis == "identity":
        return df[schema.TARGET_KEY].astype(str).rename("protein_group")
    if axis == "cluster":
        return _protein_cluster_groups(df, identity=identity)
    if axis == "family":
        if not holdout_families:
            raise ValueError("protein axis 'family' requires holdout_families")
        return df[schema.TARGET_FAMILY].astype(str).rename("protein_group")
    raise ValueError(f"unknown protein axis {axis!r}; expected one of {PROTEIN_AXES}")


def _protein_cluster_groups(df: pd.DataFrame, *, identity: float) -> pd.Series:
    from leakfree_dta.prot.identity import cluster_proteins

    sequences: Mapping[str, str] = (
        df.drop_duplicates(subset=[schema.TARGET_KEY])
        .set_index(schema.TARGET_KEY)[schema.TARGET_SEQUENCE]
        .dropna()
        .to_dict()
    )
    result = cluster_proteins(sequences, min_seq_id=identity)
    mapping = {key: f"pc{label}" for key, label in result.labels.items()}
    groups = df[schema.TARGET_KEY].map(mapping)

    # A target with no sequence cannot be clustered. Giving it its own group is the
    # permissive choice and would understate leakage, so it is excluded instead and
    # the count is logged.
    n_unclustered = int(groups.isna().sum())
    if n_unclustered:
        logger.warning(
            "%d rows target proteins without a sequence; they will be excluded from the split",
            n_unclustered,
        )
    return groups.rename("protein_group")


def temporal_sides(
    df: pd.DataFrame,
    axis: str,
    *,
    cutoff_year: int | None,
    valid_years: int = 2,
) -> pd.Series | None:
    """Fold side per row for the temporal axis, or ``None`` for no constraint.

    Unlike the other two axes this returns sides directly rather than groups,
    because time has an order: the test set must be the *later* rows, not an
    arbitrary held-out block.

    The validation set is the ``valid_years`` immediately before the cutoff, so
    model selection never sees data from after the cutoff either. A validation set
    sampled at random from the training years would leak the future into
    hyperparameter choice -- a subtle version of the same problem, and one that is
    common in published temporal splits.
    """
    if axis == "none":
        return None
    if axis != "year":
        raise ValueError(f"unknown temporal axis {axis!r}; expected one of {TEMPORAL_AXES}")

    years = pd.to_numeric(df[schema.DOC_YEAR], errors="coerce")
    if years.notna().sum() == 0:
        raise ValueError("temporal axis 'year' requires publication years, none are present")

    if cutoff_year is None:
        cutoff_year = int(np.nanpercentile(years.dropna().to_numpy(dtype=float), 80))
        logger.info("no temporal cutoff given; using the 80th percentile year, %d", cutoff_year)

    valid_cutoff = cutoff_year - valid_years
    sides = pd.Series("excluded", index=df.index, dtype=object)
    sides[years <= valid_cutoff] = "train"
    sides[(years > valid_cutoff) & (years <= cutoff_year)] = "valid"
    sides[years > cutoff_year] = "test"
    # Rows with no year cannot be placed on a timeline; excluding them is the only
    # choice that keeps the temporal claim honest.
    sides[years.isna()] = "excluded"

    counts = sides.value_counts().to_dict()
    logger.info(
        "temporal axis: train <= %d, valid (%d, %d], test > %d -> %s",
        valid_cutoff,
        valid_cutoff,
        cutoff_year,
        cutoff_year,
        counts,
    )
    return sides.rename("temporal_side")
