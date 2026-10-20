"""Checks that compare a training split against a test split.

Each function closes over one channel. They are deliberately not combined into a
single pass, because the paper's claim is about *which* channel dominates in
*which* target family, and a combined number would erase that.

Ordering matters for reading the report: the checks run from the most to the least
blatant form of leakage, so the first failure a reader sees is the one that most
plainly invalidates a reported metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from leakfree_dta.audit.report import CheckResult, Severity
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Tanimoto at or above this counts as a near-duplicate ligand. 0.9 on ECFP4 is
#: conventionally "the same series with a trivial substitution".
NEAR_DUPLICATE_TANIMOTO = 0.90

#: Sequence identity at or above this counts as the same protein for leakage
#: purposes. 0.40 is below the 50% paralogue threshold but above the 30% fold
#: threshold, so it catches close relatives without merging a whole superfamily.
PROTEIN_IDENTITY_THRESHOLD = 0.40


def check_pair_overlap(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """The same (ligand, target) pair present on both sides.

    The most direct form of leakage there is: the model has seen a measurement of
    the exact thing it is being tested on. Any nonzero count here makes the split
    unusable, which is why the tolerance is exactly zero.
    """
    shared = set(train[schema.PAIR_KEY]) & set(test[schema.PAIR_KEY])
    mask = test[schema.PAIR_KEY].isin(shared)
    n_affected = int(mask.sum())

    details = pd.DataFrame()
    if shared:
        train_means = train.groupby(schema.PAIR_KEY, observed=True)[schema.Y].mean()
        affected = test.loc[mask, [schema.ROW_KEY, schema.PAIR_KEY, schema.Y]].copy()
        affected["train_y"] = affected[schema.PAIR_KEY].map(train_means)
        affected["abs_delta"] = (affected[schema.Y] - affected["train_y"]).abs()
        details = affected.sort_values("abs_delta")

    # How well a pure lookup of the training label predicts the test label. If
    # this is near-perfect, the test set is a copy of the training set.
    lookup_rmse = (
        float(np.sqrt(np.mean(np.square(details["abs_delta"].to_numpy(dtype=float)))))
        if len(details)
        else float("nan")
    )

    return CheckResult(
        name="pair_overlap",
        axis="pair",
        severity=Severity.CRITICAL,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{n_affected:,} test rows ({n_affected / max(len(test), 1):.2%}) measure a "
            f"(ligand, target) pair that also appears in training, across {len(shared):,} "
            f"shared pairs. Simply copying the training label for these rows gives "
            f"RMSE {lookup_rmse:.3f}; a model that memorises them inherits that score for free."
        ),
        statistics={
            "n_shared_pairs": len(shared),
            "train_label_lookup_rmse": None if np.isnan(lookup_rmse) else round(lookup_rmse, 4),
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        details=details,
        tolerance=0.0,
    )


def check_ligand_overlap(train: pd.DataFrame, test: pd.DataFrame, *, stereo_insensitive: bool = False) -> CheckResult:
    """The same ligand on both sides, against a different target.

    Not automatically leakage -- generalising to a new target for a known compound
    is a legitimate task -- but it is the task most published "random split"
    numbers are actually measuring, while being described as something harder. The
    check exists so a paper can state which question its split asks.
    """
    column = schema.LIGAND_SKELETON if stereo_insensitive else schema.LIGAND_KEY
    name = "ligand_skeleton_overlap" if stereo_insensitive else "ligand_identity_overlap"

    shared = set(train[column]) & set(test[column])
    mask = test[column].isin(shared)
    n_affected = int(mask.sum())

    level = "stereochemistry-insensitive skeleton" if stereo_insensitive else "exact InChIKey"
    return CheckResult(
        name=name,
        axis="ligand",
        severity=Severity.HIGH,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{n_affected:,} test rows ({n_affected / max(len(test), 1):.2%}) involve a ligand "
            f"already seen in training at the {level} level ({len(shared):,} shared ligands). "
            "A split with this property measures generalisation to new targets, not to new chemistry."
        ),
        statistics={
            "n_shared_ligands": len(shared),
            "n_test_ligands": int(test[column].nunique()),
            "level": level,
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        tolerance=0.0,
    )


def check_near_duplicate_ligands(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    threshold: float = NEAR_DUPLICATE_TANIMOTO,
    max_reference: int = 50_000,
) -> CheckResult:
    """Test ligands whose nearest training neighbour is a near-duplicate.

    This is the channel that a conventional scaffold split leaves wide open.
    Bemis-Murcko treats a ring-system change as a new scaffold, so two compounds
    differing by one ring nitrogen land on opposite sides while being, for a
    model, the same molecule.
    """
    from leakfree_dta.chem.fingerprints import ecfp_bitvects
    from leakfree_dta.chem.similarity import nearest_neighbor

    train_ligands = train.drop_duplicates(subset=[schema.LIGAND_KEY]).sort_values(schema.LIGAND_KEY)
    if len(train_ligands) > max_reference:
        logger.info(
            "near-duplicate check: sampling %d of %d training ligands as reference",
            max_reference,
            len(train_ligands),
        )
        train_ligands = train_ligands.sample(max_reference, random_state=0)
    test_ligands = test.drop_duplicates(subset=[schema.LIGAND_KEY]).sort_values(schema.LIGAND_KEY)

    if not len(train_ligands) or not len(test_ligands):
        return _empty_result("near_duplicate_ligands", "ligand", len(test))

    reference = ecfp_bitvects(train_ligands[schema.SMILES])
    query = ecfp_bitvects(test_ligands[schema.SMILES])
    neighbors = nearest_neighbor(query, reference)

    per_ligand = pd.DataFrame(
        {
            schema.LIGAND_KEY: test_ligands[schema.LIGAND_KEY].to_numpy(),
            "nn_similarity": neighbors.similarity,
            "nn_train_ligand": [
                train_ligands[schema.LIGAND_KEY].iloc[index] if index >= 0 else None
                for index in neighbors.index
            ],
        }
    )
    flagged = set(per_ligand.loc[neighbors.above(threshold), schema.LIGAND_KEY])
    mask = test[schema.LIGAND_KEY].isin(flagged)
    n_affected = int(mask.sum())

    similarity = neighbors.similarity[np.isfinite(neighbors.similarity)]
    return CheckResult(
        name="near_duplicate_ligands",
        axis="ligand",
        severity=Severity.HIGH,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{len(flagged):,} of {len(test_ligands):,} test ligands have a training neighbour at "
            f"Tanimoto >= {threshold} ({n_affected:,} test rows, "
            f"{n_affected / max(len(test), 1):.2%}). Median nearest-neighbour similarity is "
            f"{np.median(similarity):.3f} -- the honest difficulty of this test set."
        ),
        statistics={
            "threshold": threshold,
            "n_flagged_ligands": len(flagged),
            "median_nn_similarity": round(float(np.median(similarity)), 4) if similarity.size else None,
            "p90_nn_similarity": round(float(np.percentile(similarity, 90)), 4) if similarity.size else None,
            "fraction_ligands_above_0.7": round(float((similarity >= 0.7).mean()), 4) if similarity.size else None,
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        details=per_ligand.sort_values("nn_similarity", ascending=False),
        tolerance=0.01,
    )


def check_near_duplicate_pair_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    threshold: float = NEAR_DUPLICATE_TANIMOTO,
) -> CheckResult:
    """Test rows whose ligand has a near-duplicate in training *against the same target*.

    The sharpest of the overlap checks, and the one that makes the paper's headline
    number well defined.

    ``near_duplicate_ligands`` asks only whether a similar compound was seen
    somewhere. That is routinely 100% of a test set and says little. This check
    asks the question that actually determines whether a label is recoverable: was
    a near-identical compound measured *against this same protein*? If so, the test
    label can be read off the training set by nearest-neighbour lookup, and no
    modelling is required. The lookup's RMSE is reported so the leakage is
    quantified in the units of the metric it inflates.
    """
    from leakfree_dta.chem.fingerprints import ecfp_bitvects
    from leakfree_dta.chem.similarity import nearest_neighbor

    shared_targets = set(train[schema.TARGET_KEY]) & set(test[schema.TARGET_KEY])
    if not shared_targets:
        return _empty_result(
            "near_duplicate_pair_overlap",
            "pair",
            len(test),
            reason="train and test share no target, so no pair can be recovered",
        )

    # Fingerprint every distinct ligand once, up front. The comparison below is
    # per target, and a ChEMBL-scale benchmark has each ligand measured against
    # dozens of them, so fingerprinting inside the loop would repeat the most
    # expensive step dozens of times over.
    everything = pd.concat([train, test], ignore_index=True)
    unique_ligands = everything.drop_duplicates(subset=[schema.LIGAND_KEY])
    fingerprint = dict(
        zip(
            unique_ligands[schema.LIGAND_KEY],
            ecfp_bitvects(unique_ligands[schema.SMILES]),
            strict=True,
        )
    )
    logger.info(
        "near-duplicate pair check: %d ligands fingerprinted, %d shared targets",
        len(fingerprint),
        len(shared_targets),
    )

    # Grouping once beats filtering the frame per target, which is a full scan each.
    train_by_target = dict(tuple(train.groupby(schema.TARGET_KEY, observed=True)))
    test_by_target = dict(tuple(test.groupby(schema.TARGET_KEY, observed=True)))

    affected_rows: list[str] = []
    lookup_errors: list[float] = []
    records: list[dict[str, object]] = []

    # Per target, so similarity is only ever compared within the same protein.
    for target in shared_targets:
        train_rows = train_by_target.get(target)
        test_rows = test_by_target.get(target)
        if train_rows is None or test_rows is None or not len(train_rows) or not len(test_rows):
            continue

        reference = [fingerprint.get(key) for key in train_rows[schema.LIGAND_KEY]]
        query = [fingerprint.get(key) for key in test_rows[schema.LIGAND_KEY]]
        neighbors = nearest_neighbor(query, reference)
        flagged = neighbors.above(threshold)

        train_y = pd.to_numeric(train_rows[schema.Y], errors="coerce").to_numpy(dtype=float)
        test_y = pd.to_numeric(test_rows[schema.Y], errors="coerce").to_numpy(dtype=float)
        row_keys = test_rows[schema.ROW_KEY].to_numpy()

        for position in np.flatnonzero(flagged):
            neighbor_index = neighbors.index[position]
            if neighbor_index < 0:
                continue
            affected_rows.append(str(row_keys[position]))
            error = abs(test_y[position] - train_y[neighbor_index])
            lookup_errors.append(float(error))
            records.append(
                {
                    schema.ROW_KEY: row_keys[position],
                    schema.TARGET_KEY: target,
                    "nn_similarity": round(float(neighbors.similarity[position]), 4),
                    "test_y": round(float(test_y[position]), 4),
                    "nn_train_y": round(float(train_y[neighbor_index]), 4),
                    "abs_error": round(error, 4),
                }
            )

    n_affected = len(affected_rows)
    lookup_rmse = (
        float(np.sqrt(np.mean(np.square(lookup_errors)))) if lookup_errors else float("nan")
    )

    return CheckResult(
        name="near_duplicate_pair_overlap",
        axis="pair",
        severity=Severity.CRITICAL,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{n_affected:,} test rows ({n_affected / max(len(test), 1):.2%}) have a training "
            f"measurement of a ligand at Tanimoto >= {threshold} against the same target. "
            f"Copying that neighbour's label predicts these rows with RMSE {lookup_rmse:.3f}, "
            "so they require no model at all."
        ),
        statistics={
            "threshold": threshold,
            "nearest_neighbor_lookup_rmse": None if np.isnan(lookup_rmse) else round(lookup_rmse, 4),
            "n_shared_targets": len(shared_targets),
            "affected_row_keys": affected_rows,
        },
        details=pd.DataFrame.from_records(records).sort_values("nn_similarity", ascending=False)
        if records
        else pd.DataFrame(),
        tolerance=0.0,
    )


def check_scaffold_overlap(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Test rows whose Bemis-Murcko scaffold appears in training."""
    shared = set(train[schema.SCAFFOLD].dropna()) & set(test[schema.SCAFFOLD].dropna())
    shared.discard("acyclic")
    mask = test[schema.SCAFFOLD].isin(shared)
    n_affected = int(mask.sum())

    return CheckResult(
        name="scaffold_overlap",
        axis="ligand",
        severity=Severity.MEDIUM,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{n_affected:,} test rows ({n_affected / max(len(test), 1):.2%}) carry a "
            f"Bemis-Murcko scaffold also present in training ({len(shared):,} shared scaffolds)."
        ),
        statistics={
            "n_shared_scaffolds": len(shared),
            "n_test_scaffolds": int(test[schema.SCAFFOLD].nunique(dropna=True)),
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        tolerance=0.0,
    )


def check_target_overlap(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Test rows against a target that also appears in training.

    Reported separately from sequence identity because the distinction is the
    whole point: a split can hold out every test *accession* and still train on a
    protein that is 95% identical to it.
    """
    shared = set(train[schema.TARGET_KEY]) & set(test[schema.TARGET_KEY])
    mask = test[schema.TARGET_KEY].isin(shared)
    n_affected = int(mask.sum())

    return CheckResult(
        name="target_identity_overlap",
        axis="protein",
        severity=Severity.HIGH,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{n_affected:,} test rows ({n_affected / max(len(test), 1):.2%}) target a protein "
            f"seen in training ({len(shared):,} shared targets of "
            f"{test[schema.TARGET_KEY].nunique():,} test targets). This is the transductive "
            "setting; it does not measure generalisation to new proteins."
        ),
        statistics={
            "n_shared_targets": len(shared),
            "n_test_targets": int(test[schema.TARGET_KEY].nunique()),
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        tolerance=0.0,
    )


def check_protein_identity_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    threshold: float = PROTEIN_IDENTITY_THRESHOLD,
) -> CheckResult:
    """Test targets with a training target above ``threshold`` sequence identity.

    This is the channel the 2026 Chemical Science audit found open in the most
    widely used benchmarks. Inside the kinase family it is severe: holding out an
    accession means almost nothing when a 90%-identical paralogue remains in
    training.
    """
    from leakfree_dta.prot.identity import identity_backend, max_identity_to_set

    train_targets = (
        train.drop_duplicates(subset=[schema.TARGET_KEY])
        .set_index(schema.TARGET_KEY)[schema.TARGET_SEQUENCE]
        .dropna()
        .to_dict()
    )
    test_targets = (
        test.drop_duplicates(subset=[schema.TARGET_KEY])
        .set_index(schema.TARGET_KEY)[schema.TARGET_SEQUENCE]
        .dropna()
        .to_dict()
    )
    if not train_targets or not test_targets:
        return _empty_result("protein_identity_overlap", "protein", len(test))

    # Delegated rather than looped here. An all-against-all comparison of a few
    # thousand proteins is hours of pairwise Smith-Waterman and seconds of
    # MMseqs2; max_identity_to_set picks the backend and reports which it used.
    best = max_identity_to_set(test_targets, train_targets)

    records: list[dict[str, object]] = []
    for test_key in test_targets:
        train_key, identity = best.get(test_key, (None, 0.0))
        records.append(
            {
                schema.TARGET_KEY: test_key,
                "nn_identity": round(float(identity), 4),
                "nn_train_target": train_key,
            }
        )

    per_target = pd.DataFrame.from_records(records)
    flagged = set(per_target.loc[per_target["nn_identity"] >= threshold, schema.TARGET_KEY])
    mask = test[schema.TARGET_KEY].isin(flagged)
    n_affected = int(mask.sum())
    identities = per_target["nn_identity"].to_numpy(dtype=float)

    return CheckResult(
        name="protein_identity_overlap",
        axis="protein",
        severity=Severity.CRITICAL,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"{len(flagged):,} of {len(test_targets):,} test targets have a training target at "
            f">= {threshold:.0%} sequence identity ({n_affected:,} test rows, "
            f"{n_affected / max(len(test), 1):.2%}). Median nearest-neighbour identity is "
            f"{np.median(identities):.1%}. Backend: {identity_backend()}."
        ),
        statistics={
            "threshold": threshold,
            "backend": identity_backend(),
            "n_flagged_targets": len(flagged),
            "median_nn_identity": round(float(np.median(identities)), 4),
            "max_nn_identity": round(float(identities.max()), 4),
            "affected_row_keys": test.loc[mask, schema.ROW_KEY].tolist(),
        },
        details=per_target.sort_values("nn_identity", ascending=False),
        tolerance=0.0,
    )


def check_temporal_violation(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Test rows published no later than the newest training row.

    A prospective claim -- "this model would have found these compounds" -- is
    only supported by a split where every test measurement postdates every
    training measurement. Most published temporal splits only shift the median.
    """
    if schema.DOC_YEAR not in train.columns or train[schema.DOC_YEAR].isna().all():
        return _empty_result("temporal_violation", "temporal", len(test), reason="no publication years")

    train_years = pd.to_numeric(train[schema.DOC_YEAR], errors="coerce").dropna()
    test_years = pd.to_numeric(test[schema.DOC_YEAR], errors="coerce")
    if not len(train_years):
        return _empty_result("temporal_violation", "temporal", len(test), reason="no publication years")

    cutoff = int(train_years.max())
    mask = test_years.notna() & (test_years <= cutoff)
    n_affected = int(mask.sum())

    return CheckResult(
        name="temporal_violation",
        axis="temporal",
        severity=Severity.MEDIUM,
        n_affected=n_affected,
        n_total=len(test),
        summary=(
            f"The newest training measurement is from {cutoff}; {n_affected:,} test rows "
            f"({n_affected / max(len(test), 1):.2%}) are from {cutoff} or earlier. "
            "A split with overlap here cannot support a prospective claim."
        ),
        statistics={
            "train_year_max": cutoff,
            "train_year_min": int(train_years.min()),
            "test_year_min": int(test_years.min()) if test_years.notna().any() else None,
            "test_year_max": int(test_years.max()) if test_years.notna().any() else None,
            "n_test_rows_without_year": int(test_years.isna().sum()),
            "affected_row_keys": test.loc[mask.fillna(False), schema.ROW_KEY].tolist(),
        },
        tolerance=0.0,
    )


def check_label_shift(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Whether train and test affinities come from the same distribution.

    Informational, not a leakage channel. It is reported because a split that
    closes every leakage channel necessarily opens a distribution shift, and a
    reader needs to see the size of that shift to judge whether a drop in
    performance reflects honest difficulty or a broken split.
    """
    train_y = pd.to_numeric(train[schema.Y], errors="coerce").dropna().to_numpy(dtype=float)
    test_y = pd.to_numeric(test[schema.Y], errors="coerce").dropna().to_numpy(dtype=float)
    if len(train_y) < 2 or len(test_y) < 2:
        return _empty_result("label_shift", "label", len(test), reason="too few labelled rows")

    ks = stats.ks_2samp(train_y, test_y)
    pooled_std = float(np.sqrt((np.var(train_y, ddof=1) + np.var(test_y, ddof=1)) / 2))
    cohens_d = float((test_y.mean() - train_y.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return CheckResult(
        name="label_shift",
        axis="label",
        severity=Severity.INFO,
        n_affected=0,
        n_total=len(test),
        summary=(
            f"Test affinities differ from training by KS statistic {ks.statistic:.3f} "
            f"(p={ks.pvalue:.2e}); mean shift {test_y.mean() - train_y.mean():+.3f} log units "
            f"(Cohen's d = {cohens_d:+.3f})."
        ),
        statistics={
            "ks_statistic": round(float(ks.statistic), 4),
            "ks_pvalue": float(ks.pvalue),
            "train_mean": round(float(train_y.mean()), 4),
            "test_mean": round(float(test_y.mean()), 4),
            "train_std": round(float(train_y.std(ddof=1)), 4),
            "test_std": round(float(test_y.std(ddof=1)), 4),
            "cohens_d": round(cohens_d, 4),
        },
        tolerance=1.0,
    )


def _empty_result(name: str, axis: str, n_total: int, *, reason: str = "not applicable") -> CheckResult:
    return CheckResult(
        name=name,
        axis=axis,
        severity=Severity.INFO,
        n_affected=0,
        n_total=n_total,
        summary=f"Check skipped: {reason}.",
        statistics={"skipped": True, "reason": reason},
        tolerance=1.0,
    )


#: Every cross-split check, ordered from most to least blatant.
ACROSS_CHECKS = (
    check_pair_overlap,
    check_near_duplicate_pair_overlap,
    check_protein_identity_overlap,
    check_target_overlap,
    check_ligand_overlap,
    check_near_duplicate_ligands,
    check_scaffold_overlap,
    check_temporal_violation,
    check_label_shift,
)
