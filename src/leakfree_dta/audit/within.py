"""Checks that look at one dataset on its own, before any split is applied.

These are the findings that no choice of split can fix. If the same
(ligand, target) pair carries two measurements a hundred-fold apart, then some
test label is wrong whatever the split does, and the achievable error has a floor
that no model can beat. Quantifying that floor is the honest way to read every
leaderboard the field has published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leakfree_dta.audit.report import CheckResult, Severity
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Two measurements of the same pair differing by more than this many log units
#: are treated as inconsistent. 1.0 is a 10-fold difference in Kd -- far beyond
#: inter-laboratory reproducibility, which is usually quoted at 0.3-0.5 log units.
CONFLICT_THRESHOLD = 1.0


def check_schema(df: pd.DataFrame) -> CheckResult:
    """Report schema violations as an audit finding rather than an exception."""
    violations = schema.validate(df)
    n_affected = sum(violation.n_rows for violation in violations)
    return CheckResult(
        name="schema_violations",
        axis="schema",
        severity=Severity.HIGH if violations else Severity.INFO,
        n_affected=min(n_affected, len(df)),
        n_total=len(df),
        summary=(
            "The curated table violates the canonical schema: "
            + "; ".join(str(violation) for violation in violations)
            if violations
            else "The curated table satisfies the canonical schema."
        ),
        statistics={"violations": [str(violation) for violation in violations]},
        tolerance=0.0,
    )


def check_duplicate_measurements(df: pd.DataFrame) -> CheckResult:
    """Pairs measured more than once anywhere in the dataset.

    Duplication is not itself an error -- the same compound is legitimately
    retested -- but it has two consequences the paper quantifies. It inflates
    apparent dataset size, and it is the mechanism by which a random split places
    the same measurement on both sides of the boundary.
    """
    counts = df[schema.PAIR_KEY].value_counts()
    duplicated = counts[counts > 1]
    n_affected = int(duplicated.sum())

    details = (
        df[df[schema.PAIR_KEY].isin(duplicated.index)]
        .sort_values([schema.PAIR_KEY, schema.SOURCE])
        .loc[:, [schema.ROW_KEY, schema.PAIR_KEY, schema.Y, schema.Y_TYPE, schema.SOURCE, schema.DOC_YEAR]]
    )

    return CheckResult(
        name="duplicate_measurements",
        axis="redundancy",
        severity=Severity.INFO,
        n_affected=n_affected,
        n_total=len(df),
        summary=(
            f"{len(duplicated):,} of {len(counts):,} unique pairs carry more than one "
            f"measurement, accounting for {n_affected:,} rows. The dataset's "
            f"{len(df):,} rows represent {len(counts):,} distinct pairs."
        ),
        statistics={
            "n_unique_pairs": len(counts),
            "n_duplicated_pairs": len(duplicated),
            "max_replicates": int(counts.max()) if len(counts) else 0,
            "mean_replicates": round(float(counts.mean()), 3) if len(counts) else 0.0,
        },
        details=details,
        tolerance=1.0,  # reported for context; never a failure on its own
    )


def check_conflicting_labels(
    df: pd.DataFrame, *, threshold: float = CONFLICT_THRESHOLD
) -> CheckResult:
    """Pairs whose replicate measurements disagree by more than ``threshold``.

    This is the irreducible noise floor. If 4% of pairs disagree by over one log
    unit, then a model reporting RMSE of 0.5 on a test set drawn from this data is
    reporting a number smaller than the disagreement between the labels it is
    scored against -- which means it is fitting assay idiosyncrasies, not affinity.
    """
    grouped = df.groupby(schema.PAIR_KEY, observed=True)[schema.Y]
    spread = grouped.agg(["min", "max", "count", "std", "mean"])
    spread["range"] = spread["max"] - spread["min"]
    replicated = spread[spread["count"] > 1]
    conflicting = replicated[replicated["range"] > threshold]

    n_affected = int(df[schema.PAIR_KEY].isin(conflicting.index).sum())
    affected_keys = df.loc[df[schema.PAIR_KEY].isin(conflicting.index), schema.ROW_KEY].tolist()

    # The noise floor: pooled within-pair standard deviation over replicated pairs.
    pooled_std = (
        float(np.sqrt(np.nanmean(np.square(replicated["std"].to_numpy(dtype=float)))))
        if len(replicated)
        else 0.0
    )

    details = conflicting.reset_index().sort_values("range", ascending=False)

    fraction_of_replicated = len(conflicting) / len(replicated) if len(replicated) else 0.0
    return CheckResult(
        name="conflicting_labels",
        axis="label",
        severity=Severity.HIGH if fraction_of_replicated > 0.05 else Severity.MEDIUM,
        n_affected=n_affected,
        n_total=len(df),
        summary=(
            f"{len(conflicting):,} of {len(replicated):,} replicated pairs "
            f"({fraction_of_replicated:.1%}) disagree by more than {threshold} log units. "
            f"The pooled within-pair standard deviation is {pooled_std:.3f} log units, "
            f"which is a floor on the RMSE any model can achieve on this data."
        ),
        statistics={
            "n_replicated_pairs": len(replicated),
            "n_conflicting_pairs": len(conflicting),
            "fraction_of_replicated_pairs": round(fraction_of_replicated, 4),
            "pooled_within_pair_std": round(pooled_std, 4),
            "noise_floor_rmse": round(pooled_std, 4),
            "max_range": round(float(conflicting["range"].max()), 3) if len(conflicting) else 0.0,
            "affected_row_keys": affected_keys,
        },
        details=details,
        tolerance=0.02,
    )


def check_affinity_type_mixing(df: pd.DataFrame) -> CheckResult:
    """Pairs whose replicates come from different affinity types.

    Treating pIC50 and pKi as the same quantity is standard practice and standard
    criticism. Where a pair has both, the difference is measurable, and this check
    reports it so the paper can state the cost of pooling rather than assume it.
    """
    by_pair = df.groupby(schema.PAIR_KEY, observed=True)[schema.Y_TYPE].nunique()
    mixed = by_pair[by_pair > 1]
    n_affected = int(df[schema.PAIR_KEY].isin(mixed.index).sum())

    offsets: dict[str, float] = {}
    if len(mixed):
        subset = df[df[schema.PAIR_KEY].isin(mixed.index)]
        means = subset.groupby([schema.PAIR_KEY, schema.Y_TYPE], observed=True)[schema.Y].mean().unstack()
        for left in means.columns:
            for right in means.columns:
                if left < right:
                    delta = (means[left] - means[right]).dropna()
                    if len(delta) >= 10:
                        offsets[f"{left}_minus_{right}"] = round(float(delta.mean()), 4)

    return CheckResult(
        name="affinity_type_mixing",
        axis="label",
        severity=Severity.MEDIUM if len(mixed) else Severity.INFO,
        n_affected=n_affected,
        n_total=len(df),
        summary=(
            f"{len(mixed):,} pairs carry measurements of more than one affinity type. "
            f"Mean within-pair offsets between types: {offsets or 'not estimable'}. "
            "Pooling types onto one p-scale is an approximation of this size."
        ),
        statistics={"n_mixed_pairs": len(mixed), "mean_offsets": offsets},
        tolerance=0.10,
    )


def check_ligand_redundancy(
    df: pd.DataFrame, *, cutoff: float = 0.4, max_ligands: int = 20_000
) -> CheckResult:
    """Chemical redundancy, as an effective sample size.

    A dataset of 50,000 measurements over 300 chemotypes does not carry 50,000
    independent observations, and confidence intervals computed as though it does
    are too narrow. The ratio reported here is the correction the paper applies
    before quoting any interval.
    """
    from leakfree_dta.chem.fingerprints import ecfp_bitvects
    from leakfree_dta.chem.similarity import cluster_ligands

    ligands = (
        df.drop_duplicates(subset=[schema.LIGAND_KEY])
        .sort_values(schema.LIGAND_KEY)
        .loc[:, [schema.LIGAND_KEY, schema.SMILES]]
    )
    truncated = len(ligands) > max_ligands
    if truncated:
        logger.info(
            "ligand redundancy: sampling %d of %d ligands to keep clustering tractable",
            max_ligands,
            len(ligands),
        )
        ligands = ligands.sample(max_ligands, random_state=0).sort_values(schema.LIGAND_KEY)

    vectors = ecfp_bitvects(ligands[schema.SMILES])
    labels = cluster_ligands(vectors, cutoff=cutoff)
    n_clusters = len(set(labels.tolist()))
    n_ligands = len(ligands)
    ratio = n_clusters / n_ligands if n_ligands else 1.0

    cluster_sizes = pd.Series(labels).value_counts()
    return CheckResult(
        name="ligand_redundancy",
        axis="redundancy",
        severity=Severity.INFO,
        n_affected=n_ligands - n_clusters,
        n_total=n_ligands,
        summary=(
            f"{n_ligands:,} distinct ligands collapse into {n_clusters:,} chemical clusters at "
            f"Tanimoto distance {cutoff} (effective-sample-size ratio {ratio:.3f}). "
            f"The largest cluster holds {int(cluster_sizes.iloc[0]) if len(cluster_sizes) else 0:,} "
            "ligands. Confidence intervals should be computed over clusters, not rows."
        ),
        statistics={
            "n_ligands": n_ligands,
            "n_clusters": n_clusters,
            "effective_sample_ratio": round(ratio, 4),
            "largest_cluster": int(cluster_sizes.iloc[0]) if len(cluster_sizes) else 0,
            "n_singletons": int((cluster_sizes == 1).sum()),
            "truncated": truncated,
        },
        tolerance=1.0,
    )


def check_target_imbalance(df: pd.DataFrame) -> CheckResult:
    """How concentrated the data is on a few targets.

    When one target carries a third of the measurements, an overall metric is
    mostly a metric for that target. This is the finding that motivates the
    paper's per-group reporting and its group-DRO baseline.
    """
    counts = df[schema.TARGET_KEY].value_counts()
    if not len(counts):
        return CheckResult(
            name="target_imbalance",
            axis="redundancy",
            severity=Severity.INFO,
            n_affected=0,
            n_total=0,
            summary="No targets present.",
            tolerance=1.0,
        )

    shares = (counts / counts.sum()).to_numpy()
    gini = _gini(shares)
    top_decile = int(max(1, round(0.1 * len(counts))))
    top_share = float(shares[:top_decile].sum())

    return CheckResult(
        name="target_imbalance",
        axis="redundancy",
        severity=Severity.MEDIUM if top_share > 0.5 else Severity.INFO,
        n_affected=int(counts.iloc[:top_decile].sum()),
        n_total=len(df),
        summary=(
            f"The top {top_decile:,} of {len(counts):,} targets hold {top_share:.1%} of all "
            f"measurements (Gini {gini:.3f}). Overall metrics are dominated by them, which is "
            "why every table in this paper reports per-family and worst-group numbers."
        ),
        statistics={
            "n_targets": len(counts),
            "gini": round(gini, 4),
            "top_decile_share": round(top_share, 4),
            "median_measurements_per_target": int(counts.median()),
            "max_measurements_per_target": int(counts.max()),
            "n_targets_under_10": int((counts < 10).sum()),
        },
        tolerance=1.0,
    )


def _gini(shares: np.ndarray) -> float:
    """Gini coefficient of a share vector that already sums to one."""
    sorted_shares = np.sort(shares)
    n = len(sorted_shares)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_shares) - (n + 1)) / n)


#: Every within-dataset check, in report order.
WITHIN_CHECKS = (
    check_schema,
    check_duplicate_measurements,
    check_conflicting_labels,
    check_affinity_type_mixing,
    check_target_imbalance,
    check_ligand_redundancy,
)
