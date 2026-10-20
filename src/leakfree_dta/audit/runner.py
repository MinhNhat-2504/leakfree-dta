"""Running the audit.

Two entry points, matching the two things a reader wants to know:

:func:`audit_dataset`
    What is wrong with the data itself, before any split. Run once per dataset.
:func:`audit_split`
    What crosses the boundary for a particular split. Run once per split, and once
    per target family within it, because the paper's finding is that the dominant
    channel differs by family.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from leakfree_dta.audit.across import ACROSS_CHECKS
from leakfree_dta.audit.report import LeakageReport, Severity
from leakfree_dta.audit.shortcut import SHORTCUT_CHECKS, check_shortcut_predictability
from leakfree_dta.audit.within import WITHIN_CHECKS
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.prot.identity import identity_backend

logger = get_logger(__name__)


def audit_dataset(
    df: pd.DataFrame,
    *,
    dataset: str,
    checks: Sequence = WITHIN_CHECKS,
    skip: Sequence[str] = (),
) -> LeakageReport:
    """Run every within-dataset check.

    ``skip`` names checks to omit -- useful for ``ligand_redundancy``, which is the
    slow one, during iteration.
    """
    report = LeakageReport(dataset=dataset, split="(no split)")
    report.metadata.update(
        {
            "n_rows": len(df),
            "n_ligands": int(df[schema.LIGAND_KEY].nunique()),
            "n_targets": int(df[schema.TARGET_KEY].nunique()),
            "identity_backend": identity_backend(),
        }
    )

    for check in checks:
        name = check.__name__.removeprefix("check_")
        if name in skip:
            logger.info("skipping check %s", name)
            continue
        logger.info("running %s", name)
        report.add(check(df))

    return report


def audit_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    reference_pearson: float | None = None,
    checks: Sequence = ACROSS_CHECKS,
    skip: Sequence[str] = (),
) -> LeakageReport:
    """Run every cross-split check, plus the shortcut probes.

    ``reference_pearson`` is a published model's correlation on this split. Supply
    it and the shortcut probe converts into the paper's headline statistic: the
    share of reported performance that needed no interaction modelling.
    """
    report = LeakageReport(dataset=dataset, split=split)
    report.metadata.update(
        {
            "n_train": len(train),
            "n_test": len(test),
            "n_train_ligands": int(train[schema.LIGAND_KEY].nunique()),
            "n_test_ligands": int(test[schema.LIGAND_KEY].nunique()),
            "n_train_targets": int(train[schema.TARGET_KEY].nunique()),
            "n_test_targets": int(test[schema.TARGET_KEY].nunique()),
            "identity_backend": identity_backend(),
        }
    )

    if not len(train) or not len(test):
        logger.warning("split %r has an empty fold; no cross-split check can run", split)
        report.metadata["skipped"] = "empty fold"
        return report

    for check in checks:
        name = check.__name__.removeprefix("check_")
        if name in skip:
            logger.info("skipping check %s", name)
            continue
        logger.info("running %s", name)
        report.add(check(train, test))

    for check in SHORTCUT_CHECKS:
        name = check.__name__.removeprefix("check_")
        if name in skip:
            continue
        logger.info("running %s", name)
        if check is check_shortcut_predictability:
            report.add(check(train, test, reference_pearson=reference_pearson))
        else:
            report.add(check(train, test))

    return report


def audit_split_by_family(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    families: Sequence[str] = schema.TARGET_FAMILIES,
    skip: Sequence[str] = (),
) -> dict[str, LeakageReport]:
    """Audit a split once overall and once per target family.

    This is the function that produces the paper's central table. Running the same
    checks within each family is what shows that the dominant leakage channel is
    family-specific -- protein identity in kinases, ligand similarity in GPCRs,
    label inconsistency in proteases -- and therefore that no single split rule is
    sufficient.
    """
    reports = {
        "overall": audit_split(train, test, dataset=dataset, split=split, skip=skip),
    }
    for family in families:
        family_train = train[train[schema.TARGET_FAMILY] == family]
        family_test = test[test[schema.TARGET_FAMILY] == family]
        if not len(family_train) or not len(family_test):
            logger.info("family %s has no data on one side of split %r; skipping", family, split)
            continue
        reports[family] = audit_split(
            family_train,
            family_test,
            dataset=f"{dataset}:{family}",
            split=split,
            skip=skip,
        )
    return reports


def comparison_table(reports: dict[str, LeakageReport]) -> pd.DataFrame:
    """Flatten several reports into one wide table, checks as rows.

    The shape the paper prints: one row per check, one column per family, so the
    family-specific pattern is visible at a glance.
    """
    records: dict[tuple[str, str], dict[str, object]] = {}
    for label, report in reports.items():
        for result in report.results:
            key = (result.axis, result.name)
            row = records.setdefault(key, {"axis": result.axis, "check": result.name})
            row[label] = round(result.fraction, 5)
            row[f"{label}_severity"] = result.severity.value

    frame = pd.DataFrame(list(records.values()))
    if frame.empty:
        return frame
    return frame.sort_values(["axis", "check"]).reset_index(drop=True)


def dominant_channel(report: LeakageReport) -> tuple[str, float]:
    """The axis with the largest affected fraction, excluding informational checks.

    Reported per family. Naming the dominant channel in one word per family is how
    the paper's abstract states its finding.
    """
    by_axis: dict[str, float] = {}
    for result in report.results:
        if result.severity is Severity.INFO:
            continue
        by_axis[result.axis] = max(by_axis.get(result.axis, 0.0), result.fraction)
    if not by_axis:
        return ("none", 0.0)
    axis = max(by_axis, key=lambda key: by_axis[key])
    return (axis, by_axis[axis])

