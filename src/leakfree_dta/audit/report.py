"""The audit's output types.

A report is a structured object, not a printout. It serialises to JSON for the
machine-readable artifact the paper releases, and renders to Markdown for the
human-readable one. Both are generated from the same objects so they cannot drift
apart.
"""

from __future__ import annotations

import datetime as dt
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from leakfree_dta._version import __version__
from leakfree_dta.utils.io import write_json, write_table, write_text


class Severity(str, Enum):  # noqa: UP042 - str mixin keeps report JSON comparable on 3.10
    """How much a finding threatens the validity of a reported result.

    The levels are defined by consequence, not by magnitude, so that a reader can
    act on them:

    ``CRITICAL``
        Test rows are recoverable from training rows. Any metric computed on this
        split is uninterpretable.
    ``HIGH``
        Substantial information crosses the boundary. Metrics are inflated by an
        amount the audit quantifies.
    ``MEDIUM``
        A channel is open but its effect on metrics is bounded or unclear.
    ``INFO``
        A property worth reporting that is not leakage -- distribution shift,
        class imbalance, redundancy.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"critical": 3, "high": 2, "medium": 1, "info": 0}[self.value]


#: The channels through which information can cross the train/test boundary.
AXES = ("pair", "ligand", "protein", "temporal", "label", "redundancy", "shortcut", "schema")


@dataclass
class CheckResult:
    """The outcome of one leakage check."""

    name: str
    axis: str
    severity: Severity
    n_affected: int
    n_total: int
    summary: str
    statistics: dict[str, Any] = field(default_factory=dict)
    details: pd.DataFrame | None = None
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown axis {self.axis!r}; expected one of {AXES}")
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)

    @property
    def fraction(self) -> float:
        """Affected rows as a fraction of the rows the check examined."""
        return self.n_affected / self.n_total if self.n_total else 0.0

    @property
    def passed(self) -> bool:
        """Whether the finding is within the check's declared tolerance.

        Tolerances are not zero everywhere. Near-duplicate detection, for
        instance, has an irreducible false-positive rate, so demanding exactly
        zero would make every split fail.
        """
        return self.fraction <= self.tolerance

    def to_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "axis": self.axis,
            "severity": self.severity.value,
            "n_affected": int(self.n_affected),
            "n_total": int(self.n_total),
            "fraction": round(self.fraction, 6),
            "tolerance": self.tolerance,
            "passed": self.passed,
            "summary": self.summary,
            "statistics": self.statistics,
        }
        if include_details and self.details is not None:
            payload["details_head"] = self.details.head(20).to_dict(orient="records")
        return payload


@dataclass
class LeakageReport:
    """Every check run against one dataset/split combination."""

    dataset: str
    split: str
    results: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("created_utc", dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
        self.metadata.setdefault("leakfree_dta_version", __version__)
        self.metadata.setdefault("python", platform.python_version())
        self.metadata.setdefault("platform", platform.platform())

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def extend(self, results: Sequence[CheckResult]) -> None:
        self.results.extend(results)

    def by_axis(self) -> dict[str, list[CheckResult]]:
        grouped: dict[str, list[CheckResult]] = {}
        for result in self.results:
            grouped.setdefault(result.axis, []).append(result)
        return grouped

    @property
    def failures(self) -> list[CheckResult]:
        """Checks outside tolerance, worst first."""
        return sorted(
            (result for result in self.results if not result.passed),
            key=lambda result: (-result.severity.rank, -result.fraction),
        )

    @property
    def worst_severity(self) -> Severity:
        failures = self.failures
        if not failures:
            return Severity.INFO
        return max((result.severity for result in failures), key=lambda s: s.rank)

    def recoverable_fraction(self) -> float:
        """Fraction of test rows whose label is recoverable from the training set.

        This is the report's headline number, and its definition is narrow on
        purpose. It counts only the ``pair`` axis: a test row where the same pair,
        or a near-identical ligand against the same target, was already measured
        in training. Those rows need no model -- a lookup predicts them -- so they
        are unambiguously leakage.

        The wider channels are deliberately excluded. In a protein-cold split,
        100% of test rows involve a ligand seen somewhere in training, and in a
        ligand-cold split 100% involve a seen target; folding those into one
        number would saturate it at 1.0 for every split and measure nothing. They
        are real findings, but they belong in :meth:`channel_fractions`, where each
        can be read against the question it answers.

        The union is over row keys, not a sum of fractions, because a row usually
        leaks through more than one channel.
        """
        contributing = [
            result
            for result in self.results
            if result.axis == "pair" and result.severity is not Severity.INFO
        ]
        if not contributing:
            return 0.0

        union: set[str] = set()
        exact = True
        fallback = 0.0
        total = max((result.n_total for result in contributing), default=0)

        for result in contributing:
            keys = result.statistics.get("affected_row_keys")
            if keys is None:
                exact = False
                fallback = max(fallback, result.fraction)
            else:
                union.update(keys)

        self.metadata["recoverable_union_is_exact"] = exact
        if not union:
            return fallback
        return max(len(union) / total if total else 0.0, fallback)

    def channel_fractions(self) -> dict[str, float]:
        """The largest affected fraction per axis, reported separately.

        Kept separate from :meth:`recoverable_fraction` because each axis answers a
        different question. A split with 100% ligand overlap and 0% protein overlap
        measures generalisation to new proteins for known compounds -- a legitimate
        task, badly described in most papers as prediction in general.
        """
        fractions: dict[str, float] = {}
        for result in self.results:
            if result.severity is Severity.INFO:
                continue
            fractions[result.axis] = max(fractions.get(result.axis, 0.0), result.fraction)
        return {axis: round(value, 5) for axis, value in sorted(fractions.items())}

    # --- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "metadata": self.metadata,
            "worst_severity": self.worst_severity.value,
            "n_checks": len(self.results),
            "n_failed": len(self.failures),
            "recoverable_fraction": round(self.recoverable_fraction(), 6),
            "channel_fractions": self.channel_fractions(),
            "results": [result.to_dict(include_details=True) for result in self.results],
        }

    def save(self, directory: str | Path, *, write_details: bool = True) -> Path:
        """Write ``report.json``, ``report.md`` and per-check detail tables."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        write_json(self.to_dict(), directory / "report.json")
        write_text(self.to_markdown(), directory / "report.md")

        if write_details:
            details_dir = directory / "details"
            for result in self.results:
                if result.details is not None and len(result.details):
                    write_table(result.details, details_dir / f"{result.name}.parquet")

        return directory

    def to_markdown(self) -> str:
        """Render the report as the Markdown table that goes into the paper."""
        lines: list[str] = [
            f"# Leakage audit: `{self.dataset}` / `{self.split}`",
            "",
            f"- **Worst severity:** {self.worst_severity.value}",
            f"- **Checks run:** {len(self.results)} ({len(self.failures)} outside tolerance)",
            f"- **Test rows whose label is recoverable by lookup:** "
            f"{self.recoverable_fraction():.2%}"
            + ("" if self.metadata.get("recoverable_union_is_exact", True) else " (lower bound)"),
            "",
            "Affected fraction by channel: "
            + ", ".join(f"`{axis}` {value:.1%}" for axis, value in self.channel_fractions().items()),
            "",
        ]

        if self.metadata:
            lines += ["## Provenance", ""]
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        lines += [
            "## Findings",
            "",
            "| check | axis | severity | affected | of | % | within tolerance |",
            "| --- | --- | --- | --: | --: | --: | :-: |",
        ]
        ordered = sorted(
            self.results, key=lambda result: (-result.severity.rank, result.axis, result.name)
        )
        for result in ordered:
            mark = "yes" if result.passed else "**NO**"
            lines.append(
                f"| `{result.name}` | {result.axis} | {result.severity.value} | "
                f"{result.n_affected:,} | {result.n_total:,} | {result.fraction:.3%} | {mark} |"
            )
        lines.append("")

        failures = self.failures
        if failures:
            lines += ["## What each failure means", ""]
            for result in failures:
                lines += [f"### `{result.name}` ({result.severity.value})", "", result.summary, ""]
                if result.statistics:
                    for key, value in result.statistics.items():
                        if key == "affected_row_keys":
                            continue
                        lines.append(f"- `{key}`: {_format_stat(value)}")
                    lines.append("")
        else:
            lines += ["## What each failure means", "", "No check fell outside its tolerance.", ""]

        return "\n".join(lines)


def _format_stat(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)) and len(value) > 8:
        return f"[{len(value)} values]"
    if isinstance(value, dict) and len(value) > 8:
        return f"{{{len(value)} entries}}"
    return str(value)
