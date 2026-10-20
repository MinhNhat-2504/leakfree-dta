"""Leakage auditing.

The audit answers one question in a way a reviewer can check: for a given dataset
and a given train/test split, through which channels does information cross the
boundary, and how much?

Channels are kept separate on purpose. A benchmark can be clean on the ligand axis
and badly compromised on the protein axis, and reporting a single leakage number
would hide exactly the finding this paper is about.
"""

from leakfree_dta.audit.report import CheckResult, LeakageReport, Severity
from leakfree_dta.audit.runner import audit_dataset, audit_split

__all__ = ["CheckResult", "LeakageReport", "Severity", "audit_dataset", "audit_split"]
