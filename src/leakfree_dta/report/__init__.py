"""Turning computed results into the paper's tables and figures.

Nothing here computes a result. Every number it prints was produced by the audit,
the splits or the evaluation and written to disk, so a figure can always be traced
back to the run that produced it. Regenerating the report never changes a number.
"""

from leakfree_dta.report.paper import build_report

__all__ = ["build_report"]
