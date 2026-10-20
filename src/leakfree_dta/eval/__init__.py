"""Evaluation.

Three principles the paper holds to, implemented here rather than left to each
experiment script:

* **Every number carries an interval.** A 0.01 improvement in Pearson means
  nothing without one, and most of the field's reported gains are inside it.
* **Intervals are computed over groups, not rows.** Chemical redundancy makes rows
  non-independent, so row-level bootstrapping produces intervals that are too
  narrow by a factor the audit measures.
* **Worst-group performance is reported beside the mean.** A model that is good on
  average and useless on rare targets is not useful for screening.
"""

from leakfree_dta.eval.bootstrap import BootstrapResult, bootstrap_metric, compare_models
from leakfree_dta.eval.metrics import METRICS, MetricSet, evaluate

__all__ = [
    "METRICS",
    "BootstrapResult",
    "MetricSet",
    "bootstrap_metric",
    "compare_models",
    "evaluate",
]
