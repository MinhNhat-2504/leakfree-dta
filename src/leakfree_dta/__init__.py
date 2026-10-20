"""leakfree-dta: a leakage-audited benchmark and shift-robust baseline for drug-target affinity.

The package is organised around four questions the paper asks:

``leakfree_dta.data``
    What is in the benchmark, once it is curated to a single canonical schema?
``leakfree_dta.audit``
    How much of it leaks across the train/test boundary, and through which channel?
``leakfree_dta.splits``
    What does a split look like that closes every channel at once?
``leakfree_dta.models`` / ``leakfree_dta.eval``
    How much reported performance survives, and what baseline holds up under shift?
"""

from leakfree_dta._version import __version__

__all__ = ["__version__"]
