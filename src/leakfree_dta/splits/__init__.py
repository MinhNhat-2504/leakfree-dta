"""Split construction.

Three axes, composed independently:

``ligand``   random | scaffold | cluster
``protein``  random | identity cluster (30% / 50%) | family holdout
``temporal`` none | year cutoff

Composing them gives the matrix the paper reports instead of a single number. The
reason for the matrix rather than one "hard" split is that each axis answers a
different question -- new chemistry for a known target, a known compound against a
new protein, and genuinely prospective prediction -- and a model good at one can be
poor at another.
"""

from leakfree_dta.splits.base import SplitAssignment, SplitSpec
from leakfree_dta.splits.three_axis import ThreeAxisSplitter, build_split

__all__ = ["SplitAssignment", "SplitSpec", "ThreeAxisSplitter", "build_split"]
