"""The paper's methodological contribution.

Two components, each addressing one half of what goes wrong when a model built on
a leaky benchmark meets a real screening campaign:

:mod:`leakfree_dta.models.robust.conformal`
    The predictions come with intervals that hold their stated coverage *within
    each protein group*, not merely on average. Average coverage is easy and
    useless: a model can hit 90% overall while covering 99% of a well-sampled
    kinase and 40% of everything rare.

:mod:`leakfree_dta.models.robust.group_dro`
    The model is trained to minimise its *worst* group's loss rather than its
    average loss, so performance does not collapse on the groups a screening
    campaign is most likely to care about.

Composed in :mod:`leakfree_dta.models.robust.shift_robust`.
"""

from leakfree_dta.models.robust.conformal import MondrianConformal
from leakfree_dta.models.robust.shift_robust import ShiftRobustDTA

__all__ = ["MondrianConformal", "ShiftRobustDTA"]
