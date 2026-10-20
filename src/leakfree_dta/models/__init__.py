"""Models.

Three kinds live here, for three different jobs:

``baselines``
    Cheap, fully reproducible models whose only job is to establish what the
    benchmark is worth without deep learning. If a ridge regression on ECFP4 comes
    within noise of a graph transformer on the leakage-free split, that is the
    paper's result, not a failure of the experiment.
``robust``
    The paper's methodological contribution: group DRO over protein families with
    a Mondrian conformal head, which trades a little average accuracy for
    performance that holds up under distribution shift and for intervals that mean
    what they say.
``external``
    A registry for re-running published models, with provenance. Not wrappers --
    the published code is run as published, and this records exactly which commit
    produced which number.
"""

from leakfree_dta.models.base import DTAModel, FitResult
from leakfree_dta.models.registry import MODELS, build_model

__all__ = ["MODELS", "DTAModel", "FitResult", "build_model"]
