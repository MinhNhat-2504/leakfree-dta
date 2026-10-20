"""Dataset curation and loading.

Every source is reduced to the canonical schema in :mod:`leakfree_dta.data.schema`
before anything else happens. The audit compares datasets to each other, so if
"the same ligand" meant different things in different loaders the comparison would
be meaningless.

Note that ``leakfree_dta.data.curate`` is the *module*; the entry point is
``leakfree_dta.data.curate.curate``. Re-exporting the function here would shadow
the module and make ``from leakfree_dta.data import curate`` ambiguous.
"""

from leakfree_dta.data import curate, schema

__all__ = ["curate", "schema"]
