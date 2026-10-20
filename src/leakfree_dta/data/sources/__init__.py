"""Source loaders.

Each loader's job is narrow: read one source's raw format and rename its columns to
the pre-canonical names that :func:`leakfree_dta.data.curate.curate` expects. No
filtering, no standardisation, no unit conversion beyond what the source's own
format forces -- those decisions belong in curation, where they are logged.

The split between loading and curating is what makes the audit comparable across
sources. If each loader applied its own filters, a difference between two
benchmarks could be a property of the data or an artifact of its loader, and there
would be no way to tell which.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from leakfree_dta.data.sources import benchmarks, bindingdb, chembl

#: Name -> loader. The CLI resolves ``--source`` through this registry, so adding a
#: source means adding one entry here and nothing else.
LOADERS: dict[str, Callable[..., pd.DataFrame]] = {
    "chembl": chembl.load,
    "bindingdb": bindingdb.load,
    "davis": benchmarks.load_davis,
    "kiba": benchmarks.load_kiba,
    "pdbbind": benchmarks.load_pdbbind,
}

__all__ = ["LOADERS", "benchmarks", "bindingdb", "chembl"]
