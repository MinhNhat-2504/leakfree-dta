"""Re-running published models on our splits.

The paper's re-evaluation stands or falls on whether a reader believes the
published models were run faithfully. Three rules are built into this package
rather than left to discipline:

**Only the data and the split are replaced.** Architecture, optimiser and
hyperparameters stay as the authors set them. An adapter that changes anything
else must declare it in ``adaptations``, and the declaration is printed in the
attempt log that ships with the paper. Re-tuning someone else's model and
reporting a lower number would measure our tuning effort, not their contribution.

**The environment is pinned and recorded.** Each model is cloned at a fixed
commit into its own directory, and the recorded provenance is enough to repeat
the run: commit hash, Python version, package versions, the exact command.

**Failures are published.** A model that will not run gets an attempt-log entry
with a reason, not silence. The models hardest to reproduce are disproportionately
the oldest, so dropping them would bias the re-measured trend upward -- which is
precisely the trend the paper is trying to measure honestly.
"""

from leakfree_dta.external.harness import (
    AdapterProtocol,
    RerunOutcome,
    RerunWorkspace,
    rerun_model,
)

__all__ = ["AdapterProtocol", "RerunOutcome", "RerunWorkspace", "rerun_model"]
