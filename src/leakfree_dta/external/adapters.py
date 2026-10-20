"""Adapters for the published models the paper re-runs.

Each adapter is deliberately thin: lay the data out, name a command, read the
predictions back. Everything else is the harness's job, so that a difference
between two models cannot be an artefact of how they were run.

A note on what these adapters are for. Several of them are expected to fail, and
that is a result rather than a problem. A 2018 Keras/TensorFlow-1 model does not
run under Python 3.11, and saying so with a log tail attached is more useful to a
reader than quietly dropping the model -- especially since the models hardest to
reproduce are disproportionately the oldest, so dropping them would tilt the
re-measured trend upward exactly where the paper is trying to measure it.

Where a model cannot be run as published, the paper's comparison point is a
reimplementation carried in ``leakfree_dta.models`` under a name ending in
``_reimpl``. The naming is load-bearing: no table should be able to present a
reimplementation as a re-run of the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.external.deepdta import DeepDTAAdapter
from leakfree_dta.external.harness import RerunWorkspace
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import write_table

logger = get_logger(__name__)


def write_standard_csv(workspace: RerunWorkspace, train, valid, test) -> dict[str, Path]:
    """Write the split as three plain CSVs.

    The lowest common denominator: every adapter that needs something else builds
    it from these, and a reader who wants to run a model by hand has files they
    can point at.
    """
    columns = [
        schema.ROW_KEY,
        schema.SMILES,
        schema.TARGET_KEY,
        schema.TARGET_SEQUENCE,
        schema.Y,
    ]
    written: dict[str, Path] = {}
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        written[name] = write_table(frame[columns], workspace.data / f"{name}.csv")
    return written


@dataclass
class OriginalRepoAdapter:
    """Runs a published repository as published.

    Carries no cleverness on purpose. If the published entry point needs
    arguments the repository does not document, that is a finding for the attempt
    log, not something to paper over with guesswork about what the authors meant.
    """

    name: str
    repository: str
    commit: str | None
    entrypoint: list[str]
    adaptations: list[str] = field(default_factory=list)
    prediction_file: str = "predictions.csv"
    prediction_column: str = "y_pred"

    def write_data(self, workspace: RerunWorkspace, train, valid, test) -> None:
        write_standard_csv(workspace, train, valid, test)

    def command(self, workspace: RerunWorkspace) -> list[str]:
        return [
            token.replace("{data}", str(workspace.data)).replace("{output}", str(workspace.output))
            for token in self.entrypoint
        ]

    def collect(self, workspace: RerunWorkspace, test: pd.DataFrame) -> np.ndarray:
        path = workspace.output / self.prediction_file
        if not path.exists():
            raise FileNotFoundError(
                f"the run produced no {self.prediction_file} in {workspace.output}"
            )
        frame = pd.read_csv(path)
        if self.prediction_column not in frame.columns:
            raise KeyError(
                f"{path.name} has columns {list(frame.columns)}, expected "
                f"{self.prediction_column!r}"
            )
        # Align by row key when the model preserved it; fall back to file order,
        # which is only safe because write_standard_csv wrote test rows in order.
        if schema.ROW_KEY in frame.columns:
            ordered = frame.set_index(schema.ROW_KEY).reindex(test[schema.ROW_KEY])
            return ordered[self.prediction_column].to_numpy(dtype=float)
        if len(frame) != len(test):
            raise ValueError(f"{path.name} has {len(frame)} rows, expected {len(test)}")
        logger.warning(
            "%s did not preserve row keys; aligning by file order, which assumes the "
            "model did not reorder the test set",
            self.name,
        )
        return frame[self.prediction_column].to_numpy(dtype=float)


#: The five models named in the pre-registered target list, as adapters.
#:
#: Commits are pinned where a release exists. Where the field is ``None`` the
#: harness records whatever head it obtained, so the run remains reproducible
#: afterwards even though it was not pinned beforehand.
ADAPTERS: dict[str, object] = {
    # DeepDTA has a dedicated adapter: it needs a declared TensorFlow 2
    # compatibility patch and its own matrix-plus-fold-index data format.
    "DeepDTA": DeepDTAAdapter(),
    "GraphDTA": OriginalRepoAdapter(
        name="GraphDTA",
        repository="https://github.com/thinng/GraphDTA",
        commit=None,
        entrypoint=["python", "training.py", "0", "0", "0"],
        adaptations=[
            "Data supplied from our split rather than the packaged folds.",
        ],
    ),
    "MolTrans": OriginalRepoAdapter(
        name="MolTrans",
        repository="https://github.com/kexinhuang12345/MolTrans",
        commit=None,
        entrypoint=["python", "train.py", "--task", "custom"],
        adaptations=[
            "Data supplied from our split rather than the packaged folds.",
            "The published model reports classification metrics; its regression head "
            "is used here so the number is comparable to the others.",
        ],
    ),
    "PerceiverCPI": OriginalRepoAdapter(
        name="PerceiverCPI",
        repository="https://github.com/dmis-lab/PerceiverCPI",
        commit=None,
        entrypoint=["python", "train.py", "--data_path", "{data}/train.csv"],
        adaptations=[
            "Data supplied from our split rather than the packaged folds.",
        ],
    ),
}


def available_adapters() -> list[str]:
    return sorted(ADAPTERS)
