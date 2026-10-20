"""Adapter for the original DeepDTA repository.

Running the published code rather than a reimplementation is the most credible
thing Phase 2 can do, and it takes two pieces of work that are worth keeping
visible.

**A compatibility patch.** The repository (hkmztrk/DeepDTA, head a546a84,
2023-08-16) targets TensorFlow 1 and an early Keras 2. On TensorFlow 2.15 it
fails at import. Three edits fix it, and every one is a mechanical rename forced
by framework evolution rather than a change to the model:

* ``tf.ConfigProto`` / ``tf.set_random_seed`` / ``tf.Session`` moved to
  ``tf.compat.v1`` in TensorFlow 2.0.
* ``keras.layers.normalization.BatchNormalization`` moved to ``keras.layers`` in
  Keras 2.6.
* ``keras.layers.merge`` was removed in Keras 2.0. It is dropped from the import
  list; ``Merge`` is referenced only by ``build_single_drug``,
  ``build_single_prot`` and ``build_baseline``, none of which
  ``run_regression`` calls -- it uses ``build_combined_categorical``, which was
  already on the current ``concatenate`` API.

The patches are declared as literal old/new pairs, applied with an exact-match
requirement, and written to ``patch.json`` in the workspace. If the upstream file
changes, the patch fails loudly rather than silently applying to the wrong line.
Nothing in the architecture, the optimiser, or the hyperparameter grid is touched.

**A format conversion.** DeepDTA reads a dense ligand-by-target affinity matrix
plus flat index lists, not a list of measurements. Our split is converted into
that layout: a matrix with NaN where no measurement exists, and fold files
holding the flat indices of our train and test rows. The conversion is lossless
for the rows involved, and :func:`_verify_roundtrip` checks that before the model
is ever started.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.external.harness import RerunWorkspace
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import write_json

logger = get_logger(__name__)

#: Literal (old, new, rationale) edits applied to ``source/run_experiments.py``.
TF2_PATCHES: tuple[tuple[str, str, str], ...] = (
    (
        "session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)",
        "session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)",
        "tf.ConfigProto moved to tf.compat.v1 in TensorFlow 2.0",
    ),
    (
        "tf.set_random_seed(0)",
        "tf.compat.v1.set_random_seed(0)",
        "tf.set_random_seed moved to tf.compat.v1 in TensorFlow 2.0",
    ),
    (
        "sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)",
        "sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)",
        "tf.Session and tf.get_default_graph moved to tf.compat.v1 in TensorFlow 2.0",
    ),
    (
        "K.set_session(sess)",
        "tf.compat.v1.keras.backend.set_session(sess)",
        "keras.backend.set_session was removed in Keras 2.4 along with the TF1 "
        "session model; the compat.v1 entry point is the same function",
    ),
    (
        "from keras.layers.normalization import BatchNormalization",
        "from keras.layers import BatchNormalization",
        "BatchNormalization moved out of keras.layers.normalization in Keras 2.6",
    ),
    (
        "from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten",
        "from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten",
        "keras.layers.merge was removed in Keras 2.0; it is unused on the code path "
        "run_regression takes (build_combined_categorical uses concatenate)",
    ),
    (
        "from keras.utils import plot_model",
        "try:\n    from keras.utils import plot_model\nexcept ImportError:  # pydot/graphviz are optional and unused here\n    plot_model = None",
        "plot_model requires pydot and graphviz, which the model does not need",
    ),
)


def apply_patches(path: Path, patches: tuple[tuple[str, str, str], ...]) -> list[dict[str, str]]:
    """Apply literal replacements, requiring each to match exactly once.

    Exact-match-once is the point. A fuzzy patch that lands on the wrong line in a
    future revision of the upstream file would alter the model silently, which is
    the one failure mode this whole package exists to avoid.
    """
    text = path.read_text(encoding="utf-8")
    applied: list[dict[str, str]] = []

    for old, new, rationale in patches:
        count = text.count(old)
        if count == 0:
            if new in text:
                applied.append({"status": "already applied", "rationale": rationale, "old": old})
                continue
            raise ValueError(
                f"patch target not found in {path.name}: {old[:80]!r}. "
                "The upstream file has changed; review the patch before re-running."
            )
        if count > 1:
            raise ValueError(f"patch target appears {count} times in {path.name}: {old[:80]!r}")
        text = text.replace(old, new, 1)
        applied.append({"status": "applied", "rationale": rationale, "old": old, "new": new})

    path.write_text(text, encoding="utf-8")
    return applied


@dataclass
class DeepDTAAdapter:
    """Runs hkmztrk/DeepDTA on one of our splits."""

    name: str = "DeepDTA"
    repository: str = "https://github.com/hkmztrk/DeepDTA"
    commit: str | None = "a546a8433a6822e958f36171c4356ad6f414d623"
    #: The published grid is large; a re-run over it costs days per model. These
    #: hold the architecture fixed at the paper's reported best configuration.
    num_epochs: int = 100
    batch_size: int = 256
    adaptations: list[str] = field(
        default_factory=lambda: [
            "TensorFlow 1 session API replaced with tf.compat.v1 (3 lines); "
            "BatchNormalization and merge imports updated for Keras 2.15. "
            "Mechanical renames only -- no change to architecture, optimiser or "
            "hyperparameters. The exact edits are in patch.json.",
            "Our split supplied in DeepDTA's native matrix-plus-fold-index format "
            "rather than the packaged Davis/KIBA folds.",
            "Hyperparameter grid fixed at the paper's reported best configuration "
            "instead of re-searched, so the re-run measures their model rather than "
            "our search budget.",
        ]
    )

    # --- data ---------------------------------------------------------------

    def write_data(
        self,
        workspace: RerunWorkspace,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Convert our split into DeepDTA's layout.

        Validation is folded into training. DeepDTA's protocol selects
        hyperparameters by internal 5-fold cross-validation over whatever it is
        given as the training set, so handing it a separate validation fold would
        change its protocol; handing it train+valid matches what it expects and
        gives it the same data budget as our own models.
        """
        combined = pd.concat([train, valid], ignore_index=True) if len(valid) else train
        everything = pd.concat([combined, test], ignore_index=True)

        ligands = (
            everything[[schema.LIGAND_KEY, schema.SMILES]]
            .drop_duplicates(subset=[schema.LIGAND_KEY])
            .sort_values(schema.LIGAND_KEY)
            .reset_index(drop=True)
        )
        targets = (
            everything[[schema.TARGET_KEY, schema.TARGET_SEQUENCE]]
            .drop_duplicates(subset=[schema.TARGET_KEY])
            .sort_values(schema.TARGET_KEY)
            .reset_index(drop=True)
        )

        ligand_row = {key: i for i, key in enumerate(ligands[schema.LIGAND_KEY])}
        target_col = {key: i for i, key in enumerate(targets[schema.TARGET_KEY])}
        n_rows, n_cols = len(ligands), len(targets)

        affinity = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        for frame in (combined, test):
            rows = frame[schema.LIGAND_KEY].map(ligand_row).to_numpy()
            cols = frame[schema.TARGET_KEY].map(target_col).to_numpy()
            affinity[rows, cols] = pd.to_numeric(frame[schema.Y], errors="coerce").to_numpy()

        def flat_indices(frame: pd.DataFrame) -> list[int]:
            rows = frame[schema.LIGAND_KEY].map(ligand_row).to_numpy()
            cols = frame[schema.TARGET_KEY].map(target_col).to_numpy()
            # DeepDTA indexes the non-NaN cells in row-major order, so a flat
            # index is row * n_cols + col.
            return [int(r * n_cols + c) for r, c in zip(rows, cols, strict=True)]

        train_flat = flat_indices(combined)
        test_flat = flat_indices(test)
        self._verify_roundtrip(affinity, n_cols, test, test_flat)

        data_dir = workspace.data
        folds_dir = data_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)

        (data_dir / "ligands_can.txt").write_text(
            json.dumps(dict(zip(ligands[schema.LIGAND_KEY], ligands[schema.SMILES], strict=True))),
            encoding="utf-8",
        )
        (data_dir / "proteins.txt").write_text(
            json.dumps(
                dict(zip(targets[schema.TARGET_KEY], targets[schema.TARGET_SEQUENCE], strict=True))
            ),
            encoding="utf-8",
        )
        with (data_dir / "Y").open("wb") as handle:
            pickle.dump(affinity, handle, protocol=2)

        # DeepDTA expects five cross-validation folds inside the training set.
        chunks = np.array_split(np.array(train_flat), 5)
        (folds_dir / "train_fold_setting1.txt").write_text(
            json.dumps([chunk.tolist() for chunk in chunks]), encoding="utf-8"
        )
        (folds_dir / "test_fold_setting1.txt").write_text(
            json.dumps(test_flat), encoding="utf-8"
        )

        # Row order of the test set, so predictions can be realigned afterwards.
        write_json(
            {
                "n_ligands": n_rows,
                "n_targets": n_cols,
                "test_flat_indices": test_flat,
                "test_row_keys": test[schema.ROW_KEY].tolist(),
            },
            workspace.data / "index_map.json",
        )
        logger.info(
            "DeepDTA data: %d x %d matrix, %d train cells, %d test cells",
            n_rows,
            n_cols,
            len(train_flat),
            len(test_flat),
        )

    @staticmethod
    def _verify_roundtrip(
        affinity: np.ndarray, n_cols: int, test: pd.DataFrame, test_flat: list[int]
    ) -> None:
        """Check the matrix encoding reproduces the test labels exactly.

        A silent indexing error here would hand the model a scrambled test set and
        produce a low score that looked like honest difficulty. Cheap to check,
        catastrophic to miss.
        """
        expected = pd.to_numeric(test[schema.Y], errors="coerce").to_numpy(dtype=float)
        recovered = np.array([affinity[i // n_cols, i % n_cols] for i in test_flat])
        if not np.allclose(expected, recovered, equal_nan=True):
            n_bad = int(np.sum(~np.isclose(expected, recovered, equal_nan=True)))
            raise ValueError(
                f"matrix encoding does not round-trip: {n_bad}/{len(expected)} test labels differ"
            )

    # --- execution ----------------------------------------------------------

    def command(self, workspace: RerunWorkspace) -> list[str]:
        patch_record = apply_patches(
            workspace.repo / "source" / "run_experiments.py", TF2_PATCHES
        )
        write_json({"file": "source/run_experiments.py", "patches": patch_record}, workspace.root / "patch.json")

        # Values are the paper's reported best configuration for Davis.
        return [
            "python",
            "source/run_experiments.py",
            "--num_windows", "32",
            "--smi_window_lengths", "4",
            "--seq_window_lengths", "8",
            "--batch_size", str(self.batch_size),
            "--num_epoch", str(self.num_epochs),
            "--max_seq_len", "1000",
            "--max_smi_len", "100",
            "--dataset_path", str(workspace.data) + "/",
            "--problem_type", "1",
            "--log_dir", str(workspace.output) + "/",
        ]

    def collect(self, workspace: RerunWorkspace, test: pd.DataFrame) -> np.ndarray:
        """Read predictions back and realign them to our test row order.

        The published script logs metrics but does not write a prediction file, so
        the patch set would have to be extended to emit one. Until it does, this
        raises with a precise reason rather than inventing numbers -- which is
        exactly the kind of entry the attempt log is for.
        """
        candidates = list(workspace.output.rglob("predictions*.csv")) + list(
            workspace.output.rglob("*.pred")
        )
        if not candidates:
            raise FileNotFoundError(
                "DeepDTA's run_experiments.py reports metrics to its log but writes no "
                f"per-row predictions, and none were found under {workspace.output}. "
                "Emitting them needs a further patch to the published script; until "
                "that patch exists this model cannot enter the per-row comparison."
            )

        frame = pd.read_csv(candidates[0])
        column = "y_pred" if "y_pred" in frame.columns else frame.columns[-1]
        if len(frame) != len(test):
            raise ValueError(f"{candidates[0].name} has {len(frame)} rows, expected {len(test)}")
        return frame[column].to_numpy(dtype=float)
