# ruff: noqa: RUF012 - the stub adapters below mirror the protocol's attribute
# shape and are never mutated; a ClassVar annotation would obscure that.
"""The re-run harness, and the patch mechanism it depends on.

The patch tests matter more than their size suggests. A compatibility patch that
applied to the wrong line would change a published model silently and produce a
number the paper would then attribute to its authors. The mechanism is therefore
built to fail loudly, and these tests check that it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakfree_dta.data import schema
from leakfree_dta.external.deepdta import TF2_PATCHES, DeepDTAAdapter, apply_patches
from leakfree_dta.external.harness import RerunWorkspace, environment_provenance, rerun_model


class TestPatchMechanism:
    def test_applies_an_exact_match(self, tmp_path):
        path = tmp_path / "code.py"
        path.write_text("import tensorflow as tf\nsess = tf.Session()\n", encoding="utf-8")

        record = apply_patches(
            path, (("tf.Session()", "tf.compat.v1.Session()", "moved in TF 2.0"),)
        )

        assert "tf.compat.v1.Session()" in path.read_text(encoding="utf-8")
        assert record[0]["status"] == "applied"
        assert record[0]["rationale"] == "moved in TF 2.0"

    def test_missing_target_raises(self, tmp_path):
        """An upstream change must stop the run, not be patched around."""
        path = tmp_path / "code.py"
        path.write_text("print('hello')\n", encoding="utf-8")

        with pytest.raises(ValueError, match="patch target not found"):
            apply_patches(path, (("tf.Session()", "tf.compat.v1.Session()", "why"),))

    def test_ambiguous_target_raises(self, tmp_path):
        """Two matches means the patch cannot know which line it is for."""
        path = tmp_path / "code.py"
        path.write_text("x = f()\ny = f()\n", encoding="utf-8")

        with pytest.raises(ValueError, match="appears 2 times"):
            apply_patches(path, (("f()", "g()", "why"),))

    def test_already_applied_is_tolerated(self, tmp_path):
        """Re-running on a patched workspace must not fail."""
        path = tmp_path / "code.py"
        path.write_text("sess = tf.compat.v1.Session()\n", encoding="utf-8")

        record = apply_patches(
            path, (("tf.Session()", "tf.compat.v1.Session()", "moved in TF 2.0"),)
        )
        assert record[0]["status"] == "already applied"

    def test_every_declared_patch_has_a_rationale(self):
        """No silent edits: each patch must say why it is justified."""
        for old, new, rationale in TF2_PATCHES:
            assert old and new and rationale
            assert old != new
            assert len(rationale) > 20, f"rationale too thin for {old[:40]!r}"


class TestDeepDTADataFormat:
    """The matrix conversion, which is where a silent scramble could hide."""

    @pytest.fixture
    def folds(self, aggregated):
        from leakfree_dta.splits.base import SplitSpec
        from leakfree_dta.splits.three_axis import build_split

        assignment = build_split(aggregated, SplitSpec(name="t", ligand_axis="scaffold"))
        return (
            assignment.train(aggregated),
            assignment.valid(aggregated),
            assignment.test(aggregated),
        )

    def test_matrix_round_trips_the_test_labels(self, folds, tmp_path):
        """Every test label must be recoverable from the matrix at its flat index.

        The adapter asserts this itself before starting the model; the test checks
        the assertion is actually reached and passes on real split data.
        """
        train, valid, test = folds
        workspace = RerunWorkspace(root=tmp_path, model="DeepDTA", dataset="t", split="s")
        workspace.prepare()

        DeepDTAAdapter().write_data(workspace, train, valid, test)

        import json
        import pickle

        with (workspace.data / "Y").open("rb") as handle:
            affinity = pickle.load(handle)
        index_map = json.loads((workspace.data / "index_map.json").read_text(encoding="utf-8"))

        n_cols = index_map["n_targets"]
        recovered = np.array(
            [affinity[i // n_cols, i % n_cols] for i in index_map["test_flat_indices"]]
        )
        expected = pd.to_numeric(test[schema.Y], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(recovered, expected, equal_nan=True)

    def test_unmeasured_cells_stay_nan(self, folds, tmp_path):
        """The matrix is sparse; absent pairs must not become zeros.

        A zero would be read as a real measurement of pAffinity 0, which is 1 M --
        and would train the model on hundreds of thousands of invented inactives.
        """
        train, valid, test = folds
        workspace = RerunWorkspace(root=tmp_path, model="DeepDTA", dataset="t", split="s")
        workspace.prepare()
        DeepDTAAdapter().write_data(workspace, train, valid, test)

        import pickle

        with (workspace.data / "Y").open("rb") as handle:
            affinity = pickle.load(handle)

        n_measured = int(np.isfinite(affinity).sum())
        n_expected = len(pd.concat([train, valid, test])[schema.PAIR_KEY].unique())
        assert n_measured == n_expected
        assert np.isnan(affinity).any(), "the matrix is dense; the fixture is unrealistic"

    def test_fold_files_are_valid_json_lists(self, folds, tmp_path):
        train, valid, test = folds
        workspace = RerunWorkspace(root=tmp_path, model="DeepDTA", dataset="t", split="s")
        workspace.prepare()
        DeepDTAAdapter().write_data(workspace, train, valid, test)

        import json

        train_folds = json.loads(
            (workspace.data / "folds" / "train_fold_setting1.txt").read_text(encoding="utf-8")
        )
        test_fold = json.loads(
            (workspace.data / "folds" / "test_fold_setting1.txt").read_text(encoding="utf-8")
        )

        assert len(train_folds) == 5, "DeepDTA expects five cross-validation folds"
        assert all(isinstance(fold, list) for fold in train_folds)
        assert len(test_fold) == len(test)
        # No index may appear in both training and test.
        assert not set(test_fold) & {i for fold in train_folds for i in fold}


class TestHarness:
    def test_a_failing_command_produces_an_outcome_not_an_exception(self, aggregated, tmp_path):
        """A model that will not run is data for the attempt log, not a crash."""

        class Broken:
            name = "broken"
            repository = ""
            commit = None
            adaptations = ["none"]

            def write_data(self, workspace, train, valid, test):
                pass

            def command(self, workspace):
                return ["python", "-c", "import sys; sys.exit(3)"]

            def collect(self, workspace, test):
                raise AssertionError("should not be reached")

        outcome = rerun_model(
            Broken(),
            aggregated.head(50),
            aggregated.head(10),
            aggregated.head(20),
            dataset="t",
            split="s",
            workspace_root=tmp_path,
        )

        assert outcome.status == "failed"
        assert "exit 3" in outcome.reason
        assert outcome.provenance["adaptations"] == "none"

    def test_wrong_prediction_count_is_rejected(self, aggregated, tmp_path):
        """Silently accepting a misaligned prediction vector would be worse than failing."""

        class Misaligned:
            name = "misaligned"
            repository = ""
            commit = None
            adaptations = []

            def write_data(self, workspace, train, valid, test):
                pass

            def command(self, workspace):
                return ["python", "-c", "pass"]

            def collect(self, workspace, test):
                return np.zeros(len(test) + 5)

        outcome = rerun_model(
            Misaligned(),
            aggregated.head(50),
            aggregated.head(10),
            aggregated.head(20),
            dataset="t",
            split="s",
            workspace_root=tmp_path,
        )
        assert outcome.status == "failed"
        assert "test rows" in outcome.reason

    def test_undeclared_adaptations_are_visible_as_such(self, aggregated, tmp_path):
        class Silent:
            name = "silent"
            repository = ""
            commit = None
            adaptations = []

            def write_data(self, workspace, train, valid, test):
                pass

            def command(self, workspace):
                return ["python", "-c", "pass"]

            def collect(self, workspace, test):
                return np.zeros(len(test))

        outcome = rerun_model(
            Silent(),
            aggregated.head(50),
            aggregated.head(10),
            aggregated.head(20),
            dataset="t",
            split="s",
            workspace_root=tmp_path,
        )
        # An empty list must read as an explicit claim, not as a blank cell.
        assert outcome.provenance["adaptations"] == "(none declared)"

    def test_provenance_records_the_environment(self):
        record = environment_provenance()
        assert "python" in record
        assert "numpy" in record
        assert "platform" in record
