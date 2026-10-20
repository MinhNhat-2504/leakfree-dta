"""The re-run harness: workspace, provenance, execution, collection.

One function, :func:`rerun_model`, drives the whole thing. An adapter supplies the
three model-specific pieces -- how to lay the data out, what command to run, and
how to read predictions back -- and the harness handles everything that must be
identical across models, which is everything else.

The separation matters for the paper's credibility. If each model had its own
script, a reader could not tell whether a difference between two models came from
the models or from how they were run. Here the only model-specific code is the
adapter, and each adapter is short enough to read.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import write_json, write_table

logger = get_logger(__name__)

#: How long any single model is allowed to run before being abandoned. A model
#: that exceeds it is an attempt-log entry, not a stalled pipeline.
DEFAULT_TIMEOUT_SECONDS = 24 * 3600


@dataclass
class RerunWorkspace:
    """Where one model's re-run lives on disk.

    Every model gets its own tree so that a published repository's relative paths,
    its habit of writing into its own directory, and its dependency on files it
    expects beside the script all work without interfering with anything else.
    """

    root: Path
    model: str
    dataset: str
    split: str

    @property
    def repo(self) -> Path:
        """The cloned source, untouched except for the data we place in it."""
        return self.root / "repo"

    @property
    def data(self) -> Path:
        """Our split, written in whatever layout the model expects."""
        return self.root / "data"

    @property
    def output(self) -> Path:
        """Where the model writes, and where predictions are collected from."""
        return self.root / "output"

    @property
    def log(self) -> Path:
        return self.root / "run.log"

    def prepare(self) -> None:
        for directory in (self.root, self.data, self.output):
            directory.mkdir(parents=True, exist_ok=True)


@runtime_checkable
class AdapterProtocol(Protocol):
    """What a model-specific adapter has to provide."""

    name: str
    repository: str
    commit: str | None
    #: Anything changed from the published code, in plain language. Printed in the
    #: attempt log. An empty list is a claim that nothing was changed.
    adaptations: list[str]

    def write_data(
        self,
        workspace: RerunWorkspace,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Lay our split out in the format the published code reads."""

    def command(self, workspace: RerunWorkspace) -> list[str]:
        """The command to run, as an argv list."""

    def collect(self, workspace: RerunWorkspace, test: pd.DataFrame) -> np.ndarray:
        """Read predictions back, aligned to ``test`` row order."""


@dataclass
class RerunOutcome:
    """What happened, whether or not it worked."""

    model: str
    dataset: str
    split: str
    status: str  # "ok" | "failed" | "timeout" | "skipped"
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    predictions_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "status": self.status,
            "reason": self.reason,
            "elapsed_minutes": round(self.elapsed_seconds / 60, 2),
            "predictions": self.predictions_path,
            **{f"metric_{key}": value for key, value in self.metrics.items()},
            **{f"provenance_{key}": value for key, value in self.provenance.items()},
        }


# --- repository handling ----------------------------------------------------


def clone_at_commit(repository: str, commit: str | None, destination: Path) -> dict[str, str]:
    """Clone ``repository`` and check out ``commit``, returning what was obtained.

    When ``commit`` is ``None`` the default branch's current head is used and its
    hash is recorded, so the run is still reproducible afterwards even though it
    was not pinned beforehand. Pinning is better and the adapters should do it;
    recording is the floor.
    """
    if destination.exists() and (destination / ".git").exists():
        logger.info("reusing existing clone at %s", destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s", repository)
        subprocess.run(
            ["git", "clone", "--quiet", repository, str(destination)],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )

    if commit:
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--quiet", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {"repository": repository, "requested_commit": commit or "(default head)", "commit": head}


def environment_provenance() -> dict[str, str]:
    """Versions that could plausibly change a result."""
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for module in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "torch", "tensorflow"):
        try:
            imported = __import__(module)
            record[module] = getattr(imported, "__version__", "?")
        except ImportError:
            record[module] = "not installed"
    try:
        import torch

        record["cuda"] = torch.version.cuda or "cpu-only"
        record["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        )
    except ImportError:
        pass
    return record


# --- the driver -------------------------------------------------------------


def rerun_model(
    adapter: AdapterProtocol,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    workspace_root: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    skip_clone: bool = False,
) -> RerunOutcome:
    """Run one published model on one split, recording everything.

    Never raises on a model's failure. A model that crashes, times out, or returns
    unusable predictions produces an outcome with ``status != "ok"`` and a stated
    reason, because that is what the attempt log needs.
    """
    root = (workspace_root or paths.EXTERNAL) / dataset / split / adapter.name
    workspace = RerunWorkspace(root=root, model=adapter.name, dataset=dataset, split=split)
    workspace.prepare()

    provenance: dict[str, Any] = {
        "adaptations": "; ".join(adapter.adaptations) or "(none declared)",
        **{f"env_{key}": value for key, value in environment_provenance().items()},
    }
    started = time.monotonic()

    def finish(status: str, reason: str = "", **extra: Any) -> RerunOutcome:
        outcome = RerunOutcome(
            model=adapter.name,
            dataset=dataset,
            split=split,
            status=status,
            reason=reason,
            provenance=provenance,
            elapsed_seconds=time.monotonic() - started,
            **extra,
        )
        write_json(outcome.to_dict(), root / "outcome.json")
        return outcome

    # --- obtain the source -------------------------------------------------
    if not skip_clone and adapter.repository:
        try:
            provenance.update(clone_at_commit(adapter.repository, adapter.commit, workspace.repo))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            return finish("failed", f"clone failed: {str(detail)[:300]}")

    # --- lay out the data --------------------------------------------------
    try:
        adapter.write_data(workspace, train, valid, test)
    except Exception as exc:
        return finish("failed", f"data preparation failed: {type(exc).__name__}: {exc}")

    # --- run ---------------------------------------------------------------
    command = adapter.command(workspace)
    logger.info("running %s: %s", adapter.name, " ".join(command))
    provenance["command"] = " ".join(command)

    try:
        with workspace.log.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {' '.join(command)}\n\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=workspace.repo if workspace.repo.exists() else workspace.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return finish("timeout", f"exceeded {timeout / 3600:.1f} h")
    except OSError as exc:
        return finish("failed", f"could not start: {exc}")

    if completed.returncode != 0:
        tail = _log_tail(workspace.log)
        return finish("failed", f"exit {completed.returncode}; log tail: {tail}")

    # --- collect -----------------------------------------------------------
    try:
        predictions = adapter.collect(workspace, test)
    except Exception as exc:
        return finish("failed", f"could not read predictions: {type(exc).__name__}: {exc}")

    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(test),):
        return finish(
            "failed",
            f"predicted {predictions.shape} values for {len(test)} test rows",
        )
    n_finite = int(np.isfinite(predictions).sum())
    if n_finite < len(test):
        return finish("failed", f"only {n_finite}/{len(test)} predictions are finite")

    # --- score -------------------------------------------------------------
    from leakfree_dta.eval.metrics import evaluate

    y_true = pd.to_numeric(test[schema.Y], errors="coerce").to_numpy(dtype=float)
    metrics = evaluate(y_true, predictions)

    frame = pd.DataFrame(
        {
            schema.ROW_KEY: test[schema.ROW_KEY].to_numpy(),
            schema.TARGET_KEY: test[schema.TARGET_KEY].to_numpy(),
            schema.LIGAND_KEY: test[schema.LIGAND_KEY].to_numpy(),
            "y_true": y_true,
            "y_pred": predictions,
        }
    )
    for column in (schema.TARGET_FAMILY, "protein_group", "ligand_group"):
        if column in test.columns:
            frame[column] = test[column].to_numpy()

    # Written into results/runs so that `leakfree evaluate` picks these models up
    # alongside our own, in the same comparison and the same bootstrap.
    run_dir = paths.RUNS / dataset / split / adapter.name
    predictions_path = write_table(frame, run_dir / "predictions.parquet")
    write_json(metrics.to_dict(), run_dir / "metrics.json")
    write_json(
        {
            "model": {"name": adapter.name, "source": "external"},
            "provenance": provenance,
            "split": split,
        },
        run_dir / "run.json",
    )

    logger.info(
        "%s: rmse=%.4f pearson=%.4f (%.1f min)",
        adapter.name,
        metrics.get("rmse"),
        metrics.get("pearson"),
        (time.monotonic() - started) / 60,
    )
    return finish(
        "ok",
        metrics={key: float(value) for key, value in metrics.values.items()},
        predictions_path=str(predictions_path),
    )


def _log_tail(path: Path, n_chars: int = 400) -> str:
    """The end of a failed run's log, for the attempt log's reason field."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log unreadable)"
    tail = text.strip()[-n_chars:].replace("\n", " | ")
    return tail or "(empty log)"


def write_attempt_log(outcomes: list[RerunOutcome], dataset: str, split: str) -> Path:
    """The table that ships with the paper, successes and failures together."""
    table = pd.DataFrame([outcome.to_dict() for outcome in outcomes])
    path = write_table(table, paths.TABLES / f"{dataset}.{split}.rerun_attempts.csv")

    counts = table["status"].value_counts().to_dict() if len(table) else {}
    logger.info("attempt log: %s -> %s", counts, path.name)
    if counts.get("ok", 0) < 3:
        logger.warning(
            "only %d model(s) re-ran successfully; the progress curve needs enough "
            "points to support a trend claim",
            counts.get("ok", 0),
        )
    return path
