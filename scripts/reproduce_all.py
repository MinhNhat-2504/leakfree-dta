"""Reproduce every number in the paper, in order.

One command, so that "we reran the pipeline" is a checkable claim rather than a
statement about what someone remembers doing. Each stage is skipped when its output
already exists, which makes the script safe to re-run after an interruption --
re-running the slow stages unnecessarily is how a pipeline stops being run at all.

    python scripts/reproduce_all.py --dataset chembl_3fam
    python scripts/reproduce_all.py --dataset synthetic --synthetic
    python scripts/reproduce_all.py --dataset chembl_3fam --force   # ignore caches
"""

# ruff: noqa: SIM102 - the nested ifs separate "already done" from "succeeded",
# which is the structure this script is organised around.

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.splits.three_axis import SPLIT_LADDER

#: The models run on every split. Deep models come last so that a GPU failure
#: leaves the classical results intact rather than aborting before them.
MODELS = (
    "global_mean",
    "target_mean",
    "ligand_mean",
    "additive_bias",
    "ridge_ecfp",
    "xgboost_ecfp",
    "erm_mlp",
    "group_dro_mlp",
    "shift_robust",
)


def run(args: list[str], *, label: str) -> bool:
    """Run one CLI stage, timed, and report rather than raise on failure.

    A single model failing to train must not abort the run: the other models'
    results are still valid, and the paper reports the failure as a missing row.
    """
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    command = [sys.executable, "-m", "leakfree_dta.cli", *args]
    start = time.monotonic()
    result = subprocess.run(command, check=False)
    elapsed = time.monotonic() - start

    if result.returncode == 0:
        print(f"\n[ok] {label} ({elapsed:.1f}s)")
        return True
    # The audit exits non-zero when a check fails its tolerance, which is a
    # finding, not an error. Everything else is a real failure.
    if "audit" in args and result.returncode == 1:
        print(f"\n[findings] {label} reported leakage outside tolerance ({elapsed:.1f}s)")
        return True
    print(f"\n[FAILED] {label}: exit {result.returncode} ({elapsed:.1f}s)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="chembl")
    parser.add_argument("--input", default=None, help="raw file; required unless --synthetic")
    parser.add_argument("--synthetic", action="store_true", help="use the calibration fixture")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    paths.ensure_dirs()
    failures: list[str] = []
    started = time.monotonic()

    # --- stage 1: the dataset -----------------------------------------------
    processed = paths.PROCESSED / f"{args.dataset}.aggregated.parquet"
    if args.synthetic:
        if args.force or not processed.exists():
            if not run(["demo", "--dataset", args.dataset, "--fast"], label="synthetic dataset + demo"):
                failures.append("demo")
        else:
            print(f"[skip] {processed.name} exists")
    else:
        if not args.input:
            parser.error("--input is required unless --synthetic is given")
        if args.force or not processed.exists():
            if not run(
                ["curate", "--source", args.source, "--input", args.input, "--dataset", args.dataset],
                label=f"curate {args.source}",
            ):
                return 1  # nothing downstream can run without a dataset
        else:
            print(f"[skip] {processed.name} exists")

    # --- stage 2: audit the data itself -------------------------------------
    if args.force or not (paths.AUDIT / args.dataset / "dataset" / "report.json").exists():
        if not run(["audit", "--dataset", args.dataset], label="audit dataset"):
            failures.append("audit dataset")

    # --- stage 3: splits ----------------------------------------------------
    split_dir = paths.SPLITS / args.dataset
    if args.force or not list(split_dir.glob("*.json")):
        if not run(["split", "--dataset", args.dataset, "--ladder"], label="build split ladder"):
            return 1
    else:
        print(f"[skip] {len(list(split_dir.glob('*.json')))} split manifests exist")

    available = sorted(path.stem for path in split_dir.glob("*.json"))
    wanted = args.splits or [spec.name for spec in SPLIT_LADDER]
    splits = [name for name in wanted if name in available]
    if not splits:
        print(f"[FAILED] none of the requested splits were built; available: {available}")
        return 1
    print(f"\nsplits to process: {splits}")

    # --- stage 4: audit each split ------------------------------------------
    for split in splits:
        if args.force or not (paths.AUDIT / args.dataset / split / "overall" / "report.json").exists():
            if not run(
                ["audit", "--dataset", args.dataset, "--split", split],
                label=f"audit split {split}",
            ):
                failures.append(f"audit {split}")

    # --- stage 5: train -----------------------------------------------------
    for split in splits:
        for model in args.models:
            run_dir = paths.RUNS / args.dataset / split / model
            if not args.force and (run_dir / "metrics.json").exists():
                print(f"[skip] {split}/{model} already trained")
                continue
            if not run(
                ["train", "--dataset", args.dataset, "--split", split, "--model", model],
                label=f"train {model} on {split}",
            ):
                failures.append(f"train {model}/{split}")

    # --- stage 6: evaluate --------------------------------------------------
    for split in splits:
        if not run(
            [
                "evaluate",
                "--dataset",
                args.dataset,
                "--split",
                split,
                "--n-bootstrap",
                str(args.n_bootstrap),
            ],
            label=f"evaluate {split}",
        ):
            failures.append(f"evaluate {split}")

    # --- stage 7: report ----------------------------------------------------
    if not run(["report", "--dataset", args.dataset], label="assemble report"):
        failures.append("report")

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 72}")
    print(f"finished in {elapsed / 60:.1f} min")
    if failures:
        print(f"\n{len(failures)} stage(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nFailures are reported rather than hidden: a model that would not run "
            "appears as a missing row in the results, with the reason recorded."
        )
    else:
        print("\nevery stage completed")
    print(f"\nartifacts: {paths.TABLES / args.dataset}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
