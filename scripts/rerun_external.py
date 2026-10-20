"""Attempt to re-run the published models, and write the attempt log.

Phase 2's driver. Every model named in the pre-registered target list is tried;
successes land in ``results/runs`` alongside our own models so that
``leakfree evaluate`` compares them in one table and one bootstrap, and failures
land in the attempt log with a reason and a log tail.

    python scripts/rerun_external.py --dataset chembl_3fam --split leakfree
    python scripts/rerun_external.py --dataset synthetic --split ligand_cluster --models DeepDTA
    python scripts/rerun_external.py --dataset synthetic --split ligand_cluster --dry-run

The exit code is 0 whether or not individual models ran. A model failing to
reproduce is data, not a pipeline error, and making the script fail on it would
tempt whoever runs it next to quietly drop the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.external.adapters import ADAPTERS, available_adapters
from leakfree_dta.external.harness import rerun_model, write_attempt_log
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.splits.manifest import apply_manifest, load_manifest
from leakfree_dta.utils.io import read_table

logger = get_logger(__name__)


def load_folds(dataset: str, split: str):
    path = paths.PROCESSED / f"{dataset}.aggregated.parquet"
    if not path.exists():
        raise SystemExit(f"no curated dataset at {path}; run `leakfree curate` first")

    manifest_path = paths.SPLITS / dataset / f"{split}.json"
    if not manifest_path.exists():
        raise SystemExit(f"no split manifest at {manifest_path}; run `leakfree split` first")

    frame = read_table(path)
    assignment = apply_manifest(frame, load_manifest(manifest_path), allow_partial=True)
    return assignment.train(frame), assignment.valid(frame), assignment.test(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models", nargs="*", default=None, help="default: all of them")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be attempted, without cloning or running",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    paths.ensure_dirs()

    wanted = args.models or available_adapters()
    unknown = [name for name in wanted if name not in ADAPTERS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; available: {available_adapters()}")

    if args.dry_run:
        print(f"would attempt {len(wanted)} model(s) on {args.dataset}/{args.split}:\n")
        for name in wanted:
            adapter = ADAPTERS[name]
            print(f"  {adapter.name}")
            print(f"    repo        {adapter.repository}")
            print(f"    commit      {adapter.commit or '(default head, recorded after clone)'}")
            print(f"    entrypoint  {' '.join(adapter.entrypoint)}")
            for adaptation in adapter.adaptations:
                print(f"    adaptation  {adaptation}")
            print()
        return 0

    train, valid, test = load_folds(args.dataset, args.split)
    logger.info(
        "re-running %d model(s) on %s/%s (train=%d valid=%d test=%d)",
        len(wanted),
        args.dataset,
        args.split,
        len(train),
        len(valid),
        len(test),
    )

    outcomes = []
    for name in wanted:
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        outcome = rerun_model(
            ADAPTERS[name],
            train,
            valid,
            test,
            dataset=args.dataset,
            split=args.split,
            timeout=int(args.timeout_hours * 3600),
        )
        outcomes.append(outcome)
        if outcome.status == "ok":
            print(
                f"  ok: rmse={outcome.metrics.get('rmse', float('nan')):.4f} "
                f"pearson={outcome.metrics.get('pearson', float('nan')):.4f} "
                f"({outcome.elapsed_seconds / 60:.1f} min)"
            )
        else:
            print(f"  {outcome.status}: {outcome.reason[:200]}")

    path = write_attempt_log(outcomes, args.dataset, args.split)

    succeeded = [o for o in outcomes if o.status == "ok"]
    print(f"\n{'=' * 72}")
    print(f"{len(succeeded)}/{len(outcomes)} models re-ran successfully")
    print(f"attempt log: {path}")
    if len(succeeded) < len(outcomes):
        print(
            "\nFailures are a result, not an error. They belong in the paper's attempt\n"
            "log: the models hardest to reproduce are disproportionately the oldest, so\n"
            "dropping them would bias the re-measured trend upward."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
