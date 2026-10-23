"""Split the Boltz-2 subset into session-sized chunks, and track what is done.

Thirty-five GPU-hours is not a thing you can sit through. Kaggle allows twelve
hours per session and roughly thirty a week; Colab is shorter and less
predictable; a rented instance is billed by the hour and worth releasing
between sittings. So the run has to survive being stopped, and that is a
property of how the work is divided rather than something to improvise when a
session dies.

The 600 complexes are independent, so the division is easy — but dividing by
*count* is the wrong unit. Sequence length drives the cost, and the subset's
lengths run from under 300 residues to 1,306, so equal-sized chunks would differ
several-fold in runtime and the long ones would overrun the session that was
sized for the average. Chunks are therefore packed by *estimated minutes*, each
staying under a budget you give it.

Progress is read from the output directories rather than from a state file. A
chunk is done when every one of its complexes has an affinity prediction on
disk, so a session killed mid-chunk resumes at the right place, and a state
file can never disagree with reality.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.io import read_json, read_table, write_json

logger = get_logger(__name__)

BASE_MINUTES = 2.0
BASE_LENGTH = 400
LENGTH_EXPONENT = 1.5

#: Leave room in a session for setup, model download and the final write. A
#: chunk packed to the full session length is a chunk that does not finish.
SAFETY_FRACTION = 0.75


def estimate_minutes(length: int) -> float:
    return BASE_MINUTES * (length / BASE_LENGTH) ** LENGTH_EXPONENT


def pack(subset: pd.DataFrame, budget_minutes: float) -> list[pd.DataFrame]:
    """Greedy longest-first packing into chunks under ``budget_minutes``.

    Longest first because a long complex placed last would not fit anywhere and
    would need a chunk of its own anyway; placing it first lets short ones fill
    in around it. Any single complex exceeding the budget gets a chunk to
    itself, and the caller is warned rather than having it silently dropped.
    """
    work = subset.copy()
    work["_minutes"] = work[schema.TARGET_SEQUENCE].str.len().map(estimate_minutes)
    work = work.sort_values("_minutes", ascending=False).reset_index(drop=True)

    chunks: list[list[int]] = []
    loads: list[float] = []
    for index, row in work.iterrows():
        minutes = float(row["_minutes"])
        placed = False
        for position, load in enumerate(loads):
            if load + minutes <= budget_minutes:
                chunks[position].append(index)
                loads[position] += minutes
                placed = True
                break
        if not placed:
            chunks.append([index])
            loads.append(minutes)
            if minutes > budget_minutes:
                logger.warning(
                    "one complex is estimated at %.0f min, over the %.0f min budget; "
                    "it gets a chunk of its own and may still overrun",
                    minutes,
                    budget_minutes,
                )

    # Return chunks longest-first so the expensive sessions happen while there
    # is still appetite for them.
    order = sorted(range(len(chunks)), key=lambda i: -loads[i])
    return [work.loc[chunks[i]].drop(columns="_minutes") for i in order]


def chunk_progress(chunk_dir: Path, output_root: Path) -> tuple[int, int]:
    """How many of a chunk's complexes already have a prediction on disk."""
    wanted = {p.stem for p in chunk_dir.glob("*.yaml")}
    if not wanted:
        return 0, 0
    done = set()
    for item in output_root.rglob("affinity*.json"):
        stem = item.parent.name if item.parent.name != output_root.name else item.stem
        done.add(stem.replace("affinity_", ""))
    return len(wanted & done), len(wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--split", default="leakfree")
    parser.add_argument(
        "--session-hours",
        type=float,
        default=12.0,
        help="length of one session; Kaggle allows 12 (default)",
    )
    parser.add_argument("--status", action="store_true", help="report progress and exit")
    args = parser.parse_args()

    setup_logging("INFO")
    root = paths.EXTERNAL / args.dataset / args.split / "Boltz-2"
    manifest = read_json(root / "subset_manifest.json")
    subset = read_table(root / "subset.csv")
    chunk_root = root / "chunks"
    output_root = root / "outputs"

    if args.status:
        if not chunk_root.exists():
            raise SystemExit(f"no chunks at {chunk_root}; run without --status first")
        total_done = total_all = 0
        print(f"{'chunk':10s} {'done':>12s}  state")
        for chunk_dir in sorted(chunk_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            done, total = chunk_progress(chunk_dir, output_root)
            total_done += done
            total_all += total
            state = "complete" if done == total and total else ("partial" if done else "not started")
            print(f"{chunk_dir.name:10s} {done:5d} / {total:<4d}  {state}")
        print(f"\noverall {total_done} / {total_all} complexes "
              f"({100 * total_done / max(total_all, 1):.1f}%)")
        if total_done:
            print(f"\ningest what exists at any point:\n"
                  f"  python scripts/ingest_boltz2.py --predictions {output_root}")
        return 0

    budget = args.session_hours * 60 * SAFETY_FRACTION
    chunks = pack(subset, budget)

    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    inputs = root / "inputs"
    if not inputs.exists():
        raise SystemExit(f"no inputs at {inputs}; run scripts/build_boltz2_subset.py first")

    plan = []
    for number, chunk in enumerate(chunks, start=1):
        chunk_dir = chunk_root / f"chunk{number:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        minutes = 0.0
        for row_key in chunk[schema.ROW_KEY]:
            name = str(row_key).replace("||", "__").replace("/", "_").replace("\\", "_")
            source = inputs / f"{name}.yaml"
            if not source.exists():
                logger.warning("missing input for %s", row_key)
                continue
            shutil.copy2(source, chunk_dir / source.name)
        minutes = float(chunk[schema.TARGET_SEQUENCE].str.len().map(estimate_minutes).sum())
        plan.append({"chunk": chunk_dir.name, "n_complexes": len(chunk),
                     "estimated_minutes": round(minutes, 1),
                     "max_sequence_length": int(chunk[schema.TARGET_SEQUENCE].str.len().max())})

    write_json(
        {
            "subset_digest": manifest["subset_digest"],
            "session_hours": args.session_hours,
            "budget_minutes_per_chunk": round(budget, 1),
            "safety_fraction": SAFETY_FRACTION,
            "n_chunks": len(plan),
            "chunks": plan,
        },
        chunk_root / "_plan.json",
    )

    total = sum(item["estimated_minutes"] for item in plan)
    print(f"\n{len(plan)} chunks for {args.session_hours:.0f}-hour sessions "
          f"(packed to {budget / 60:.1f} h each, {SAFETY_FRACTION:.0%} of the session)")
    print(f"{'chunk':10s} {'complexes':>10s} {'est. hours':>11s} {'longest seq':>12s}")
    for item in plan:
        print(f"{item['chunk']:10s} {item['n_complexes']:10d} "
              f"{item['estimated_minutes'] / 60:11.1f} {item['max_sequence_length']:12d}")
    print(f"{'TOTAL':10s} {sum(i['n_complexes'] for i in plan):10d} {total / 60:11.1f}")
    print("\nRun one chunk per session:")
    print(f"  boltz predict {chunk_root}/chunk01 --out_dir {output_root} --use_msa_server")
    print("\nCheck progress at any time:")
    print("  python scripts/chunk_boltz2_subset.py --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
