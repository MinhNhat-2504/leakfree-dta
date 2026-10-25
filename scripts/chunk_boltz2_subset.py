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

#: Measured, not assumed. A smoke run of three complexes averaging 505 residues
#: on a Kaggle T4 took 20.1 minutes of Boltz compute: 13.9 for the structure
#: pass and 6.2 for the affinity pass. The first figure here was 2.0, derived
#: from the structure pass alone -- Boltz runs affinity as a second pass over
#: the same inputs, and leaving it out understated every chunk by well over a
#: factor of two.
#:
#: 20.1 min / 3 complexes = 6.7 min each at 505 aa, so
#: BASE_MINUTES = 6.7 / (505/400)**1.5 = 4.74.
BASE_MINUTES = 4.74
BASE_LENGTH = 400
LENGTH_EXPONENT = 1.5

#: Boltz already hit a CUDA OOM warning at ~500 residues on a 16 GB T4 and
#: recovered. Above this it is unlikely to, so the plan flags them rather than
#: letting a session discover it nine hours in.
OOM_RISK_LENGTH_16GB = 900

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
        # The filename carries the row key in both layouts Boltz output arrives
        # in; the parent directory only does in one of them. See
        # row_key_from_path in scripts/ingest_boltz2.py.
        name = item.stem
        done.add(name[len("affinity_") :] if name.startswith("affinity_") else item.parent.name)
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
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help=(
            "only chunk complexes at or below this length. For splitting a run "
            "across GPU tiers: a 16 GB card does not fit the long targets, and "
            "the excluded rows are reported rather than dropped silently."
        ),
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

    if args.max_sequence_length:
        lengths = subset[schema.TARGET_SEQUENCE].str.len()
        excluded = subset[lengths > args.max_sequence_length]
        subset = subset[lengths <= args.max_sequence_length]
        if len(excluded):
            # Length correlates with family here, so this is never a neutral
            # filter. Saying which families it costs is the point.
            by_family = excluded[schema.TARGET_FAMILY].value_counts()
            logger.warning(
                "excluding %d of %d complexes over %d aa; these are NOT dropped "
                "from the subset, they are left for a larger GPU",
                len(excluded),
                len(excluded) + len(subset),
                args.max_sequence_length,
            )
            for family, count in by_family.items():
                total = int((read_table(root / "subset.csv")[schema.TARGET_FAMILY] == family).sum())
                logger.warning("  %s: %d of %d (%.1f%%)", family, count, total, 100 * count / total)

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
