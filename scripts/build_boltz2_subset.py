"""Freeze the Boltz-2 evaluation subset and write its inputs.

Boltz-2 predicts a structure per complex at minutes of GPU time each, so scoring
the full 4,903-row leakage-free test fold is out of reach here and the registry
records its coverage as a subset. This script defines that subset once, freezes
it with a content digest, and writes the per-complex inputs an external machine
needs. It does not run anything.

Two design decisions that change how the results may be read, both deliberate.

**The subset over-samples the rare families.** The leakage-free test fold is 85%
GPCR -- 4,159 rows against 586 kinase and 158 protease -- so a proportional
sample of six hundred would give about nineteen proteases, enough to report a
number and not enough to mean anything. The subset instead takes roughly 300
GPCR, 200 kinase and 100 protease. It is therefore **not representative of the
test fold**, and an aggregate metric computed over it is not comparable to the
aggregate metrics of the other models. Per-family numbers are comparable;
anything pooled must be reweighted to the fold's family proportions, and
``weight_to_fold`` in the manifest carries the factor for doing that.

**No sequence-length cap.** Five targets exceed 1,000 residues and are the most
expensive complexes in the set. Excluding them in advance would quietly bias the
subset towards short proteins, and length correlates with family here -- capping
at 1,000 residues would drop 38% of the kinase rows. They stay in; if the
external hardware cannot fit one, that is an attempt-log entry with a stated
reason, which is the same standard applied to every other model in this paper.

Usage:

    python scripts/build_boltz2_subset.py
    python scripts/build_boltz2_subset.py --n-gpcr 150 --n-kinase 100 --n-protease 50
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.splits.manifest import apply_manifest, load_manifest
from leakfree_dta.utils.hashing import digest_frame_keys, hash_unit
from leakfree_dta.utils.io import read_table, write_json, write_table

logger = get_logger(__name__)

SUBSET_VERSION = 1

#: Minutes of GPU time for a complex of ~400 residues, on one modern
#: accelerator. An order of magnitude, not a promise.
BASE_MINUTES = 2.0

#: Reference length the base figure applies to.
BASE_LENGTH = 400

#: Boltz-2's structure module has attention over residues, so cost grows faster
#: than linearly with sequence length. A quadratic term is the usual
#: approximation and is used here only to keep the estimate from being
#: optimistic: the subset over-samples kinases, which are the long targets, so
#: 12% of its complexes exceed 1,000 residues against about 5% of the fold.
LENGTH_EXPONENT = 1.5


def stratified_sample(
    frame: pd.DataFrame, n_wanted: int, *, salt: str, n_affinity_bins: int = 3
) -> pd.DataFrame:
    """Pick ``n_wanted`` rows spread across protein clusters and affinity range.

    Selection is by hash rather than by a random generator, so the subset is a
    function of the data and the salt alone: it can be rebuilt anywhere, and
    adding rows to the benchmark does not reshuffle the ones already chosen.

    Rows are bucketed by (protein group, affinity tertile) and taken round-robin
    across buckets, so a single heavily-measured target cannot dominate and the
    affinity range stays covered.
    """
    if len(frame) <= n_wanted:
        return frame.copy()

    group_column = "protein_group" if "protein_group" in frame.columns else schema.TARGET_KEY
    work = frame.copy()
    work["_affinity_bin"] = pd.qcut(
        work[schema.Y].rank(method="first"), n_affinity_bins, labels=False
    )
    work["_draw"] = [hash_unit(key, salt=salt) for key in work[schema.ROW_KEY]]

    buckets: dict[tuple, list] = defaultdict(list)
    for _, row in work.sort_values("_draw").iterrows():
        buckets[(row[group_column], row["_affinity_bin"])].append(row[schema.ROW_KEY])

    # Round-robin over buckets, in a deterministic bucket order.
    order = sorted(buckets, key=lambda key: hash_unit(str(key), salt=f"{salt}|bucket"))
    chosen: list[str] = []
    depth = 0
    while len(chosen) < n_wanted:
        added = False
        for key in order:
            if depth < len(buckets[key]):
                chosen.append(buckets[key][depth])
                added = True
                if len(chosen) == n_wanted:
                    break
        if not added:
            break
        depth += 1

    return frame[frame[schema.ROW_KEY].isin(chosen)].copy()


def write_boltz_inputs(subset: pd.DataFrame, directory: Path) -> int:
    """One YAML per complex, in the layout Boltz-2 reads for affinity prediction.

    Files are named by row key so a prediction can be traced back to the exact
    benchmark row without a lookup table, which is the mistake that makes
    external results impossible to align later.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for row in subset.itertuples(index=False):
        row_key = getattr(row, schema.ROW_KEY)
        sequence = str(getattr(row, schema.TARGET_SEQUENCE))
        smiles = str(getattr(row, schema.SMILES))
        # Single quotes around SMILES: they routinely contain characters YAML
        # would otherwise interpret, and a silently mangled ligand would be
        # scored as if it were the real one.
        (directory / f"{_safe_name(row_key)}.yaml").write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {sequence}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{smiles}'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n",
            encoding="utf-8",
        )
    return len(subset)


def _estimate_minutes(lengths: pd.Series) -> pd.Series:
    """Per-complex GPU minutes, scaled by sequence length.

    Reported alongside the flat estimate so the gap between them is visible: on
    this subset the length-scaled figure is the larger one, and the difference
    is entirely the long kinase targets.
    """
    return BASE_MINUTES * (lengths.astype(float) / BASE_LENGTH) ** LENGTH_EXPONENT


def _safe_name(row_key: str) -> str:
    """A filename-safe form of a row key, reversible through the manifest."""
    return row_key.replace("||", "__").replace("/", "_").replace("\\", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--split", default="leakfree")
    parser.add_argument("--n-gpcr", type=int, default=300)
    parser.add_argument("--n-kinase", type=int, default=200)
    parser.add_argument("--n-protease", type=int, default=100)
    parser.add_argument("--salt", default="boltz2-subset-v1")
    parser.add_argument(
        "--no-inputs", action="store_true", help="write the manifest but not the YAML files"
    )
    args = parser.parse_args()

    setup_logging("INFO")
    paths.ensure_dirs()

    frame = read_table(paths.PROCESSED / f"{args.dataset}.aggregated.parquet")
    manifest = load_manifest(paths.SPLITS / args.dataset / f"{args.split}.json")
    assignment = apply_manifest(frame, manifest, allow_partial=True)
    test = assignment.test(frame)
    logger.info("test fold: %d rows, %d targets", len(test), test[schema.TARGET_KEY].nunique())

    wanted = {"gpcr": args.n_gpcr, "kinase": args.n_kinase, "protease": args.n_protease}
    parts = []
    for family, n_wanted in wanted.items():
        available = test[test[schema.TARGET_FAMILY] == family]
        if available.empty:
            logger.warning("family %s has no test rows; skipping", family)
            continue
        picked = stratified_sample(available, n_wanted, salt=f"{args.salt}|{family}")
        logger.info(
            "%s: %d of %d rows, %d targets",
            family,
            len(picked),
            len(available),
            picked[schema.TARGET_KEY].nunique(),
        )
        parts.append(picked)

    subset = pd.concat(parts, ignore_index=True).sort_values(schema.ROW_KEY).reset_index(drop=True)

    # The factor that converts a per-family subset metric back to the fold's
    # composition. Without it an aggregate over this subset silently reweights
    # the benchmark towards its rare families.
    fold_share = test[schema.TARGET_FAMILY].value_counts(normalize=True)
    subset_share = subset[schema.TARGET_FAMILY].value_counts(normalize=True)
    weight = {
        family: round(float(fold_share[family] / subset_share[family]), 5)
        for family in subset_share.index
    }

    lengths = subset[schema.TARGET_SEQUENCE].str.len()
    out_dir = paths.EXTERNAL / args.dataset / args.split / "Boltz-2"
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "subset_version": SUBSET_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "salt": args.salt,
        "n_rows": len(subset),
        "n_targets": int(subset[schema.TARGET_KEY].nunique()),
        "n_ligands": int(subset[schema.LIGAND_KEY].nunique()),
        "by_family": {k: int(v) for k, v in subset[schema.TARGET_FAMILY].value_counts().items()},
        "fold_by_family": {k: int(v) for k, v in test[schema.TARGET_FAMILY].value_counts().items()},
        "weight_to_fold": weight,
        "representative_of_fold": False,
        "sequence_length": {
            "median": int(lengths.median()),
            "max": int(lengths.max()),
            "n_over_1000": int((lengths > 1000).sum()),
        },
        "estimated_gpu_hours": round(float(_estimate_minutes(lengths).sum() / 60), 1),
        "estimated_gpu_hours_flat": round(len(subset) * BASE_MINUTES / 60, 1),
        "subset_digest": digest_frame_keys(subset[schema.ROW_KEY].tolist()),
        "parent_split_digest": manifest.get("dataset_digest"),
        "row_keys": sorted(subset[schema.ROW_KEY].tolist()),
    }
    write_json(record, out_dir / "subset_manifest.json")
    write_table(
        subset[
            [
                schema.ROW_KEY,
                schema.LIGAND_KEY,
                schema.SMILES,
                schema.TARGET_KEY,
                schema.TARGET_SEQUENCE,
                schema.TARGET_FAMILY,
                schema.Y,
            ]
        ],
        out_dir / "subset.csv",
    )

    if not args.no_inputs:
        n = write_boltz_inputs(subset, out_dir / "inputs")
        logger.info("wrote %d Boltz-2 YAML inputs", n)

    print(f"\nfrozen Boltz-2 subset -> {out_dir}")
    print(f"  rows            {record['n_rows']:,}  ({record['by_family']})")
    print(f"  targets         {record['n_targets']}   ligands {record['n_ligands']}")
    print(f"  fold composition{record['fold_by_family']}")
    print(f"  weight to fold  {record['weight_to_fold']}")
    print(f"  sequence length median {record['sequence_length']['median']}, "
          f"max {record['sequence_length']['max']}, "
          f"{record['sequence_length']['n_over_1000']} over 1000 aa")
    print(f"  estimated       ~{record['estimated_gpu_hours']} GPU-hours "
          f"(length-scaled; {record['estimated_gpu_hours_flat']} if every complex "
          f"cost the {BASE_MINUTES}-min base)")
    print(f"  digest          {record['subset_digest'][:16]}...")
    print("\nThis subset over-samples the rare families and is NOT representative of the")
    print("test fold. Report per-family metrics, or reweight with weight_to_fold.")
    print(f"\nNext: see {out_dir / 'README.md'} for the external run procedure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
