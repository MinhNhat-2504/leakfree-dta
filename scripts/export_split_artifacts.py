"""Write the two split artifacts: a small verifiable index, and the full assignment.

The split manifests this project generates hold every row key per fold, which
makes them 42 MB each and 385 MB for the ladder. They cannot go in git, and the
repository was claiming to release something it did not contain -- the negation
in `.gitignore` never fired, because git does not descend into a directory its
parent pattern already excluded.

Rather than paper over that, the artifacts are split by purpose.

``data/splits/<dataset>/_index.json`` -- a few kilobytes, committed. Carries each
split's specification, fold counts, per-fold content digests and the dataset
digest. That is enough for a reader to *verify* a split they have rebuilt, and
enough for a reviewer to see exactly how each rung was defined, without any row
keys.

``results/splits/<dataset>_assignment.parquet`` -- one table, all rungs as
columns, dictionary-encoded. This is what someone applies without rerunning the
two-hour clustering, and it goes in the archival deposit rather than in git.

A reader with neither can still regenerate both: splits are a deterministic
function of the curated dataset and the committed specification, which is what
the hash-based assignment is for. The index lets them confirm they got the same
answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.splits.base import FOLDS
from leakfree_dta.splits.manifest import load_manifest
from leakfree_dta.utils.io import read_table, write_json, write_table

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    args = parser.parse_args()

    setup_logging("INFO")
    split_dir = paths.SPLITS / args.dataset
    manifests = sorted(split_dir.glob("*.json"))
    manifests = [p for p in manifests if not p.name.startswith("_")]
    if not manifests:
        raise SystemExit(f"no split manifests under {split_dir}")

    frame = read_table(paths.PROCESSED / f"{args.dataset}.aggregated.parquet")
    keys = frame[schema.ROW_KEY].astype(str)

    index: dict[str, object] = {"dataset": args.dataset, "splits": {}}
    assignment = pd.DataFrame({schema.ROW_KEY: keys})

    for path in manifests:
        payload = load_manifest(path)
        name = payload["spec"]["name"]

        index["splits"][name] = {
            "spec": payload["spec"],
            "spec_digest": payload.get("spec_digest"),
            "counts": payload.get("counts"),
            "fold_digests": payload.get("digests"),
            "notes": payload.get("notes"),
        }
        index.setdefault("dataset_digest", payload.get("dataset_digest"))

        lookup: dict[str, str] = {}
        for fold in FOLDS:
            for key in payload["row_keys"].get(fold, []):
                lookup[str(key)] = fold
        assignment[name] = keys.map(lookup).fillna("excluded").astype("category")
        logger.info("%s: %s", name, payload.get("counts"))

    write_json(index, split_dir / "_index.json")

    out = paths.RESULTS / "splits" / f"{args.dataset}_assignment.parquet"
    write_table(assignment, out)

    index_size = (split_dir / "_index.json").stat().st_size
    full_size = out.stat().st_size
    raw = sum(p.stat().st_size for p in manifests)

    print(f"\nindex      {split_dir / '_index.json'}")
    print(f"           {index_size / 1024:.1f} KB, {len(index['splits'])} splits -- commit this")
    print(f"assignment {out}")
    print(f"           {full_size / 1048576:.1f} MB for {len(assignment):,} rows "
          f"x {len(index['splits'])} splits -- archive this")
    print(f"           (the per-split JSON manifests are {raw / 1048576:.0f} MB and stay local)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
