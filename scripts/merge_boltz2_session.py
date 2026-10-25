"""Merge one downloaded Kaggle session into the accumulated Boltz-2 output.

Six sessions produce six downloads, each a zip called something like
"results (7).zip" depending on how many times the browser has seen that name.
Extracting the wrong one, or into the wrong directory, is a quiet failure: the
counts simply do not add up later, and by then it is not obvious which session
went missing.

So this finds the newest archive, refuses to proceed unless it actually adds
predictions, and reports the before and after counts. It never deletes: merging
is additive, and a session already merged is detected rather than duplicated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.logging import get_logger, setup_logging

logger = get_logger(__name__)


def newest_archive(folder: Path) -> Path:
    archives = sorted(folder.glob("results*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        raise SystemExit(f"no results*.zip in {folder}. Download the session output first.")
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--split", default="leakfree")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="the downloaded zip; defaults to the newest results*.zip in Downloads",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    downloads = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads"
    archive = args.archive or newest_archive(downloads)

    target = paths.EXTERNAL / args.dataset / args.split / "Boltz-2" / "outputs"
    affinity = target / "affinity"
    before = len(list(affinity.glob("affinity_*.json"))) if affinity.exists() else 0

    stat = archive.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    print(f"archive  {archive}")
    print(f"         {stat.st_size / 1024:.0f} KB, downloaded {modified}")
    print(f"target   {target}")
    print(f"before   {before} predictions")

    # Extract to a scratch directory first. A partly-extracted archive merged
    # straight into the accumulated output is hard to tell from a short session.
    with tempfile.TemporaryDirectory() as scratch:
        with zipfile.ZipFile(archive) as z:
            z.extractall(scratch)
        staged = Path(scratch)
        found = list(staged.rglob("affinity_*.json"))
        if not found:
            raise SystemExit(
                f"no affinity_*.json inside {archive.name}. Is this the right download?"
            )
        affinity.mkdir(parents=True, exist_ok=True)
        added = 0
        for item in found:
            destination = affinity / item.name
            if destination.exists():
                continue
            shutil.copy2(item, destination)
            added += 1
        for meta in staged.rglob("run_*.json"):
            shutil.copy2(meta, target / meta.name)

    after = len(list(affinity.glob("affinity_*.json")))
    print(f"in zip   {len(found)} predictions")
    print(f"added    {added} new")
    print(f"after    {after} predictions")

    if added == 0:
        print("\nNothing new. This session was already merged, or it is the same download.")
    print(f"\nnext:\n  python scripts/ingest_boltz2.py --predictions {target}\n"
          "  python scripts/compare_boltz2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
