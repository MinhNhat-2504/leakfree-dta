"""Assemble the archival deposit from what the pipeline wrote.

The Zenodo record is not the repository. The repository holds code and the small
verifiable index; the deposit holds the evidence that is too large for git -- the
full fold assignment, the audit reports, and per-row predictions for every model
that ran. Section 8 of the manuscript promises exactly those, so this script
exists to make the promise checkable rather than assembled by hand at midnight
before a submission.

Three things it does that a zip of `results/` would not:

*Bundles into archives.* Zenodo allows 100 files per record and `results/` holds
several hundred. The grouping is by what a reader would want separately: someone
reusing the splits should not download 450 MB of audit detail.

*Excludes the test harness's own output.* `results/runs/t/s/silent/` is a dummy
model invoked as `python -c pass` to test the external-model harness. It is not a
result, and a reader finding a model called "silent" in the archive would be
right to wonder what else is in there.

*Records digests.* Every archive is hashed and listed in MANIFEST.md, so a
download can be checked against what was deposited, and so the deposit can be
compared against a later rebuild.

Licensing: predictions carry ChEMBL-derived values (InChIKeys, UniProt
accessions, activity values). ChEMBL is CC BY-SA 3.0, which permits
redistribution with attribution under the same terms, and MANIFEST.md states
this. The code is MIT. A Zenodo record carries one license field, so the
distinction has to live in the description and the manifest rather than in the
metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.logging import get_logger, setup_logging

logger = get_logger(__name__)

#: Dummy models the external harness runs against itself. Not results.
EXCLUDED_RUN_DIRS = ("t",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(root: Path, excluded: tuple[str, ...] = ()) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        parts = item.relative_to(root).parts
        if parts and parts[0] in excluded:
            continue
        files.append(item)
    return files


def write_archive(target: Path, members: list[tuple[Path, str]]) -> tuple[int, int]:
    """Write a zip, returning (file count, compressed size)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, name in members:
            archive.write(source, name)
    return len(members), target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.RESULTS / "zenodo",
        help="where the archives are written (gitignored)",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    bundles: dict[str, list[tuple[Path, str]]] = {}

    index = paths.SPLITS / args.dataset / "_index.json"
    assignment = paths.RESULTS / "splits" / f"{args.dataset}_assignment.parquet"
    bundles["splits"] = [
        (p, str(p.relative_to(repo)).replace("\\", "/"))
        for p in (index, assignment)
        if p.exists()
    ]

    bundles["audit"] = [
        (p, str(p.relative_to(repo)).replace("\\", "/")) for p in collect(paths.RESULTS / "audit")
    ]
    bundles["predictions"] = [
        (p, str(p.relative_to(repo)).replace("\\", "/"))
        for p in collect(paths.RESULTS / "runs", excluded=EXCLUDED_RUN_DIRS)
    ]
    bundles["tables_figures"] = [
        (p, str(p.relative_to(repo)).replace("\\", "/"))
        for p in collect(paths.RESULTS / "tables") + collect(paths.RESULTS / "figures")
    ]

    missing = [name for name, members in bundles.items() if not members]
    if missing:
        raise SystemExit(
            f"nothing to archive for {missing}. Run the pipeline first, or "
            f"scripts/export_split_artifacts.py for the split index."
        )

    rows = []
    for name, members in bundles.items():
        target = out / f"{args.dataset}_{name}.zip"
        count, size = write_archive(target, members)
        digest = sha256(target)
        rows.append((target.name, count, size, digest))
        logger.info("%s: %d files, %.1f MB", target.name, count, size / 1048576)

    manifest = out / "MANIFEST.md"
    lines = [
        "# Archival deposit contents",
        "",
        f"Dataset `{args.dataset}`. Generated by `scripts/build_zenodo_bundle.py`.",
        "",
        "| archive | files | size | sha256 |",
        "| --- | --: | --: | --- |",
    ]
    for name, count, size, digest in rows:
        lines.append(f"| `{name}` | {count} | {size / 1048576:.1f} MB | `{digest}` |")
    lines += [
        "",
        "## What is in each",
        "",
        "- **splits** -- the split index (specifications, fold counts, per-fold",
        "  digests) and the full per-row fold assignment for every rung. Applying",
        "  a split needs only this; rebuilding it from source needs the",
        "  repository and a ChEMBL release number.",
        "- **audit** -- leakage reports per split and per target family, in JSON",
        "  (with per-row detail) and Markdown (summary).",
        "- **predictions** -- per-row predictions, metrics and run provenance for",
        "  every model evaluated. Excludes the external harness's own test model.",
        "- **tables_figures** -- the tables and figures the manuscript reproduces.",
        "",
        "## Licensing",
        "",
        "Code in the accompanying repository is MIT. The data files here are",
        "derived from ChEMBL, which is CC BY-SA 3.0: they may be redistributed",
        "with attribution under the same terms. Predictions carry InChIKeys,",
        "UniProt accessions and activity values derived from ChEMBL; no BindingDB",
        "records are redistributed, only aggregate concordance statistics.",
        "",
        "## Verifying a download",
        "",
        "```bash",
        "sha256sum *.zip   # compare against the table above",
        "```",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(size for _, _, size, _ in rows)
    print(f"\n{len(rows)} archives in {out}")
    print(f"{'archive':40s} {'files':>6s} {'size':>10s}")
    for name, count, size, _ in rows:
        print(f"{name:40s} {count:6d} {size / 1048576:9.1f} MB")
    print(f"{'TOTAL':40s} {sum(c for _, c, _, _ in rows):6d} {total / 1048576:9.1f} MB")
    print(f"\nmanifest: {manifest}")
    print(f"Upload these {len(rows)} archives plus MANIFEST.md to the Zenodo draft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
