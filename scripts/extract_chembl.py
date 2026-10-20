"""Run the ChEMBL extraction query and write the pre-canonical table.

Phase 1's second half. Takes the SQLite release downloaded by
``scripts/download_chembl.py``, runs ``sql/chembl_activities.sql``, assigns target
families, and writes a parquet file that ``leakfree curate`` can consume.

    python scripts/extract_chembl.py --tarball D:/chembl_dl/chembl_37_sqlite.tar.gz
    python scripts/extract_chembl.py --db D:/chembl_dl/chembl_37.db
    python scripts/extract_chembl.py --db ... --families kinase gpcr --limit 50000

Given a ``--tarball`` the script unpacks it, runs the query, and deletes the
unpacked database again. That is the default because the database is about 30 GB
against roughly 50 GB free on a typical laptop, and the cleanup is the step most
easily forgotten; ``--keep-db`` opts out.

The query result is streamed in chunks rather than loaded whole: the three-family
extract runs to roughly a million rows, which is fine in memory but needless to
hold there, and streaming keeps the peak low enough to run alongside the rest of
the pipeline.

The release number is read from the database and written into the output's
metadata sidecar. "ChEMBL" without a version is not a reproducible input -- the
same query against next year's release returns different data -- so the extract
records which release produced it.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.io import write_json, write_table

logger = get_logger(__name__)

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "chembl_activities.sql"
CHUNK_ROWS = 200_000


def unpack_release(tarball: Path) -> tuple[Path, Path]:
    """Unpack the SQLite release and return ``(database, unpacked root)``.

    Checks free space first. The archive is about 5.8 GB and unpacks to roughly
    30 GB, so on a disk with 50-odd GB free there is no margin for discovering
    the problem halfway through: a part-written database fails the query in a way
    that looks like a schema problem rather than a full disk.
    """
    if not tarball.is_file():
        raise SystemExit(f"no archive at {tarball}")

    needed = 32 * 1024**3
    free = shutil.disk_usage(tarball.parent).free
    if free < needed:
        raise SystemExit(
            f"{free / 1e9:.0f} GB free at {tarball.parent}, and unpacking needs about "
            f"{needed / 1e9:.0f} GB. Free some space or unpack elsewhere."
        )

    destination = tarball.parent / tarball.name.replace(".tar.gz", "")
    if destination.exists():
        logger.info("reusing the existing unpacked tree at %s", destination)
    else:
        logger.info("unpacking %s (this takes several minutes)", tarball.name)
        with tarfile.open(tarball, "r:gz") as archive:
            # filter="data" refuses absolute paths and traversal entries. The
            # archive is from EBI, but it is still downloaded input.
            archive.extractall(destination, filter="data")

    candidates = sorted(destination.rglob("*.db"))
    if not candidates:
        raise SystemExit(f"no .db file found under {destination}")
    database = max(candidates, key=lambda path: path.stat().st_size)
    logger.info("using %s (%.1f GB)", database.name, database.stat().st_size / 1e9)
    return database, destination


def release_version(connection: sqlite3.Connection) -> str:
    """The release name recorded inside the database itself."""
    try:
        row = connection.execute(
            "SELECT name FROM version ORDER BY creation_date DESC LIMIT 1"
        ).fetchone()
        if row:
            return str(row[0])
    except sqlite3.Error as exc:
        logger.warning("could not read the version table: %s", exc)
    return "unknown"


def load_query() -> str:
    """Read the committed SQL, stripping the sqlite3 dot-commands.

    The file is written to be runnable by hand with ``sqlite3 < file``, so it
    carries ``.headers`` and ``.mode`` lines that the Python driver rejects.
    """
    text = SQL_FILE.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.strip().startswith(".")]
    return "\n".join(lines).strip().rstrip(";")


def run_extraction(db_path: Path, limit: int | None) -> tuple[pd.DataFrame, dict[str, object]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    # The joins are heavy; a large page cache makes a substantial difference and
    # costs nothing on a machine that is only running this.
    connection.execute("PRAGMA cache_size = -2000000")  # ~2 GB

    version = release_version(connection)
    logger.info("database reports release %s", version)

    query = load_query()
    if limit:
        query = f"{query}\nLIMIT {int(limit)}"

    logger.info("running the extraction query (this takes several minutes)")
    started = time.monotonic()
    chunks: list[pd.DataFrame] = []
    n_rows = 0
    for chunk in pd.read_sql_query(query, connection, chunksize=CHUNK_ROWS):
        chunks.append(chunk)
        n_rows += len(chunk)
        logger.info("  %s rows", f"{n_rows:,}")
    connection.close()

    if not chunks:
        raise SystemExit(
            "the query returned no rows. Check that this is a full ChEMBL release "
            "and not a partial extract."
        )

    frame = pd.concat(chunks, ignore_index=True)
    elapsed = time.monotonic() - started
    logger.info("extracted %s rows in %.1f min", f"{len(frame):,}", elapsed / 60)

    metadata: dict[str, object] = {
        "chembl_release": version,
        "database": str(db_path),
        "query_file": str(SQL_FILE.relative_to(paths.ROOT)),
        "n_rows": len(frame),
        "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
    }
    return frame, metadata


def assign_families(frame: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    """Add target_family, then restrict to the families in scope."""
    from leakfree_dta.prot.families import assign_families as assign
    from leakfree_dta.prot.families import load_overrides, scope_to_families

    overrides = load_overrides(str(paths.CONFIGS / "data" / "target_families.csv"))

    targets = frame.drop_duplicates(subset=["target_uniprot"])[
        ["target_uniprot", "target_protein_class", "target_sequence"]
    ].rename(columns={"target_uniprot": "target_key"})

    assigned = assign(
        targets,
        key_column="target_key",
        class_column="target_protein_class",
        keyword_column=None,
        sequence_column="target_sequence",
        overrides=overrides,
    )

    out = frame.merge(
        assigned[["target_key", "target_family", "family_rule", "family_confidence"]],
        left_on="target_uniprot",
        right_on="target_key",
        how="left",
    ).drop(columns=["target_key"])

    before = len(out)
    out = scope_to_families(out, families)
    logger.info(
        "scoped to %s: %s of %s rows, %s targets",
        list(families),
        f"{len(out):,}",
        f"{before:,}",
        f"{out['target_uniprot'].nunique():,}",
    )
    return out


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_activities": len(frame),
        "n_targets": int(frame["target_uniprot"].nunique()),
        "n_compounds": int(frame["molecule_chembl_id"].nunique()),
        "n_assays": int(frame["assay_id"].nunique()),
        "n_documents": int(frame["document_chembl_id"].nunique())
        if "document_chembl_id" in frame
        else None,
    }
    for column in ("target_family", "standard_type", "family_confidence"):
        if column in frame.columns:
            summary[f"by_{column}"] = {
                str(key): int(value) for key, value in frame[column].value_counts().items()
            }
    if "document_year" in frame.columns:
        years = pd.to_numeric(frame["document_year"], errors="coerce").dropna()
        if len(years):
            summary["year_min"] = int(years.min())
            summary["year_max"] = int(years.max())
            summary["n_without_year"] = int(frame["document_year"].isna().sum())
    if "pchembl_value" in frame.columns:
        summary["n_with_pchembl"] = int(frame["pchembl_value"].notna().sum())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", help="path to an already-extracted chembl_NN.db")
    source.add_argument(
        "--tarball",
        help="path to chembl_NN_sqlite.tar.gz; it is unpacked, queried, and the "
        "unpacked database deleted again unless --keep-db is given",
    )
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--families", nargs="*", default=["kinase", "gpcr", "protease"])
    parser.add_argument("--limit", type=int, default=None, help="cap rows, for a trial run")
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="with --tarball, keep the ~30 GB unpacked database afterwards",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    paths.ensure_dirs()

    unpacked_root: Path | None = None
    if args.tarball:
        db_path, unpacked_root = unpack_release(Path(args.tarball))
    else:
        db_path = Path(args.db)
    if not db_path.is_file():
        raise SystemExit(f"no database at {db_path}")

    try:
        frame, metadata = run_extraction(db_path, args.limit)
    finally:
        if unpacked_root is not None and not args.keep_db:
            # Deleted in a finally block on purpose. The unpacked database is
            # about 30 GB against roughly 50 GB free; leaving it behind after a
            # failed query is how the next run dies with no disk space and no
            # obvious cause.
            logger.info("removing the unpacked database (%s)", unpacked_root)
            shutil.rmtree(unpacked_root, ignore_errors=True)
    frame = assign_families(frame, tuple(args.families))

    if frame.empty:
        raise SystemExit(
            "no rows survived family scoping. Check that target_protein_class is "
            "populated, and add targets to configs/data/target_families.csv."
        )

    out_path = paths.RAW / f"{args.dataset}.parquet"
    write_table(frame, out_path)

    summary = summarise(frame)
    metadata["families"] = list(args.families)
    metadata["summary"] = summary
    write_json(metadata, paths.RAW / f"{args.dataset}.metadata.json")

    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  release        {metadata['chembl_release']}")
    print(f"  activities     {summary['n_activities']:,}")
    print(f"  targets        {summary['n_targets']:,}")
    print(f"  compounds      {summary['n_compounds']:,}")
    print(f"  years          {summary.get('year_min')}-{summary.get('year_max')}")
    print(f"  by family      {summary.get('by_target_family')}")
    print(f"  family source  {summary.get('by_family_confidence')}")
    print(f"\nnext:\n  leakfree curate --source chembl --input {out_path} --dataset {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
