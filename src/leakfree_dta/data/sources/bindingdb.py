"""BindingDB: the cross-source consistency check.

BindingDB and ChEMBL overlap heavily -- BindingDB imports from ChEMBL -- yet the
same measurement often differs between them after each database's own curation.
That disagreement is a finding in its own right, and it is the reason this source
is loaded even though it adds few genuinely new measurements: a pair whose value
differs between two curated databases cannot be predicted to better than that
difference by anyone.

Expected input
--------------
``BindingDB_All.tsv`` from bindingdb.org/rwd/bind/downloads, or any subset of it.
The file is large (several GB uncompressed) and tab-separated with over 190
columns; only the ones below are read, via ``usecols``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

SOURCE_NAME = "bindingdb"

#: BindingDB's column names, which contain spaces and parentheses.
COLUMNS = {
    "smiles": "Ligand SMILES",
    "inchikey": "Ligand InChI Key",
    "uniprot": "UniProt (SwissProt) Primary ID of Target Chain",
    "sequence": "BindingDB Target Chain Sequence",
    "ki": "Ki (nM)",
    "kd": "Kd (nM)",
    "ic50": "IC50 (nM)",
    "ec50": "EC50 (nM)",
    "year": "Article DOI",  # replaced below when a year column is present
    "pmid": "PMID",
    "assay": "BindingDB Reactant_set_id",
    "n_chains": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "curation_source": "Curation/DataSource",
    "doi": "Article DOI",
}

#: Affinity columns in order of preference. Ki and Kd are thermodynamic constants;
#: IC50 and EC50 depend on assay conditions, so they are used only as a fallback.
AFFINITY_PREFERENCE = (("ki", "pKi"), ("kd", "pKd"), ("ic50", "pIC50"), ("ec50", "pEC50"))


def load(path: str | Path, *, chunksize: int = 250_000) -> pd.DataFrame:
    """Stream ``BindingDB_All.tsv`` and return pre-canonical rows.

    Read in chunks because the full file does not fit comfortably in memory on a
    laptop, and the pipeline is meant to be runnable on one.
    """
    path = Path(path)
    header = pd.read_csv(path, sep="\t", nrows=0, low_memory=False)

    # Column aliases: some BindingDB releases append ' 1' for single-chain targets
    actual_columns = dict(COLUMNS)
    if (
        "UniProt (SwissProt) Primary ID of Target Chain" not in header.columns
        and "UniProt (SwissProt) Primary ID of Target Chain 1" in header.columns
    ):
        actual_columns["uniprot"] = "UniProt (SwissProt) Primary ID of Target Chain 1"
    if (
        "BindingDB Target Chain Sequence" not in header.columns
        and "BindingDB Target Chain Sequence 1" in header.columns
    ):
        actual_columns["sequence"] = "BindingDB Target Chain Sequence 1"

    wanted = [
        actual_columns["smiles"],
        actual_columns["inchikey"],
        actual_columns["uniprot"],
        actual_columns["sequence"],
        actual_columns["ki"],
        actual_columns["kd"],
        actual_columns["ic50"],
        actual_columns["ec50"],
        actual_columns["pmid"],
        actual_columns["assay"],
        actual_columns["n_chains"],
        actual_columns["curation_source"],
        actual_columns["doi"],
    ]

    usecols = [column for column in wanted if column in header.columns]
    missing = set(wanted) - set(usecols)
    if missing:
        logger.warning("BindingDB file lacks columns %s; they will be absent", sorted(missing))
    if actual_columns["smiles"] not in usecols or actual_columns["uniprot"] not in usecols:
        raise ValueError(
            f"{path.name} lacks the ligand SMILES or target accession column; "
            "this does not look like a BindingDB export"
        )

    frames: list[pd.DataFrame] = []
    n_raw = 0
    for chunk in pd.read_csv(
        path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False, on_bad_lines="skip"
    ):
        n_raw += len(chunk)
        frames.append(_convert_chunk(chunk, col_map=actual_columns))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("BindingDB: %d raw rows -> %d usable measurements", n_raw, len(out))
    return out


def _convert_chunk(chunk: pd.DataFrame, col_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Melt BindingDB's four affinity columns into one, preferring Ki and Kd."""
    cols = col_map or COLUMNS
    values = pd.Series(np.nan, index=chunk.index, dtype="float64")
    kinds = pd.Series(pd.NA, index=chunk.index, dtype="object")
    relations = pd.Series("=", index=chunk.index, dtype="object")

    for key, p_type in AFFINITY_PREFERENCE:
        column = cols[key]
        if column not in chunk.columns:
            continue
        raw = chunk[column].astype(str).str.strip()
        # BindingDB encodes censored values in the cell itself, e.g. ">10000".
        relation = np.where(raw.str.startswith(">"), ">", np.where(raw.str.startswith("<"), "<", "="))
        numeric = pd.to_numeric(raw.str.lstrip("><="), errors="coerce")
        # nM to the p-scale.
        p_value = -np.log10(numeric.where(numeric > 0) * 1e-9)

        fill = values.isna() & p_value.notna()
        values[fill] = p_value[fill]
        kinds[fill] = p_type
        relations[fill] = pd.Series(relation, index=chunk.index)[fill]

    out = pd.DataFrame(
        {
            schema.SMILES: chunk[cols["smiles"]].astype(str),
            schema.TARGET_KEY: chunk[cols["uniprot"]].astype(str).str.strip().str.upper(),
            schema.TARGET_SEQUENCE: chunk.get(cols["sequence"]),
            schema.Y: values,
            schema.Y_TYPE: kinds,
            schema.Y_RELATION: relations,
            schema.SOURCE: SOURCE_NAME,
            schema.ASSAY_KEY: chunk.get(cols["assay"]),
        }
    )
    if cols.get("inchikey") in chunk.columns:
        keys = chunk[cols["inchikey"]].astype("string").str.strip().str.upper()
        # Avoid turning a missing value into the literal key "NAN".  The regex
        # accepts standard 27-character InChIKeys only.
        out[schema.LIGAND_KEY] = keys.where(keys.str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", na=False))
    if cols["pmid"] in chunk.columns:
        out["pmid"] = chunk[cols["pmid"]]
    if cols.get("curation_source") in chunk.columns:
        out["curation_source"] = chunk[cols["curation_source"]]
    if cols.get("doi") in chunk.columns:
        out["doi"] = chunk[cols["doi"]]
    if cols.get("n_chains") in chunk.columns:
        out["n_target_chains"] = pd.to_numeric(chunk[cols["n_chains"]], errors="coerce")
        # The ChEMBL benchmark is restricted to single-protein targets.  A value
        # missing in older exports is retained; an explicitly multi-chain target
        # is not comparable and is removed.
        out = out[out["n_target_chains"].isna() | out["n_target_chains"].eq(1)]
    return out[out[schema.Y].notna() & out[schema.TARGET_KEY].notna()].reset_index(drop=True)


def cross_source_disagreement(chembl: pd.DataFrame, bindingdb: pd.DataFrame) -> pd.DataFrame:
    """Pairs present in both sources, with the difference between their values.

    Produces the paper's cross-source consistency table. A pair whose two curated
    databases disagree by a log unit sets a floor on achievable accuracy that no
    amount of modelling can get under, and the size of that floor is a number the
    field has not published.
    """
    # IC50, Ki, Kd and EC50 are not interchangeable labels.  Matching only by
    # ligand and target can manufacture disagreement simply by comparing two
    # different endpoints, so endpoint identity is part of the cross-source key.
    keys = [schema.PAIR_KEY, schema.Y_TYPE]
    left = chembl.groupby(keys, observed=True)[schema.Y].agg(
        chembl_y="median", chembl_n="size"
    )
    right = bindingdb.groupby(keys, observed=True)[schema.Y].agg(
        bindingdb_y="median", bindingdb_n="size"
    )
    joined = left.join(right, how="inner")
    joined["abs_delta"] = (joined["chembl_y"] - joined["bindingdb_y"]).abs()

    if len(joined):
        logger.info(
            "cross-source: %d shared pairs, median |delta| = %.3f, %.1f%% disagree by > 1 log unit",
            len(joined),
            float(joined["abs_delta"].median()),
            100 * float((joined["abs_delta"] > 1.0).mean()),
        )
    return joined.sort_values("abs_delta", ascending=False).reset_index()
