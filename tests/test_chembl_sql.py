"""The ChEMBL extraction query, run against a miniature ChEMBL.

The real query is tested here rather than only against the 5.8 GB release,
because a mistake in it is otherwise discovered after a multi-hour download. The
schema below mirrors the ChEMBL tables the query joins, with only the columns it
touches, and the rows are chosen so that every WHERE clause has something to both
keep and reject.

This caught a real bug: the query selected one level of the protein
classification hierarchy with a COALESCE over its ancestors, which returns the
most *generic* name -- "Enzyme" for a kinase. The family regexes match on
"kinase", "protease" and "7tm", so nearly every target would have been dropped
as unassigned and the extract would have come out close to empty.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

DDL = """
CREATE TABLE version (name TEXT, creation_date TEXT);
CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, year INTEGER, doi TEXT, pubmed_id INTEGER);
CREATE TABLE molecule_dictionary (molregno INTEGER PRIMARY KEY, chembl_id TEXT);
CREATE TABLE compound_structures (molregno INTEGER PRIMARY KEY, canonical_smiles TEXT);
CREATE TABLE component_sequences (component_id INTEGER PRIMARY KEY, accession TEXT, sequence TEXT);
CREATE TABLE target_dictionary (
    tid INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT,
    target_type TEXT, organism TEXT
);
CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
CREATE TABLE protein_classification (
    protein_class_id INTEGER PRIMARY KEY, pref_name TEXT, parent_id INTEGER
);
CREATE TABLE component_class (component_id INTEGER, protein_class_id INTEGER);
CREATE TABLE assays (
    assay_id INTEGER PRIMARY KEY, tid INTEGER, confidence_score INTEGER, assay_type TEXT
);
CREATE TABLE activities (
    activity_id INTEGER PRIMARY KEY, assay_id INTEGER, molregno INTEGER, doc_id INTEGER,
    standard_type TEXT, standard_relation TEXT, standard_value REAL, standard_units TEXT,
    pchembl_value REAL, activity_comment TEXT, data_validity_comment TEXT,
    potential_duplicate INTEGER
);
"""

KINASE_SEQ = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVF" * 4
GPCR_SEQ = "MDVLSPGQGNNTTSPPAPFETGGNTTGISDVTVSYQVITSLLLGTLIFCAVLGNACVVAAI" * 4
PROTEASE_SEQ = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAGHCYKSGIQVRLGEDNINVVEGN" * 4


def build(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(DDL)

    connection.execute("INSERT INTO version VALUES ('CHEMBL_37', '2026-01-15')")
    connection.executemany(
        "INSERT INTO docs VALUES (?,?,?,?)",
        [(1, 2015, "10.1/a", 111), (2, 2021, "10.1/b", 222), (3, None, None, None)],
    )

    # Three protein classes, nested so the COALESCE over parent levels is exercised.
    connection.executemany(
        "INSERT INTO protein_classification VALUES (?,?,?)",
        [
            (1, "Enzyme", None),
            (2, "Kinase", 1),
            (3, "Protein Kinase", 2),
            (4, "Membrane receptor", None),
            (5, "Family A G protein-coupled receptor", 4),
            (6, "Protease", 1),
            (7, "Serine protease", 6),
        ],
    )

    targets = [
        # tid, chembl, name, type, organism, component_id, accession, sequence, class
        (1, "CHEMBL203", "EGFR", "SINGLE PROTEIN", "Homo sapiens", 11, "P00533", KINASE_SEQ, 3),
        (2, "CHEMBL214", "5-HT1A", "SINGLE PROTEIN", "Homo sapiens", 12, "P08908", GPCR_SEQ, 5),
        (3, "CHEMBL204", "Thrombin", "SINGLE PROTEIN", "Homo sapiens", 13, "P00734", PROTEASE_SEQ, 7),
        # Rejected: not human.
        (4, "CHEMBL999", "Rat EGFR", "SINGLE PROTEIN", "Rattus norvegicus", 14, "Q01234", KINASE_SEQ, 3),
        # Rejected: not a single protein.
        (5, "CHEMBL888", "A complex", "PROTEIN COMPLEX", "Homo sapiens", 15, "P11111", KINASE_SEQ, 3),
        # Rejected: two components, so no single sequence.
        (6, "CHEMBL777", "Two parts", "SINGLE PROTEIN", "Homo sapiens", 16, "P22222", KINASE_SEQ, 3),
    ]
    for tid, chembl, name, kind, organism, component, accession, sequence, protein_class in targets:
        connection.execute(
            "INSERT INTO target_dictionary VALUES (?,?,?,?,?)", (tid, chembl, name, kind, organism)
        )
        connection.execute("INSERT INTO target_components VALUES (?,?)", (tid, component))
        connection.execute(
            "INSERT INTO component_sequences VALUES (?,?,?)", (component, accession, sequence)
        )
        connection.execute("INSERT INTO component_class VALUES (?,?)", (component, protein_class))
    # Give target 6 a second component so the single-component filter rejects it.
    connection.execute("INSERT INTO target_components VALUES (6, 99)")
    connection.execute("INSERT INTO component_sequences VALUES (99, 'P33333', 'MKV')")

    connection.executemany(
        "INSERT INTO assays VALUES (?,?,?,?)",
        [
            (101, 1, 9, "B"),
            (102, 2, 8, "B"),
            (103, 3, 9, "B"),
            (104, 1, 5, "B"),  # rejected: confidence below 8
            (105, 4, 9, "B"),
            (106, 5, 9, "B"),
            (107, 6, 9, "B"),
        ],
    )

    connection.executemany(
        "INSERT INTO molecule_dictionary VALUES (?,?)",
        [(i, f"CHEMBL{1000+i}") for i in range(1, 11)],
    )
    connection.executemany(
        "INSERT INTO compound_structures VALUES (?,?)",
        [(i, "CC(=O)Nc1ccc(O)cc1") for i in range(1, 10)] + [(10, None)],
    )

    rows = [
        # kept: three good activities, one per family
        (1, 101, 1, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),
        (2, 102, 2, 2, "Ki", "=", 10.0, "nM", 8.0, None, None, 0),
        (3, 103, 3, 1, "Kd", "=", 100.0, "nM", 7.0, None, None, 0),
        # kept: no pchembl, so the loader must convert standard_value
        (4, 101, 4, 2, "Ki", "=", 1000.0, "nM", None, None, None, 0),
        # kept: censored, dropped later in curation rather than in SQL
        (5, 101, 5, 1, "IC50", ">", 10000.0, "nM", None, None, None, 0),
        # rejected by SQL for the stated reasons
        (6, 104, 1, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),       # confidence 5
        (7, 101, 6, 1, "IC50", "=", 50.0, "uM", 5.3, None, None, 0),       # units not nM
        (8, 101, 7, 1, "Potency", "=", 50.0, "nM", 7.3, None, None, 0),    # type not wanted
        (9, 101, 8, 1, "IC50", "=", 0.0, "nM", None, None, None, 0),       # value not positive
        (10, 101, 9, 1, "IC50", "=", 50.0, "nM", 7.3, None, "Outside typical range", 0),
        (11, 101, 9, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 1),      # potential duplicate
        (12, 101, 10, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),     # no SMILES
        (13, 105, 1, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),      # rat
        (14, 106, 1, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),      # complex
        (15, 107, 1, 1, "IC50", "=", 50.0, "nM", 7.3, None, None, 0),      # two components
    ]
    connection.executemany("INSERT INTO activities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    connection.commit()
    connection.close()




REPO_ROOT = Path(__file__).resolve().parents[1]

#: Activities the query must keep, by activity_id, and why each survives.
EXPECTED_KEPT = {
    1: "ordinary IC50 with a pchembl value",
    2: "Ki against a GPCR",
    3: "Kd against a protease",
    4: "no pchembl value, so the loader must convert standard_value",
    5: "censored, dropped later in curation rather than in SQL",
}

#: Activities the query must reject, and the clause that should reject each.
EXPECTED_REJECTED = {
    6: "assay confidence below 8",
    7: "units are uM, not nM",
    8: "standard_type not among Ki/Kd/IC50/EC50",
    9: "standard_value is not positive",
    10: "data_validity_comment is set",
    11: "potential_duplicate is set",
    12: "no canonical SMILES",
    13: "organism is not human",
    14: "target is a protein complex",
    15: "target has two components, so no single sequence",
}


@pytest.fixture(scope="module")
def extract_module():
    """Import scripts/extract_chembl.py, which is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "extract_chembl", REPO_ROOT / "scripts" / "extract_chembl.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_chembl"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mock_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("chembl") / "mock_chembl.db"
    build(path)
    return path


@pytest.fixture(scope="module")
def extracted(extract_module, mock_db):
    frame, metadata = extract_module.run_extraction(mock_db, None)
    return frame, metadata


class TestQuerySelection:
    def test_keeps_exactly_the_expected_rows(self, extracted):
        frame, _ = extracted
        assert len(frame) == len(EXPECTED_KEPT), (
            f"query returned {len(frame)} rows, expected {len(EXPECTED_KEPT)}. "
            f"Kept rows should be: {EXPECTED_KEPT}"
        )

    def test_reads_the_release_version(self, extracted):
        _, metadata = extracted
        assert metadata["chembl_release"] == "CHEMBL_37", (
            "the release number is written into the extract's metadata; without it "
            "the input is not reproducible"
        )

    def test_carries_every_column_the_loader_needs(self, extracted):
        frame, _ = extracted
        from leakfree_dta.data.sources.chembl import REQUIRED_RAW_COLUMNS

        missing = set(REQUIRED_RAW_COLUMNS) - set(frame.columns)
        assert not missing, f"the ChEMBL loader needs {sorted(missing)}"
        for column in ("document_year", "confidence_score", "assay_id", "target_sequence"):
            assert column in frame.columns, f"{column} is needed downstream"

    def test_publication_years_survive(self, extracted):
        """Without years there is no temporal split, which is half the benchmark."""
        frame, _ = extracted
        import pandas as pd

        years = pd.to_numeric(frame["document_year"], errors="coerce")
        assert years.notna().any()


class TestProteinClassPath:
    """The column the family assignment depends on."""

    def test_path_is_joined_root_to_leaf(self, extracted):
        frame, _ = extracted
        paths = set(frame["target_protein_class"].dropna())
        assert "Enzyme/Kinase/Protein Kinase" in paths, (
            f"expected a joined classification path, got {sorted(paths)}. A bare "
            "top-level class means the COALESCE is picking an ancestor again."
        )

    def test_no_path_is_a_bare_top_level_class(self, extracted):
        frame, _ = extracted
        bare = {
            value
            for value in frame["target_protein_class"].dropna()
            if value in {"Enzyme", "Membrane receptor", "Transporter"}
        }
        assert not bare, f"these rows carry only a top-level class: {sorted(bare)}"

    def test_families_resolve_from_the_database_annotation(self, extracted, extract_module):
        """End to end: the query's output must assign without the override file.

        Overrides exist for the targets the automated rules get wrong, not as the
        primary mechanism. If assignment only works through them, the extract
        would drop every target nobody thought to list.
        """
        frame, _ = extracted
        from leakfree_dta.prot.families import assign_families

        targets = frame.drop_duplicates(subset=["target_uniprot"])[
            ["target_uniprot", "target_protein_class", "target_sequence"]
        ].rename(columns={"target_uniprot": "target_key"})

        assigned = assign_families(
            targets,
            key_column="target_key",
            class_column="target_protein_class",
            keyword_column=None,
            sequence_column="target_sequence",
            overrides={},  # deliberately empty
        )

        families = set(assigned["target_family"])
        assert families == {"kinase", "gpcr", "protease"}, (
            f"without overrides the query's output resolves to {sorted(families)}"
        )
        assert set(assigned["family_confidence"]) == {"database"}
