"""Schema, curation, standardisation, and the fixture's own integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakfree_dta.data import curate as curate_module
from leakfree_dta.data import schema


class TestSchema:
    def test_fixture_satisfies_the_schema(self, synthetic_df):
        assert schema.validate(synthetic_df) == []

    def test_missing_required_column_is_reported(self, synthetic_df):
        violations = schema.validate(synthetic_df.drop(columns=[schema.Y_TYPE]))
        assert any(violation.column == schema.Y_TYPE for violation in violations)

    def test_implausible_affinity_is_reported(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[broken.index[:3], schema.Y] = 99.0
        violations = schema.validate(broken)
        assert any("plausible range" in violation.problem for violation in violations)

    def test_row_key_is_unique_and_order_independent(self, synthetic_df):
        keys = schema.make_row_key(synthetic_df)
        shuffled = synthetic_df.sample(frac=1.0, random_state=0)
        assert set(keys) == set(schema.make_row_key(shuffled))

    def test_pair_key_identifies_a_measurement_target(self, synthetic_df):
        pairs = schema.make_pair_key(
            synthetic_df[schema.LIGAND_KEY], synthetic_df[schema.TARGET_KEY]
        )
        assert pairs.equals(synthetic_df[schema.PAIR_KEY])


class TestUnitConversion:
    """``pX = -log10([M])`` is the one conversion the whole benchmark rests on."""

    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (1.0, "nM", 9.0),
            (1000.0, "nM", 6.0),
            (1.0, "uM", 6.0),
            (1.0, "mM", 3.0),
            (1.0, "M", 0.0),
            (1.0, "pM", 12.0),
        ],
    )
    def test_known_conversions(self, value, unit, expected):
        result = curate_module.to_p_scale(pd.Series([value]), pd.Series([unit]))
        assert float(result.iloc[0]) == pytest.approx(expected, abs=1e-5)

    def test_non_positive_concentration_becomes_nan(self):
        result = curate_module.to_p_scale(pd.Series([0.0, -5.0]), pd.Series(["nM", "nM"]))
        assert result.isna().all()

    def test_unknown_unit_becomes_nan(self):
        result = curate_module.to_p_scale(pd.Series([1.0]), pd.Series(["furlongs"]))
        assert result.isna().all()

    def test_already_p_scale_passes_through(self):
        result = curate_module.to_p_scale(
            pd.Series([7.5]), pd.Series(["nM"]), already_p=pd.Series([True])
        )
        assert float(result.iloc[0]) == pytest.approx(7.5)


class TestStandardization:
    def test_salt_is_stripped(self):
        from leakfree_dta.chem.standardize import standardize_smiles

        plain = standardize_smiles("CCN")
        salt = standardize_smiles("CCN.Cl")
        assert plain.ok and salt.ok
        assert plain.inchikey == salt.inchikey

    def test_charge_is_neutralised(self):
        from leakfree_dta.chem.standardize import standardize_smiles

        neutral = standardize_smiles("CC(=O)O")
        anion = standardize_smiles("CC(=O)[O-]")
        assert neutral.ok and anion.ok
        assert neutral.inchikey == anion.inchikey

    def test_invalid_smiles_is_reported_not_raised(self):
        from leakfree_dta.chem.standardize import standardize_smiles

        result = standardize_smiles("this is not a molecule")
        assert not result.ok
        assert result.inchikey is None

    def test_skeleton_ignores_stereochemistry(self):
        from leakfree_dta.chem.standardize import standardize_smiles

        left = standardize_smiles("C[C@H](N)C(=O)O")
        right = standardize_smiles("C[C@@H](N)C(=O)O")
        assert left.ok and right.ok
        assert left.inchikey != right.inchikey, "enantiomers must have distinct keys"
        assert left.skeleton == right.skeleton, "but the same skeleton"

    def test_peptides_are_excluded_by_the_druglike_filter(self):
        from leakfree_dta.chem.standardize import is_druglike

        assert is_druglike("CC(=O)Nc1ccc(O)cc1")  # paracetamol
        assert not is_druglike("[Pt](Cl)(Cl)(N)N")  # cisplatin: metal complex
        assert not is_druglike("C")  # too small


class TestAggregation:
    def test_replicates_collapse_to_one_row_per_pair(self, synthetic_df):
        aggregated = curate_module.aggregate_replicates(synthetic_df, how="median")
        assert aggregated[schema.PAIR_KEY].is_unique
        assert len(aggregated) == synthetic_df[schema.PAIR_KEY].nunique()

    def test_spread_is_carried_forward(self, synthetic_df):
        """Replicate spread must survive aggregation.

        The audit measures the noise floor on the replicate table; the models train
        on the aggregated one. Carrying the spread across is what lets a result
        reported on the second be read against a floor measured on the first.
        """
        aggregated = curate_module.aggregate_replicates(synthetic_df)
        assert "y_spread" in aggregated.columns
        assert "y_n_replicates" in aggregated.columns
        assert (aggregated["y_spread"] > 0).any()

    def test_median_resists_an_outlier(self):
        base = pd.DataFrame(
            {
                schema.LIGAND_KEY: ["K"] * 3,
                schema.SMILES: ["CCO"] * 3,
                schema.TARGET_KEY: ["T"] * 3,
                schema.TARGET_SEQUENCE: ["MKV" * 20] * 3,
                schema.Y: [7.0, 7.1, 12.0],
                schema.Y_TYPE: ["pKi"] * 3,
                schema.SOURCE: ["a", "b", "c"],
                schema.ASSAY_KEY: ["1", "2", "3"],
            }
        )
        finalized = schema.finalize(base)
        aggregated = curate_module.aggregate_replicates(finalized, how="median")
        assert float(aggregated[schema.Y].iloc[0]) == pytest.approx(7.1)


class TestFixtureIntegrity:
    """The fixture is an instrument; these tests check it has not drifted."""

    def test_every_template_and_head_combination_is_valid(self):
        from rdkit import Chem, RDLogger

        from leakfree_dta.data.synthetic import (
            _BASE_LINKER_LENGTHS,
            _CORE_TEMPLATES,
            _HEADS,
            _make_series,
        )

        RDLogger.DisableLog("rdApp.*")
        n_series = len(_CORE_TEMPLATES) * len(_HEADS)
        failures: list[str] = []
        for series in range(n_series):
            for length in _BASE_LINKER_LENGTHS:
                smiles = _make_series(series, length)
                if Chem.MolFromSmiles(smiles) is None:
                    failures.append(smiles)
        assert not failures, f"{len(failures)} template combinations do not parse: {failures[:3]}"

    def test_fixture_has_enough_distinct_ligands(self, synthetic_df, synthetic_truth):
        """A collapsed fixture would silently weaken every calibration test."""
        assert synthetic_truth.n_ligands >= 120, (
            f"only {synthetic_truth.n_ligands} distinct ligands; the generator is colliding"
        )
        assert len(synthetic_truth.analog_ligand_keys) >= 10

    def test_shortcut_ceiling_is_in_a_sensible_range(self, synthetic_truth):
        # The affinity model is mostly additive by construction, so the ceiling
        # should be high but not unity -- otherwise there is no interaction to find.
        assert 0.7 < synthetic_truth.shortcut_ceiling < 0.97

    def test_paralogues_are_highly_similar(self, synthetic_df, synthetic_truth):
        from leakfree_dta.prot.identity import local_identity

        sequences = (
            synthetic_df.drop_duplicates(subset=[schema.TARGET_KEY])
            .set_index(schema.TARGET_KEY)[schema.TARGET_SEQUENCE]
            .to_dict()
        )
        identities = [
            local_identity(sequences[parent], sequences[child])
            for parent, child in synthetic_truth.paralogue_target_keys
            if parent in sequences and child in sequences
        ]
        assert identities, "no paralogue pair survived into the dataset"
        assert float(np.min(identities)) > 0.80, (
            f"planted paralogues are only {np.min(identities):.0%} identical; "
            "the protein channel is not being exercised"
        )


class TestProteinIdentityBackends:
    """The two identity backends must be interchangeable.

    They are not an implementation detail: pairwise Smith-Waterman costs roughly
    0.3 s per 400-residue pair, which puts an all-against-all comparison of a few
    thousand proteins at a hundred hours or more, so every real run uses MMseqs2.
    If the two disagreed, a threshold tuned on one would mean something different
    on the other, and the audit's protein-axis numbers would depend on which
    machine produced them.
    """

    @staticmethod
    def _mutate(sequence: str, identity: float, rng) -> str:
        residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        chars = list(sequence)
        for position in rng.choice(len(chars), round((1 - identity) * len(chars)), replace=False):
            chars[position] = str(rng.choice(residues))
        return "".join(chars)

    def test_backends_agree_on_planted_identities(self):
        from leakfree_dta.prot.identity import (
            _max_identity_pairwise,
            identity_backend,
            max_identity_to_set,
        )

        if identity_backend() != "mmseqs2":
            pytest.skip("MMseqs2 not installed; there is only one backend to compare")

        rng = np.random.default_rng(0)
        residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        references = {f"T{i}": "".join(rng.choice(residues, 400)) for i in range(25)}
        planted = {0.95: "P95", 0.80: "P80", 0.60: "P60"}
        queries = {
            name: self._mutate(references["T0"], identity, rng)
            for identity, name in planted.items()
        }

        fast = max_identity_to_set(queries, references)
        exact = _max_identity_pairwise(queries, references)

        for identity, name in planted.items():
            assert fast[name][1] == pytest.approx(identity, abs=0.08), (
                f"MMseqs2 reported {fast[name][1]:.3f} for a planted {identity:.2f}"
            )
            assert fast[name][1] == pytest.approx(exact[name][1], abs=0.05), (
                f"backends disagree on {name}: mmseqs2 {fast[name][1]:.3f}, "
                f"pairwise {exact[name][1]:.3f}"
            )

    def test_a_target_on_both_sides_scores_one(self):
        """The same accession in train and test is total leakage, not a near miss.

        Alignment trimming can report a self-match slightly below 1.0, which would
        quietly drop transductive splits under a 1.0 threshold. The identical key
        is therefore seeded rather than left to the search.
        """
        from leakfree_dta.prot.identity import _max_identity_pairwise, max_identity_to_set

        rng = np.random.default_rng(1)
        residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        shared = {"SAME": "".join(rng.choice(residues, 300))}
        references = {**shared, "OTHER": "".join(rng.choice(residues, 300))}

        for backend in (max_identity_to_set, _max_identity_pairwise):
            result = backend(shared, references)
            assert result["SAME"][1] == pytest.approx(1.0), f"{backend.__name__} gave a self-match below 1.0"
            assert result["SAME"][0] == "SAME"


class TestChemblProteinClassPaths:
    """Family assignment against real ChEMBL protein_classification paths.

    These strings are what `sql/chembl_activities.sql` produces, and the test
    exists because an earlier version of that query produced something else. It
    selected one level of the classification hierarchy with a COALESCE over the
    ancestors, which returns the *most generic* name -- "Enzyme" for a kinase,
    "Membrane receptor" for a GPCR. Neither contains a word the regexes look for,
    so nearly every target would have fallen through to the sequence-motif
    heuristic and then been dropped as unassigned, and the three-family extract
    would have come out close to empty after a multi-hour download.

    Pinning the paths here means a change to either the query or the regexes has
    to confront the other.
    """

    IN_SCOPE = (
        ("Enzyme/Kinase/Protein Kinase", "kinase"),
        (
            "Enzyme/Kinase/Protein Kinase/TK protein kinase group/"
            "Tyrosine protein kinase EGFR family",
            "kinase",
        ),
        ("Enzyme/Kinase/Protein Kinase/CMGC protein kinase group", "kinase"),
        ("Membrane receptor/Family A G protein-coupled receptor", "gpcr"),
        (
            "Membrane receptor/Family A G protein-coupled receptor/"
            "Small molecule receptor (family A GPCR)/Monoamine receptor",
            "gpcr",
        ),
        ("Membrane receptor/Family B G protein-coupled receptor", "gpcr"),
        ("Enzyme/Protease/Serine protease", "protease"),
        ("Enzyme/Protease/Serine protease/Serine protease SC clan", "protease"),
        ("Enzyme/Protease/Metallo protease", "protease"),
        ("Enzyme/Protease/Cysteine protease", "protease"),
        ("Enzyme/Protease/Aspartic protease", "protease"),
    )

    OUT_OF_SCOPE = (
        "Enzyme/Oxidoreductase/Cytochrome P450",
        "Transcription factor/Nuclear receptor",
        "Ion channel/Voltage-gated ion channel",
        "Transporter/Electrochemical transporter",
    )

    @pytest.mark.parametrize(("path", "expected"), IN_SCOPE)
    def test_in_scope_paths_are_assigned(self, path, expected):
        from leakfree_dta.prot.families import assign_family

        result = assign_family(target_key="X", protein_class=path)
        assert result.family == expected, f"{path!r} -> {result.family!r}, expected {expected!r}"
        assert result.confidence == "database", (
            "assignment fell back below the database annotation, which means the "
            "classification path is not being matched"
        )

    @pytest.mark.parametrize("path", OUT_OF_SCOPE)
    def test_out_of_scope_paths_are_not_claimed(self, path):
        from leakfree_dta.prot.families import assign_family

        result = assign_family(target_key="X", protein_class=path)
        assert result.family == "unassigned", (
            f"{path!r} was claimed as {result.family!r}; the scope filter is too loose"
        )

    def test_a_bare_top_level_class_is_not_enough(self):
        """The symptom the old query produced, pinned so it cannot return quietly."""
        from leakfree_dta.prot.families import assign_family

        for generic in ("Enzyme", "Membrane receptor"):
            result = assign_family(target_key="X", protein_class=generic)
            assert result.family == "unassigned", (
                f"{generic!r} was assigned to {result.family!r}. A top-level class "
                "carries no family information, and treating it as if it does would "
                "hide a broken extraction query."
            )
