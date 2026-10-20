"""A synthetic dataset with leakage planted on purpose.

This exists for a reason beyond convenience. An audit tool that reports "4.2% of
test rows leak" is a measuring instrument, and an instrument has to be calibrated
against a known quantity before its readings mean anything. So this module builds
a dataset where the answer is known by construction:

* ``n_duplicate_pairs`` pairs are measured twice, with a controlled disagreement.
* ``n_analog_series`` ligands are near-duplicates of another ligand by design.
* ``n_paralogue_pairs`` targets are high-identity copies of another target.
* Affinity is generated from an explicit additive-plus-interaction model, so the
  share of variance that a shortcut predictor *should* be able to reach is known
  in advance.

A check that fails to recover these planted quantities is broken, and the test
suite asserts that it recovers them. Without this, the audit's numbers would rest
on nothing but the plausibility of the code.

The molecules are small and chemically simple -- they are not meant to be
realistic drug candidates, only to exercise RDKit's standardisation, fingerprint
and scaffold machinery on valid structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Decorated drug-sized cores, each with two substitution slots: ``{linker}`` for
#: an aliphatic chain and ``{head}`` for a terminal group. Every combination used
#: below parses and sanitises in RDKit, which is asserted by the test suite rather
#: than assumed -- a template that silently failed would shrink the fixture and
#: weaken the calibration it exists to provide.
_CORE_TEMPLATES = (
    "O=C(Nc1ccc(S(=O)(=O)N)cc1){linker}c1ccc{head}cc1",
    "COc1ccc(NC(=O){linker}c2ccc{head}cc2)cc1",
    "c1ccc2nc(N3CCN(C){linker}CC3)nc(Nc3ccc{head}cc3)c2c1",
    "O=C(NCc1ccncc1){linker}c1ccc2[nH]nc{head}c2c1",
    "Clc1ccc(C(=O)N2CCC({linker}c3ccc{head}cc3)CC2)cc1",
    "N#Cc1ccc(Oc2ccc({linker}C(=O)NC3CCC{head}CC3)cc2)cc1",
    "CN(C)C(=O){linker}c1ccc(-c2ccc{head}cc2)cc1",
)

#: Terminal groups. Each already carries its own parentheses so the empty option
#: is a valid substitution too.
_HEADS = ("", "(C)", "(CC)", "(OC)", "(F)", "(Cl)", "(N)", "(C#N)", "(C(F)(F)F)", "(S(C)=O)")

#: Linker lengths for base ligands. Seven cores times ten heads times six lengths
#: gives 420 distinct molecules, comfortably more than the fixture needs.
_BASE_LINKER_LENGTHS = (3, 5, 7, 9, 11, 13)


@dataclass(frozen=True)
class _LigandDraw:
    """One generated ligand and, for an analogue, which series it belongs to."""

    smiles: str
    series: int
    homolog_length: int
    is_analog: bool


def _make_series(series: int, length: int) -> str:
    """One member of a homologous series: a fixed core and head, n-carbon linker.

    A series is identified by ``(core, head)``; its members differ only in linker
    length. Members of one series have distinct InChIKeys and distinct
    Bemis-Murcko scaffolds, but identical ECFP4 fingerprints.
    """
    template = _CORE_TEMPLATES[series % len(_CORE_TEMPLATES)]
    head = _HEADS[(series // len(_CORE_TEMPLATES)) % len(_HEADS)]
    return template.format(linker="C" * length, head=head)


@dataclass
class SyntheticGroundTruth:
    """What was planted, for the calibration tests to assert against."""

    n_rows: int
    n_ligands: int
    n_targets: int
    duplicate_pair_keys: list[str] = field(default_factory=list)
    conflicting_pair_keys: list[str] = field(default_factory=list)
    analog_ligand_keys: list[str] = field(default_factory=list)
    paralogue_target_keys: list[tuple[str, str]] = field(default_factory=list)
    target_effect_sd: float = 0.0
    ligand_effect_sd: float = 0.0
    interaction_sd: float = 0.0
    noise_sd: float = 0.0

    @property
    def shortcut_ceiling(self) -> float:
        """The Pearson correlation a perfect additive-bias model should reach.

        With affinity generated as ``target + ligand + interaction + noise`` and
        all four terms independent, a model capturing only the two main effects
        explains ``(t^2 + l^2) / (t^2 + l^2 + i^2 + n^2)`` of the variance, so its
        correlation is the square root of that. Any shortcut probe reporting much
        above this on the synthetic data has a bug.
        """
        main = self.target_effect_sd**2 + self.ligand_effect_sd**2
        total = main + self.interaction_sd**2 + self.noise_sd**2
        return float(np.sqrt(main / total)) if total > 0 else 0.0


def _make_sequence(rng: np.random.Generator, length: int = 240) -> str:
    """A random sequence, with no family structure.

    Deliberately unstructured: base targets are mutually unrelated, so the only
    high-identity pairs in the fixture are the paralogues planted on purpose by
    :func:`_mutate_sequence`. That is what makes the protein-overlap check's
    recovered count comparable to a known number.
    """
    residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    return "".join(rng.choice(residues, size=length))


def _mutate_sequence(sequence: str, rng: np.random.Generator, identity: float = 0.92) -> str:
    """A high-identity paralogue: substitute a fraction of positions."""
    residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    chars = list(sequence)
    n_mutations = round((1.0 - identity) * len(chars))
    positions = rng.choice(len(chars), size=n_mutations, replace=False)
    for position in positions:
        chars[position] = str(rng.choice(residues))
    return "".join(chars)


def make_synthetic_dataset(
    *,
    n_ligands: int = 180,
    n_targets: int = 24,
    measurements_per_target: int = 40,
    n_duplicate_pairs: int = 60,
    conflict_log_units: float = 1.6,
    n_analog_series: int = 30,
    n_paralogue_pairs: int = 4,
    target_effect_sd: float = 0.9,
    ligand_effect_sd: float = 0.7,
    interaction_sd: float = 0.5,
    noise_sd: float = 0.3,
    year_range: tuple[int, int] = (2005, 2024),
    seed: int = 0,
) -> tuple[pd.DataFrame, SyntheticGroundTruth]:
    """Build the calibration dataset and return it with its ground truth.

    The default sizes are chosen so the whole pipeline -- standardisation,
    clustering, every audit check, the fallback sequence alignment -- runs in
    seconds, which is what makes it usable as a test fixture.
    """
    rng = np.random.default_rng(seed)

    # --- ligands ------------------------------------------------------------
    # Analogues are planted as homologous series: the same core with a linker one
    # or two carbons longer. This is the fixture that makes the audit testable,
    # and it reproduces a real and under-appreciated leakage channel. Homologues
    # have DIFFERENT InChIKeys and DIFFERENT Bemis-Murcko scaffolds, so a
    # scaffold split separates them -- yet ECFP4 assigns them identical
    # fingerprints, so a model cannot tell them apart. A scaffold-split test set
    # built from such data is, for the model, partly a copy of its training set.
    # Enumerated rather than sampled: (series, linker length) is walked in order so
    # every base ligand is distinct by construction. Sampling at random collides
    # often enough to silently shrink the fixture, which would weaken the very
    # calibration this dataset exists to provide.
    n_series = len(_CORE_TEMPLATES) * len(_HEADS)
    draws: list[_LigandDraw] = []

    for index in range(n_ligands - n_analog_series):
        series = index % n_series
        length = _BASE_LINKER_LENGTHS[(index // n_series) % len(_BASE_LINKER_LENGTHS)]
        draws.append(_LigandDraw(_make_series(series, length), series, length, False))

    # Each analogue is a homologue of a base ligand: same core and head, linker one
    # or two carbons longer. Distinct compound, identical ECFP4.
    for index in range(n_analog_series):
        parent = draws[index % len(draws)]
        # Base linkers are odd-length, so +1 or +3 keeps every analogue even and
        # therefore distinct from every base ligand.
        length = parent.homolog_length + 1 + 2 * (index % 2)
        draws.append(_LigandDraw(_make_series(parent.series, length), parent.series, length, True))

    from leakfree_dta.chem.standardize import standardize_series

    standardized = standardize_series([draw.smiles for draw in draws])
    valid = standardized["std_ok"].to_numpy()
    ligands = pd.DataFrame(
        {
            schema.SMILES: standardized.loc[valid, "smiles_std"].to_numpy(),
            schema.LIGAND_KEY: standardized.loc[valid, "inchikey"].to_numpy(),
            "series": [draw.series for draw, ok in zip(draws, valid, strict=True) if ok],
            "is_analog": [draw.is_analog for draw, ok in zip(draws, valid, strict=True) if ok],
        }
    ).drop_duplicates(subset=[schema.LIGAND_KEY], keep="first")

    analog_keys = ligands.loc[ligands["is_analog"], schema.LIGAND_KEY].tolist()

    # --- targets ------------------------------------------------------------
    families = np.array(schema.TARGET_FAMILIES)
    n_base_targets = n_targets - n_paralogue_pairs
    records: list[dict[str, object]] = []
    for index in range(n_base_targets):
        records.append(
            {
                schema.TARGET_KEY: f"SYNT{index:04d}",
                schema.TARGET_SEQUENCE: _make_sequence(rng),
                schema.TARGET_FAMILY: str(families[index % len(families)]),
            }
        )

    paralogue_pairs: list[tuple[str, str]] = []
    for index in range(n_paralogue_pairs):
        parent = records[index % n_base_targets]
        key = f"SYNP{index:04d}"
        records.append(
            {
                schema.TARGET_KEY: key,
                schema.TARGET_SEQUENCE: _mutate_sequence(
                    str(parent[schema.TARGET_SEQUENCE]), rng, identity=0.92
                ),
                schema.TARGET_FAMILY: parent[schema.TARGET_FAMILY],
            }
        )
        paralogue_pairs.append((str(parent[schema.TARGET_KEY]), key))

    targets = pd.DataFrame.from_records(records)

    # --- affinity model -----------------------------------------------------
    target_effect = dict(
        zip(
            targets[schema.TARGET_KEY],
            rng.normal(0.0, target_effect_sd, len(targets)),
            strict=True,
        )
    )
    ligand_effect = dict(
        zip(
            ligands[schema.LIGAND_KEY],
            rng.normal(0.0, ligand_effect_sd, len(ligands)),
            strict=True,
        )
    )

    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        target_key = str(getattr(target, schema.TARGET_KEY))
        n_draw = min(measurements_per_target, len(ligands))
        chosen = ligands.sample(n_draw, random_state=int(rng.integers(0, 2**31 - 1)))
        for ligand in chosen.itertuples(index=False):
            ligand_key = str(getattr(ligand, schema.LIGAND_KEY))
            value = (
                6.5
                + target_effect[target_key]
                + ligand_effect[ligand_key]
                + rng.normal(0.0, interaction_sd)
                + rng.normal(0.0, noise_sd)
            )
            rows.append(
                {
                    schema.LIGAND_KEY: ligand_key,
                    schema.SMILES: str(getattr(ligand, schema.SMILES)),
                    schema.TARGET_KEY: target_key,
                    schema.TARGET_SEQUENCE: str(getattr(target, schema.TARGET_SEQUENCE)),
                    schema.TARGET_FAMILY: str(getattr(target, schema.TARGET_FAMILY)),
                    schema.Y: float(np.clip(value, 2.0, 12.0)),
                    schema.Y_TYPE: "pKi" if rng.random() < 0.7 else "pIC50",
                    schema.SOURCE: "synthetic",
                    schema.DOC_YEAR: int(rng.integers(year_range[0], year_range[1] + 1)),
                    schema.ASSAY_KEY: f"ASY{rng.integers(0, 10**6):06d}",
                }
            )

    df = pd.DataFrame.from_records(rows)

    # --- plant duplicates and conflicts -------------------------------------
    duplicate_keys: list[str] = []
    conflicting_keys: list[str] = []
    if n_duplicate_pairs > 0 and len(df):
        picks = df.sample(min(n_duplicate_pairs, len(df)), random_state=seed)
        replicas = picks.copy()
        n_conflicting = len(replicas) // 2
        offsets = np.concatenate(
            [
                # Half the replicates disagree beyond the conflict threshold.
                rng.choice([-1, 1], n_conflicting) * conflict_log_units,
                # The other half agree to within normal assay reproducibility.
                rng.normal(0.0, 0.15, len(replicas) - n_conflicting),
            ]
        )
        replicas[schema.Y] = np.clip(replicas[schema.Y].to_numpy() + offsets, 2.0, 12.0)
        replicas[schema.ASSAY_KEY] = [f"ASYR{index:06d}" for index in range(len(replicas))]
        replicas[schema.SOURCE] = "synthetic_replicate"

        duplicate_keys = schema.make_pair_key(
            picks[schema.LIGAND_KEY], picks[schema.TARGET_KEY]
        ).tolist()
        conflicting_keys = duplicate_keys[:n_conflicting]
        df = pd.concat([df, replicas], ignore_index=True)

    df = schema.finalize(df)
    truth = SyntheticGroundTruth(
        n_rows=len(df),
        n_ligands=int(df[schema.LIGAND_KEY].nunique()),
        n_targets=int(df[schema.TARGET_KEY].nunique()),
        duplicate_pair_keys=duplicate_keys,
        conflicting_pair_keys=conflicting_keys,
        analog_ligand_keys=analog_keys,
        paralogue_target_keys=paralogue_pairs,
        target_effect_sd=target_effect_sd,
        ligand_effect_sd=ligand_effect_sd,
        interaction_sd=interaction_sd,
        noise_sd=noise_sd,
    )

    logger.info(
        "synthetic dataset: %d rows, %d ligands, %d targets; planted %d duplicate pairs "
        "(%d conflicting), %d analogue ligands, %d paralogue target pairs; shortcut ceiling %.3f",
        truth.n_rows,
        truth.n_ligands,
        truth.n_targets,
        len(duplicate_keys),
        len(conflicting_keys),
        len(analog_keys),
        len(paralogue_pairs),
        truth.shortcut_ceiling,
    )
    return df, truth
