"""Shared fixtures.

The synthetic dataset is built once per session because standardising and
fingerprinting it is the slow part, and every test wants the same one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from leakfree_dta.data.synthetic import SyntheticGroundTruth, make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic() -> tuple[pd.DataFrame, SyntheticGroundTruth]:
    """The calibration dataset with replicates, and what was planted in it."""
    return make_synthetic_dataset(seed=0)


@pytest.fixture(scope="session")
def synthetic_df(synthetic) -> pd.DataFrame:
    return synthetic[0]


@pytest.fixture(scope="session")
def synthetic_truth(synthetic) -> SyntheticGroundTruth:
    return synthetic[1]


@pytest.fixture(scope="session")
def aggregated(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """The dataset with replicates collapsed, as the models see it."""
    from leakfree_dta.data.curate import aggregate_replicates

    return aggregate_replicates(synthetic_df)


@pytest.fixture(scope="session")
def small_synthetic() -> tuple[pd.DataFrame, SyntheticGroundTruth]:
    """A smaller dataset, for tests whose cost grows with size."""
    return make_synthetic_dataset(
        n_ligands=40, n_targets=8, measurements_per_target=12, n_analog_series=6, seed=1
    )
