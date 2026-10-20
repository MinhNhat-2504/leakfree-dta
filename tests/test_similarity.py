"""Ligand clustering: the invariant it must satisfy, and the scaling it must hold.

Clustering is on the critical path for the ligand-cluster and leakage-free
splits, and it runs over the full ligand set, so both correctness and cost
matter. The scaling test exists because an earlier implementation rebuilt Python
lists inside the loop and scaled super-quadratically: doubling from 5,000 to
10,000 real ChEMBL ligands took 7.7x longer, which put the 466,000-structure
extract days out of reach.
"""

from __future__ import annotations

import time

import pytest
from rdkit import DataStructs

from leakfree_dta.chem.fingerprints import ecfp_bitvects
from leakfree_dta.chem.similarity import cluster_ligands


def _ligands(n: int) -> list:
    """Chemically varied molecules: a decorated core with several variable parts."""
    from leakfree_dta.data.synthetic import _CORE_TEMPLATES, _HEADS, _make_series

    n_series = len(_CORE_TEMPLATES) * len(_HEADS)
    smiles = [_make_series(i % n_series, 3 + 2 * ((i // n_series) % 8)) for i in range(n)]
    return ecfp_bitvects(smiles)


class TestClusteringIsValid:
    def test_every_member_is_close_to_its_leader(self):
        """The defining property of the leader path: a cluster is a sphere.

        Checked directly rather than against a reference implementation, so the
        test survives the implementation being replaced.

        ``exact_limit=0`` forces the leader algorithm. Butina, which handles
        small inputs, picks the molecule with the most neighbours as centroid
        rather than the first one encountered, so this invariant is not its
        invariant and asserting it there would fail for the wrong reason.
        """
        cutoff = 0.4
        vectors = _ligands(600)
        labels = cluster_ligands(vectors, cutoff=cutoff, exact_limit=0)

        # The leader of a cluster is its first member, by construction.
        first_seen: dict[int, int] = {}
        for index, label in enumerate(labels):
            first_seen.setdefault(int(label), index)

        violations = 0
        for index, label in enumerate(labels):
            leader = first_seen[int(label)]
            if leader == index or vectors[index] is None or vectors[leader] is None:
                continue
            similarity = DataStructs.TanimotoSimilarity(vectors[index], vectors[leader])
            if similarity < 1.0 - cutoff:
                violations += 1

        assert violations == 0, f"{violations} molecules sit outside their leader's sphere"

    def test_identical_molecules_cluster_together(self):
        vectors = _ligands(200)
        doubled = vectors + vectors
        labels = cluster_ligands(doubled, cutoff=0.4)
        assert (labels[:200] == labels[200:]).all()

    def test_a_tighter_cutoff_gives_more_clusters(self):
        vectors = _ligands(400)
        loose = len(set(cluster_ligands(vectors, cutoff=0.6).tolist()))
        tight = len(set(cluster_ligands(vectors, cutoff=0.2).tolist()))
        assert tight > loose

    def test_unfingerprintable_molecules_get_their_own_cluster(self):
        vectors = _ligands(100)
        vectors[10] = None
        vectors[20] = None
        labels = cluster_ligands(vectors, cutoff=0.4)
        assert len(labels) == len(vectors)
        assert labels[10] != labels[20], "molecules with no fingerprint were merged"

    def test_result_is_deterministic(self):
        vectors = _ligands(300)
        assert (cluster_ligands(vectors, cutoff=0.4) == cluster_ligands(vectors, cutoff=0.4)).all()

    def test_the_two_backends_broadly_agree(self):
        """Butina and leader clustering should not disagree wildly.

        They are different algorithms and will not match exactly -- Butina
        chooses centroids by neighbour count, the leader path takes molecules in
        order -- but if the cluster counts diverged by much, the switch at
        EXACT_CLUSTERING_LIMIT would put a visible discontinuity in the middle of
        the benchmark.
        """
        vectors = _ligands(500)
        exact = len(set(cluster_ligands(vectors, cutoff=0.4, exact_limit=10_000).tolist()))
        leader = len(set(cluster_ligands(vectors, cutoff=0.4, exact_limit=0).tolist()))
        assert 0.5 < leader / exact < 2.0, (
            f"Butina found {exact} clusters, the leader algorithm {leader}; the two "
            "paths disagree enough that results would jump at the size threshold"
        )


class TestClusteringScales:
    @pytest.mark.slow
    def test_scaling_stays_near_quadratic(self):
        """Doubling the input must not much more than quadruple the time.

        The leader algorithm is O(n x n_leaders), so with cluster count roughly
        proportional to n the expected factor is about 4. The regression this
        guards against showed 7.7x, from interpreted work inside the loop.

        The bound is loose because a test machine is a noisy clock; it is set to
        catch an order-of-magnitude regression, not a slow afternoon.
        """
        timings: dict[int, float] = {}
        for n in (2000, 4000):
            vectors = _ligands(n)
            start = time.monotonic()
            # The leader path explicitly: it is the one that runs at real scale,
            # and the one the regression was in.
            cluster_ligands(vectors, cutoff=0.4, exact_limit=0)
            timings[n] = time.monotonic() - start

        if timings[2000] < 0.05:
            pytest.skip("too fast to time reliably on this machine")

        factor = timings[4000] / timings[2000]
        assert factor < 6.5, (
            f"doubling the input multiplied the time by {factor:.1f}. Expected about 4 "
            "for O(n x n_leaders); a larger factor means interpreted work has crept "
            "back into the loop."
        )
