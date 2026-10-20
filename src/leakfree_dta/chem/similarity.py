"""Tanimoto similarity and ligand clustering.

Near-duplicate detection is the most expensive part of the audit: a full pairwise
comparison over a ChEMBL-scale ligand set is ~10^10 similarities. Two things keep
it tractable, and both are design decisions worth stating in the paper's methods:

* The audit only ever needs **cross-split** similarities, which is a
  ``n_test x n_train`` block rather than the full square, and only the maximum
  per test ligand. That is computed in chunks with RDKit's bulk Tanimoto.
* Clustering uses **sphere exclusion** (the leader-follower algorithm Butina
  approximates) above a size threshold, which is ``O(n_ligands x n_clusters)``
  instead of quadratic. Below the threshold, RDKit's exact Butina is used so that
  small-scale results match the reference implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

BitVect = DataStructs.ExplicitBitVect

#: Above this many molecules, switch from exact Butina to sphere exclusion.
EXACT_CLUSTERING_LIMIT = 8_000


@dataclass(frozen=True)
class NearestNeighbor:
    """For each query molecule, its closest reference molecule."""

    similarity: np.ndarray  # (n_query,) float32, NaN where the query failed to parse
    index: np.ndarray  # (n_query,) int32, -1 where undefined

    def above(self, threshold: float) -> np.ndarray:
        """Boolean mask of queries whose nearest reference is at least ``threshold``."""
        with np.errstate(invalid="ignore"):
            return np.nan_to_num(self.similarity, nan=-1.0) >= threshold


def nearest_neighbor(
    query: Sequence[BitVect | None],
    reference: Sequence[BitVect | None],
    *,
    chunk_size: int = 512,
) -> NearestNeighbor:
    """Maximum Tanimoto from each query molecule to any reference molecule.

    This is the primitive behind the near-duplicate-ligand check: a test ligand
    whose nearest training neighbour is at 0.9 Tanimoto is, for practical
    purposes, already in the training set.
    """
    reference_valid = [vector for vector in reference if vector is not None]
    n_query = len(query)
    similarity = np.full(n_query, np.nan, dtype=np.float32)
    index = np.full(n_query, -1, dtype=np.int32)

    if not reference_valid:
        logger.warning("no valid reference fingerprints; nearest-neighbour is undefined")
        return NearestNeighbor(similarity, index)

    # Map positions in the filtered reference list back to the caller's indices.
    reference_positions = np.array(
        [i for i, vector in enumerate(reference) if vector is not None], dtype=np.int32
    )

    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        for offset in range(start, stop):
            vector = query[offset]
            if vector is None:
                continue
            scores = np.asarray(
                DataStructs.BulkTanimotoSimilarity(vector, reference_valid), dtype=np.float32
            )
            best = int(np.argmax(scores))
            similarity[offset] = scores[best]
            index[offset] = reference_positions[best]

    return NearestNeighbor(similarity, index)


def pairs_above(
    vectors: Sequence[BitVect | None],
    threshold: float,
    *,
    max_pairs: int = 1_000_000,
) -> list[tuple[int, int, float]]:
    """All within-set pairs at or above ``threshold``, as ``(i, j, similarity)``.

    Used to quantify redundancy inside a single split -- an effective-sample-size
    correction the paper applies before computing confidence intervals, since
    1000 measurements of 50 distinct chemotypes are not 1000 independent samples.
    """
    out: list[tuple[int, int, float]] = []
    n = len(vectors)
    for i in range(n):
        left = vectors[i]
        if left is None:
            continue
        tail = [vector for vector in vectors[i + 1 :] if vector is not None]
        if not tail:
            continue
        tail_positions = [j for j in range(i + 1, n) if vectors[j] is not None]
        scores = DataStructs.BulkTanimotoSimilarity(left, tail)
        for position, score in zip(tail_positions, scores, strict=True):
            if score >= threshold:
                out.append((i, position, float(score)))
                if len(out) >= max_pairs:
                    logger.warning("pairs_above hit max_pairs=%d; results truncated", max_pairs)
                    return out
    return out


def cluster_ligands(
    vectors: Sequence[BitVect | None],
    *,
    cutoff: float = 0.4,
    exact_limit: int = EXACT_CLUSTERING_LIMIT,
) -> np.ndarray:
    """Cluster molecules at a Tanimoto *distance* cutoff.

    ``cutoff`` is a distance, following RDKit's Butina convention: 0.4 means
    molecules within 0.6 Tanimoto similarity join the same cluster. Molecules
    without a fingerprint get their own singleton cluster rather than being
    dropped, so cluster labels stay aligned with the input rows.
    """
    n = len(vectors)
    labels = np.full(n, -1, dtype=np.int32)
    valid = [i for i, vector in enumerate(vectors) if vector is not None]

    if not valid:
        return np.arange(n, dtype=np.int32)

    if len(valid) <= exact_limit:
        sub_labels = _butina_exact([vectors[i] for i in valid], cutoff)
    else:
        logger.info(
            "clustering %d molecules by sphere exclusion (exact Butina limit is %d)",
            len(valid),
            exact_limit,
        )
        sub_labels = _sphere_exclusion([vectors[i] for i in valid], cutoff)

    for position, label in zip(valid, sub_labels, strict=True):
        labels[position] = label

    # Give every unfingerprintable molecule a distinct label above the real ones.
    next_label = int(sub_labels.max()) + 1 if len(sub_labels) else 0
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1
    return labels


def _butina_exact(vectors: Sequence[BitVect], cutoff: float) -> np.ndarray:
    """RDKit's Butina clustering on the full condensed distance matrix."""
    from rdkit.ML.Cluster import Butina

    n = len(vectors)
    distances: list[float] = []
    for i in range(1, n):
        similarities = DataStructs.BulkTanimotoSimilarity(vectors[i], list(vectors[:i]))
        distances.extend(1.0 - score for score in similarities)

    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    labels = np.zeros(n, dtype=np.int32)
    for label, cluster in enumerate(clusters):
        for member in cluster:
            labels[member] = label
    return labels


def _sphere_exclusion(vectors: Sequence[BitVect], cutoff: float) -> np.ndarray:
    """Leader-follower clustering: pick a leader, absorb everything within cutoff.

    Deterministic given the input order. Callers that need order-independence
    should sort the molecules by a stable key (the pipeline sorts by InChIKey)
    before calling.
    """
    n = len(vectors)
    threshold = 1.0 - cutoff
    labels = np.full(n, -1, dtype=np.int32)
    leaders: list[int] = []

    for i in range(n):
        if labels[i] != -1:
            continue
        if leaders:
            scores = np.asarray(
                DataStructs.BulkTanimotoSimilarity(vectors[i], [vectors[j] for j in leaders]),
                dtype=np.float32,
            )
            best = int(np.argmax(scores))
            if scores[best] >= threshold:
                labels[i] = labels[leaders[best]]
                continue

        label = len(leaders)
        labels[i] = label
        leaders.append(i)

        # Absorb all remaining unassigned molecules within the new leader's sphere.
        remaining = [j for j in range(i + 1, n) if labels[j] == -1]
        if not remaining:
            continue
        scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(vectors[i], [vectors[j] for j in remaining]),
            dtype=np.float32,
        )
        for j, score in zip(remaining, scores, strict=True):
            if score >= threshold:
                labels[j] = label

    return labels
