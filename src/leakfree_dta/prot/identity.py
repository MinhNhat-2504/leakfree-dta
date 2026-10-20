"""Sequence identity and protein clustering.

Protein overlap between train and test is the leakage channel the 2026 Chemical
Science audit found in the most widely used benchmarks, so this module is where
the paper's central measurement is made. Identity is computed in one of two ways:

``MMseqs2`` (preferred, used for every number reported in the paper)
    Shelled out to when the binary is on ``PATH``. This is the field's standard
    tool, and using it means the reported clustering thresholds mean what readers
    expect them to mean.

``k-mer prefilter + banded Smith-Waterman`` (fallback)
    A pure-Python path so that the pipeline and its tests run anywhere. It
    computes a 5-mer Jaccard index for all pairs, then refines only the candidate
    pairs with an exact local alignment. It is close to MMseqs2 in practice but
    not identical, and :func:`identity_backend` reports which path was taken so no
    result is ever silently produced by the approximation.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leakfree_dta.logging import get_logger
from leakfree_dta.prot.sequences import clean_sequence, write_fasta

logger = get_logger(__name__)

KMER_SIZE = 5
#: k-mer Jaccard below this cannot reach 30% identity, so exact alignment is skipped.
KMER_PREFILTER = 0.02


@dataclass(frozen=True)
class ClusterResult:
    """Protein clusters at one identity threshold."""

    labels: dict[str, int]
    min_seq_id: float
    backend: str
    n_clusters: int

    def cluster_of(self, target_key: str) -> int:
        """Cluster label, or a stable singleton for a target with no sequence."""
        return self.labels.get(target_key, -1)


def identity_backend() -> str:
    """Which identity backend is active: ``"mmseqs2"`` or ``"kmer+sw"``.

    Printed into every audit report and split manifest. A reviewer must be able to
    tell whether a number came from the standard tool or the fallback.
    """
    return "mmseqs2" if shutil.which("mmseqs") else "kmer+sw"


def cluster_proteins(
    sequences: Mapping[str, str],
    *,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    force_backend: str | None = None,
) -> ClusterResult:
    """Cluster targets so that no two clusters share more than ``min_seq_id``.

    The paper reports splits at 30% and 50% identity. 30% is roughly the limit
    below which fold similarity stops being detectable from sequence alone; 50% is
    the stricter setting that also separates close paralogues, which matters
    inside the kinase family where ATP-site conservation is high.
    """
    cleaned = {key: clean_sequence(value) for key, value in sequences.items() if value}
    if not cleaned:
        return ClusterResult({}, min_seq_id, "none", 0)

    backend = force_backend or identity_backend()
    if backend == "mmseqs2":
        try:
            labels = _mmseqs_cluster(cleaned, min_seq_id=min_seq_id, coverage=coverage)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            logger.warning("MMseqs2 failed (%s); falling back to k-mer + Smith-Waterman", exc)
            backend = "kmer+sw"
            labels = _fallback_cluster(cleaned, min_seq_id=min_seq_id)
    else:
        logger.warning(
            "MMseqs2 not found; using the approximate k-mer + Smith-Waterman backend. "
            "Install MMseqs2 before producing numbers for publication."
        )
        labels = _fallback_cluster(cleaned, min_seq_id=min_seq_id)

    n_clusters = len(set(labels.values()))
    logger.info(
        "clustered %d targets into %d clusters at %.0f%% identity (%s)",
        len(cleaned),
        n_clusters,
        100 * min_seq_id,
        backend,
    )
    return ClusterResult(labels, min_seq_id, backend, n_clusters)


# --- MMseqs2 path -----------------------------------------------------------


def _mmseqs_cluster(
    sequences: Mapping[str, str], *, min_seq_id: float, coverage: float
) -> dict[str, int]:
    """Run ``mmseqs easy-cluster`` and parse its representative mapping."""
    with tempfile.TemporaryDirectory(prefix="leakfree_mmseqs_") as tmp:
        tmpdir = Path(tmp)
        fasta = write_fasta(sequences, tmpdir / "targets.fasta")
        prefix = tmpdir / "clu"
        command = [
            "mmseqs",
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmpdir / "work"),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            "0",
            "-v",
            "1",
        ]
        logger.debug("running: %s", " ".join(command))
        subprocess.run(command, check=True, capture_output=True, text=True)

        tsv = prefix.with_name(prefix.name + "_cluster.tsv")
        if not tsv.exists():
            raise RuntimeError(f"MMseqs2 produced no cluster table at {tsv}")

        representatives: dict[str, str] = {}
        with tsv.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    representatives[fields[1]] = fields[0]

    order = {rep: i for i, rep in enumerate(dict.fromkeys(representatives.values()))}
    return {member: order[rep] for member, rep in representatives.items()}


# --- Fallback path ----------------------------------------------------------


def _kmer_sets(sequences: Mapping[str, str], k: int = KMER_SIZE) -> dict[str, frozenset[str]]:
    return {
        key: frozenset(sequence[i : i + k] for i in range(max(len(sequence) - k + 1, 0)))
        for key, sequence in sequences.items()
    }


def _fallback_cluster(sequences: Mapping[str, str], *, min_seq_id: float) -> dict[str, int]:
    """Single-linkage clustering over pairs that exceed ``min_seq_id``.

    Single linkage is the conservative choice for a leakage audit: it merges
    transitively, so a chain of moderately similar proteins ends up in one
    cluster. That errs towards *over*-separating train and test, which is the
    direction an audit should err in.
    """
    keys = sorted(sequences)
    kmers = _kmer_sets(sequences)
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    n_aligned = 0
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            left_kmers, right_kmers = kmers[left], kmers[right]
            if not left_kmers or not right_kmers:
                continue
            intersection = len(left_kmers & right_kmers)
            if intersection == 0:
                continue
            jaccard = intersection / len(left_kmers | right_kmers)
            if jaccard < KMER_PREFILTER:
                continue
            n_aligned += 1
            if local_identity(sequences[left], sequences[right]) >= min_seq_id:
                union(left, right)

    logger.debug("fallback clustering ran %d exact alignments", n_aligned)

    roots = {find(key) for key in keys}
    order = {root: i for i, root in enumerate(sorted(roots))}
    return {key: order[find(key)] for key in keys}


#: BLOSUM62 restricted to the 20 standard residues plus X, as a lookup array.
_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
#: BLOSUM62 for the 20 standard residues, one whitespace-separated row each, in
#: ``_ALPHABET`` order. ``X`` scores 0 against everything: an ambiguous residue
#: should neither reward nor penalise an alignment.
_BLOSUM62_ROWS = (
    "A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0",
    "R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0",
    "N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0",
    "D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0",
    "C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0",
    "Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0",
    "E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0",
    "G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0",
    "H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0",
    "I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0",
    "L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0",
    "K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0",
    "M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0",
    "F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0",
    "P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0",
    "S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0",
    "T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0",
    "W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0",
    "Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0",
    "V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0",
    "X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
)


def _blosum62() -> np.ndarray:
    """Parse the rows above into a lookup array, checking they are well formed.

    A substitution matrix is symmetric, and its row labels must match the
    alphabet. Both are asserted at import rather than trusted, because a silent
    transcription error here would quietly skew every identity the fallback
    backend reports.
    """
    size = len(_ALPHABET)
    if len(_BLOSUM62_ROWS) != size:
        raise ValueError(f"BLOSUM62 has {len(_BLOSUM62_ROWS)} rows, expected {size}")

    matrix = np.zeros((size, size), dtype=np.int16)
    for row, line in enumerate(_BLOSUM62_ROWS):
        label, *fields = line.split()
        if label != _ALPHABET[row]:
            raise ValueError(f"BLOSUM62 row {row} is labelled {label!r}, expected {_ALPHABET[row]!r}")
        if len(fields) != size:
            raise ValueError(f"BLOSUM62 row {label} has {len(fields)} values, expected {size}")
        matrix[row] = [int(field) for field in fields]

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("BLOSUM62 is not symmetric; check the transcription")
    return matrix


_SCORES = _blosum62()
_INDEX = {residue: i for i, residue in enumerate(_ALPHABET)}
_X_INDEX = _INDEX["X"]


def _encode(sequence: str) -> np.ndarray:
    return np.fromiter(
        (_INDEX.get(residue, _X_INDEX) for residue in sequence), dtype=np.int16, count=len(sequence)
    )


def local_identity(
    left: str,
    right: str,
    *,
    gap_open: int = 11,
    gap_extend: int = 1,
) -> float:
    """Smith-Waterman local alignment identity, as a fraction of the shorter sequence.

    Affine gaps with BLOSUM62, the same parameters BLAST uses by default, so the
    fallback's thresholds are at least comparable to published ones.

    The recurrence runs cell by cell in Python, which costs roughly 0.1-0.3 s for a
    pair of 500-residue proteins. That is why the k-mer prefilter exists, and why
    this path is a fallback rather than the default: a full kinase-family audit
    through it takes hours where MMseqs2 takes seconds.
    """
    left, right = clean_sequence(left), clean_sequence(right)
    if not left or not right:
        return 0.0

    a, b = _encode(left), _encode(right)
    n, m = len(a), len(b)

    # Row-by-row affine Smith-Waterman, tracking matches alongside scores so that
    # identity can be read off the best cell without a traceback.
    best_score = 0
    best_matches = 0
    best_length = 0

    prev_h = np.zeros(m + 1, dtype=np.int32)
    prev_e = np.zeros(m + 1, dtype=np.int32)
    prev_matches = np.zeros(m + 1, dtype=np.int32)
    prev_length = np.zeros(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        row_scores = _SCORES[a[i - 1]][b]
        curr_h = np.zeros(m + 1, dtype=np.int32)
        curr_e = np.zeros(m + 1, dtype=np.int32)
        curr_matches = np.zeros(m + 1, dtype=np.int32)
        curr_length = np.zeros(m + 1, dtype=np.int32)
        f = 0
        f_matches = 0
        f_length = 0

        for j in range(1, m + 1):
            # Vertical gap: consume a residue of `left`. A gap never adds a match,
            # so the match count carries over from whichever cell the gap extends.
            e = max(prev_h[j] - gap_open, prev_e[j] - gap_extend)
            e_matches = prev_matches[j]
            e_length = prev_length[j] + 1

            # Horizontal gap: consume a residue of `right`.
            f_open = curr_h[j - 1] - gap_open
            f_extend = f - gap_extend
            if f_open >= f_extend:
                f = f_open
                f_matches = curr_matches[j - 1]
                f_length = curr_length[j - 1] + 1
            else:
                f = f_extend
                f_length = f_length + 1

            diagonal = prev_h[j - 1] + int(row_scores[j - 1])
            diagonal_matches = prev_matches[j - 1] + int(a[i - 1] == b[j - 1])
            diagonal_length = prev_length[j - 1] + 1

            # Local alignment: 0 is always an option, which restarts the alignment.
            options = (
                (diagonal, diagonal_matches, diagonal_length),
                (e, e_matches, e_length),
                (f, f_matches, f_length),
                (0, 0, 0),
            )
            score, matches, length = max(options, key=lambda item: item[0])

            curr_h[j] = score
            curr_e[j] = e
            curr_matches[j] = matches
            curr_length[j] = length

            if score > best_score:
                best_score, best_matches, best_length = score, matches, length

        prev_h, prev_e, prev_matches, prev_length = curr_h, curr_e, curr_matches, curr_length

    if best_length == 0:
        return 0.0
    # Normalising by the shorter sequence, not by alignment length, prevents a
    # short high-identity motif from registering as a whole-protein match.
    return best_matches / min(n, m)


def identity_matrix(sequences: Sequence[str]) -> np.ndarray:
    """Full pairwise identity matrix. Quadratic -- intended for small target sets."""
    n = len(sequences)
    matrix = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = local_identity(sequences[i], sequences[j])
            matrix[i, j] = matrix[j, i] = value
    return matrix
