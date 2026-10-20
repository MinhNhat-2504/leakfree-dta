"""Deterministic hashing.

Splits in this project are assigned by hashing a key rather than by drawing from a
random number generator. Two consequences matter for the paper:

* A split is reproducible from the key alone -- no RNG state to carry around, and
  no dependence on NumPy's generator version.
* Adding new compounds to the dataset does not reshuffle the existing ones. A
  compound's fold is a function of its own identity, so the benchmark can grow
  without invalidating published numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any


def stable_hash(value: object, *, salt: str = "") -> int:
    """A 64-bit hash of ``value`` that is stable across processes and versions.

    Python's builtin ``hash`` is salted per interpreter run for strings, so it
    cannot be used for anything persisted.
    """
    payload = f"{salt}\x00{value}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def hash_unit(value: object, *, salt: str = "") -> float:
    """Map ``value`` deterministically into ``[0, 1)``.

    This is the primitive behind every hash-based split: assign a key to the test
    fold when its unit hash falls below the test fraction.
    """
    return stable_hash(value, salt=salt) / 2**64


def hash_fold(value: object, n_folds: int, *, salt: str = "") -> int:
    """Assign ``value`` to one of ``n_folds`` folds, deterministically."""
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    return stable_hash(value, salt=salt) % n_folds


def digest_frame_keys(keys: Iterable[str]) -> str:
    """A content digest over a collection of row keys, order-independent.

    Written into every split manifest so a reviewer can verify that the splits
    they downloaded are the splits the paper reports on.
    """
    hasher = hashlib.sha256()
    for key in sorted(keys):
        hasher.update(key.encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


def digest_config(config: dict[str, Any]) -> str:
    """A digest over a configuration mapping, insensitive to key order."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
