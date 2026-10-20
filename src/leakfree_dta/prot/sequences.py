"""Target sequences: fetching, caching, and FASTA interchange.

Sequences are cached on disk by accession because the audit reads them many times
-- once per identity clustering, once per split axis, once per per-group metric --
and because a benchmark whose target sequences change under it is not reproducible.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from leakfree_dta import paths
from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import read_json, write_json

logger = get_logger(__name__)

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
_ACCESSION_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


def is_uniprot_accession(value: str) -> bool:
    """Whether ``value`` looks like a UniProt accession.

    Source tables mix accessions with ChEMBL target IDs and gene symbols; mapping
    them to one namespace is a precondition for detecting that two "different"
    targets are the same protein.
    """
    return bool(_ACCESSION_RE.match(value.strip().upper()))


def _cache_path() -> Path:
    return paths.cache_dir("sequences") / "uniprot.json"


def load_cache() -> dict[str, str]:
    path = _cache_path()
    if not path.exists():
        return {}
    cached = read_json(path)
    return {str(key): str(value) for key, value in cached.items()}


def save_cache(cache: Mapping[str, str]) -> Path:
    return write_json(dict(cache), _cache_path())


def fetch_sequences(
    accessions: Iterable[str],
    *,
    use_cache: bool = True,
    sleep: float = 0.2,
    timeout: float = 30.0,
) -> dict[str, str]:
    """Fetch sequences from UniProt, one accession at a time, with caching.

    Deliberately serial and rate-limited: the target sets here are hundreds of
    proteins, not millions, and a polite client avoids the throttling that makes
    a pipeline non-reproducible. Failures are logged and omitted rather than
    raised, and the caller is expected to report how many targets were resolved.
    """
    wanted = [accession.strip().upper() for accession in accessions if accession]
    wanted = list(dict.fromkeys(wanted))
    cache = load_cache() if use_cache else {}
    missing = [accession for accession in wanted if accession not in cache]

    if missing:
        logger.info("fetching %d sequences from UniProt (%d cached)", len(missing), len(wanted) - len(missing))

    for position, accession in enumerate(missing, start=1):
        try:
            request = Request(
                UNIPROT_FASTA_URL.format(accession=accession),
                headers={"User-Agent": "leakfree-dta/0.1 (research use)"},
            )
            with urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning("UniProt fetch failed for %s: %s", accession, exc)
            continue

        records = parse_fasta(text)
        if records:
            cache[accession] = next(iter(records.values()))
        if position % 50 == 0:
            logger.info("  ... %d/%d", position, len(missing))
        time.sleep(sleep)

    if missing and use_cache:
        save_cache(cache)

    resolved = {accession: cache[accession] for accession in wanted if accession in cache}
    n_unresolved = len(wanted) - len(resolved)
    if n_unresolved:
        logger.warning("%d/%d accessions unresolved", n_unresolved, len(wanted))
    return resolved


def parse_fasta(text: str) -> dict[str, str]:
    """Parse FASTA text into ``{header_id: sequence}``.

    UniProt headers look like ``sp|P00533|EGFR_HUMAN ...``; the accession is taken
    from the second pipe-delimited field when present, else the first token.
    """
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(chunks)
            header = line[1:]
            fields = header.split("|")
            identifier = fields[1] if len(fields) >= 3 else header.split()[0]
            chunks = []
        elif identifier is not None:
            chunks.append(line.upper())

    if identifier is not None:
        records[identifier] = "".join(chunks)
    return records


def write_fasta(sequences: Mapping[str, str], path: str | Path, *, wrap: int = 60) -> Path:
    """Write sequences to FASTA, for handing to MMseqs2 or BLAST."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in sequences.items():
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), wrap):
                handle.write(sequence[start : start + wrap] + "\n")
    return path


def clean_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, and map rare residues to ``X``.

    ``U`` (selenocysteine) and ``O`` (pyrrolysine) break substitution matrices
    that most alignment tools ship with, and ``B``/``Z`` are ambiguity codes.
    """
    cleaned = re.sub(r"[^A-Za-z]", "", sequence).upper()
    return re.sub(r"[BZJUO]", "X", cleaned)
