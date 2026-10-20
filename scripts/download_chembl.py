"""Download the ChEMBL SQLite release over several connections at once.

The EBI mirror throttles a single connection hard -- about 90 KB/s from here,
which puts the 5.8 GB SQLite tarball at eighteen hours. The throttle is per
connection rather than per client, so requesting byte ranges in parallel gets
most of the available bandwidth back; eight streams measured roughly seven times
a single one.

Resumable by design, because a multi-hour download will be interrupted. Each
stream writes its own part file and skips the bytes it already has, so re-running
the script continues rather than restarting.

    python scripts/download_chembl.py --out D:/chembl_dl
    python scripts/download_chembl.py --out D:/chembl_dl --streams 16
    python scripts/download_chembl.py --out D:/chembl_dl --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
DEFAULT_FILE = "chembl_37_sqlite.tar.gz"
CHECKSUMS = "checksums.txt"
USER_AGENT = "leakfree-dta/0.1 (research use)"
CHUNK = 1 << 20


class Stream(threading.Thread):
    """One byte-range request, resuming from whatever its part file holds."""

    def __init__(self, url: str, part: Path, first_byte: int, last_byte: int) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.part = part
        # Named first_byte/last_byte, not start/end: `self.start` would shadow
        # Thread.start and the thread would never run.
        self.first_byte = first_byte
        self.last_byte = last_byte  # inclusive
        self.downloaded = part.stat().st_size if part.exists() else 0
        self.error: str | None = None

    @property
    def total(self) -> int:
        return self.last_byte - self.first_byte + 1

    @property
    def done(self) -> bool:
        return self.downloaded >= self.total

    def run(self) -> None:
        attempt = 0
        while not self.done and attempt < 20:
            attempt += 1
            first = self.first_byte + self.downloaded
            try:
                request = Request(
                    self.url,
                    headers={"User-Agent": USER_AGENT, "Range": f"bytes={first}-{self.last_byte}"},
                )
                with urlopen(request, timeout=60) as response, self.part.open("ab") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        self.downloaded += len(block)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                # The mirror drops connections regularly on long transfers; that is
                # expected, so back off briefly and resume rather than giving up.
                time.sleep(min(2 * attempt, 30))
        if not self.done:
            self.error = self.error or "incomplete after 20 attempts"


def content_length(url: str) -> int:
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def expected_checksum(filename: str) -> str | None:
    """The release's published SHA-256 for ``filename``, if it lists one."""
    try:
        request = Request(f"{BASE_URL}/{CHECKSUMS}", headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"  could not fetch {CHECKSUMS}: {exc}")
        return None

    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and Path(fields[-1]).name == filename:
            return fields[0]
    return None


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    size = path.stat().st_size
    read = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 22):
            hasher.update(block)
            read += len(block)
            print(f"\r  hashing {100 * read / size:5.1f}%", end="", flush=True)
    print()
    return hasher.hexdigest()


def download(url: str, target: Path, n_streams: int) -> bool:
    size = content_length(url)
    print(f"  {size / 1e9:.2f} GB over {n_streams} streams")

    span = size // n_streams
    parts = [target.with_suffix(target.suffix + f".part{i}") for i in range(n_streams)]
    streams = [
        Stream(
            url,
            parts[i],
            i * span,
            (size - 1) if i == n_streams - 1 else ((i + 1) * span - 1),
        )
        for i in range(n_streams)
    ]

    already = sum(stream.downloaded for stream in streams)
    if already:
        print(f"  resuming: {already / 1e9:.2f} GB already on disk")

    started = time.monotonic()
    for stream in streams:
        stream.start()

    while any(stream.is_alive() for stream in streams):
        time.sleep(5)
        got = sum(stream.downloaded for stream in streams)
        elapsed = time.monotonic() - started
        rate = (got - already) / elapsed if elapsed > 0 else 0
        remaining = (size - got) / rate / 60 if rate > 0 else float("inf")
        print(
            f"\r  {got / 1e9:6.2f} / {size / 1e9:.2f} GB  "
            f"{rate / 1e6:5.2f} MB/s  eta {remaining:5.0f} min   ",
            end="",
            flush=True,
        )
    print()

    failed = [(i, stream.error) for i, stream in enumerate(streams) if stream.error and not stream.done]
    if failed:
        print("  streams failed; re-run to resume:")
        for index, error in failed:
            print(f"    stream {index}: {error}")
        return False

    print("  joining parts")
    with target.open("wb") as out:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, CHUNK)
    for part in parts:
        part.unlink()

    if target.stat().st_size != size:
        print(f"  size mismatch: got {target.stat().st_size}, expected {size}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to download into")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true", help="only check an existing file")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / args.file
    url = f"{BASE_URL}/{args.file}"

    if not args.verify_only:
        if target.exists():
            print(f"{target} already exists; verifying instead")
        else:
            print(f"downloading {args.file}")
            if not download(url, target, args.streams):
                return 1

    print("verifying checksum")
    expected = expected_checksum(args.file)
    if expected is None:
        print("  no published checksum found; skipping verification")
        print(f"\n{target} ({target.stat().st_size / 1e9:.2f} GB)")
        return 0

    actual = sha256_of(target)
    if actual == expected:
        print(f"  sha256 ok: {actual[:16]}...")
    else:
        # A corrupt 5.8 GB archive that extracts partially would produce a
        # silently incomplete benchmark, so this is a hard failure.
        print(f"  CHECKSUM MISMATCH\n    expected {expected}\n    actual   {actual}")
        print("  delete the file and re-run")
        return 1

    print(f"\n{target} ({target.stat().st_size / 1e9:.2f} GB)")
    print("\nnext:")
    print(f"  tar -xzf {target} -C {out_dir}")
    print("  python scripts/extract_chembl.py --db <path to chembl_37.db>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
