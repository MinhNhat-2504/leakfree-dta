"""Reading and writing the pipeline's intermediate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a table, dispatching on suffix.

    Parquet is the pipeline's native format; CSV is accepted so that small
    hand-curated inputs (target family maps, exclusion lists) stay reviewable in a
    diff.
    """
    path = Path(path)
    suffixes = {s.lower() for s in path.suffixes}
    if ".parquet" in suffixes or path.suffix.lower() == ".pq":
        return pd.read_parquet(path)
    if ".csv" in suffixes:
        return pd.read_csv(path)
    if ".tsv" in suffixes or ".tab" in suffixes:
        return pd.read_csv(path, sep="\t")
    if ".json" in suffixes:
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path.name}")


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a table, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = {s.lower() for s in path.suffixes}
    if ".parquet" in suffixes or path.suffix.lower() == ".pq":
        df.to_parquet(path, index=index)
    elif ".csv" in suffixes:
        df.to_csv(path, index=index)
    elif ".tsv" in suffixes:
        df.to_csv(path, sep="\t", index=index)
    else:
        raise ValueError(f"Unsupported table format: {path.name}")
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")
    return path


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def write_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
