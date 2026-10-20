"""Logging setup shared by the CLI and the library."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> None:
    """Install a single stderr handler. Idempotent.

    The level comes from ``LEAKFREE_LOG_LEVEL`` when not given explicitly, so
    notebooks can turn up verbosity without editing code.
    """
    global _CONFIGURED
    if level is None:
        level = os.environ.get("LEAKFREE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("leakfree_dta")
    root.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    else:
        for handler in root.handlers:
            handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """A logger under the package namespace."""
    setup_logging()
    if name.startswith("leakfree_dta"):
        return logging.getLogger(name)
    return logging.getLogger(f"leakfree_dta.{name}")
