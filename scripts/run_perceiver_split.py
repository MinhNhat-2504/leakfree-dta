"""GPU wrapper for the published PerceiverCPI CLI on our explicit split.

The upstream architecture/training entry point is unchanged. This wrapper only
sets the split paths and resource settings, suppresses optional TensorBoard event
logging (which requires sandbox-blocked Windows pipes), and disables worker/molecule
caches to avoid sandbox-blocked pipes and excessive host-RAM use.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


class _NullSummaryWriter:
    """No-op replacement for nonessential event logging in restricted Windows."""

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-cutoff", type=int, default=10000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PerceiverCPI GPU run requested, but CUDA is unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    import tensorboardX

    tensorboardX.SummaryWriter = _NullSummaryWriter
    train = args.repo / "train.py"
    sys.path.insert(0, str(args.repo))
    sys.argv = [
        str(train),
        "--data_path",
        str(args.data / "train.csv"),
        "--dataset_type",
        "regression",
        "--smiles_columns",
        "smiles",
        "--target_columns",
        "y",
        "--separate_val_path",
        str(args.data / "valid.csv"),
        "--separate_test_path",
        str(args.data / "test.csv"),
        "--save_dir",
        str(args.output),
        "--save_preds",
        "--batch_size",
        str(args.batch_size),
        "--cache_cutoff",
        str(args.cache_cutoff),
        "--num_workers",
        "0",
        "--no_cache_mol",
    ]
    runpy.run_path(str(train), run_name="__main__")


if __name__ == "__main__":
    main()
