"""Set up and run the pipeline inside a Kaggle or Colab notebook.

The deep models and the external re-runs are what need a GPU, and a laptop with
8 GB is tight for several of them. This script makes a hosted notebook behave like
a local checkout: it installs the package, points ``LEAKFREE_ROOT`` at writable
storage, and verifies the environment before any long job starts.

Usage, in one notebook cell:

    !git clone -q https://github.com/MinhNhat-2504/leakfree-dta /kaggle/working/leakfree-dta
    %run /kaggle/working/leakfree-dta/scripts/kaggle_setup.py

Then drive the pipeline with ``leakfree`` as usual, or import the package.

What this script will not do is install MMseqs2. Kaggle images do not carry it and
conda installs are slow and unreliable there, so protein clustering in a hosted
notebook uses the approximate fallback. Build the split manifests locally with
MMseqs2 installed, commit them, and let the notebook load them -- which is exactly
what split manifests are for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_CANDIDATES = (
    Path("/kaggle/working/leakfree-dta"),
    Path("/content/leakfree-dta"),
    Path.cwd(),
)

PACKAGES = ("rdkit", "pyarrow")


def find_repo() -> Path:
    for candidate in REPO_CANDIDATES:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise SystemExit(
        "could not locate the repository. Clone it first, or run this script from "
        "inside the checkout."
    )


def pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("  $", " ".join(command[3:]))
    subprocess.run(command, check=True)


def main() -> int:
    repo = find_repo()
    print(f"repository: {repo}")

    print("\ninstalling dependencies")
    missing = []
    for package in PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        pip_install(*missing)
    else:
        print("  all present")

    print("\ninstalling leakfree-dta (editable)")
    pip_install("-e", str(repo))

    # Writable storage. Kaggle's input directory is read-only, so the pipeline's
    # data and results directories have to live under working storage.
    root = repo
    os.environ["LEAKFREE_ROOT"] = str(root)
    print(f"\nLEAKFREE_ROOT = {root}")

    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))

    from leakfree_dta import paths
    from leakfree_dta.logging import setup_logging

    setup_logging("INFO")
    paths.ensure_dirs()

    print("\nenvironment:")
    subprocess.run([sys.executable, "-m", "leakfree_dta.cli", "info"], check=False)

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            print(
                f"\nGPU: {device.name}, {device.total_memory / 1e9:.1f} GB. "
                "Enough for the MLP baselines and the DRO ablation."
            )
        else:
            print(
                "\nNo GPU visible. Enable an accelerator in the notebook settings "
                "before training the deep models."
            )
    except ImportError:
        print("\nPyTorch is not installed; only the audit and classical baselines will run.")

    print(
        "\nReady. Try:\n"
        "  !leakfree demo --fast\n"
        "  !leakfree audit --dataset synthetic\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
