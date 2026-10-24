"""Kaggle notebook cell for one Boltz-2 session. Paste this into a GPU notebook.

One session runs one chunk. The pieces that matter are the ones that survive the
session ending badly, because it will: Kaggle stops at twelve hours whether or
not the work is done.

Resume works by reading what is already predicted. Attach the previous
sessions' output as a second input dataset and this skips those complexes, so a
session killed at hour eleven costs eleven hours of compute and nothing else.
No state file, because a state file can disagree with the predictions on disk
and then you are debugging bookkeeping instead of running the model.

Only affinity JSONs are kept. Boltz also writes structures, which for 600
complexes would be gigabytes that the evaluation never reads -- the manuscript
scores affinity. Keeping them would spend the 20 GB output budget on files
nobody opens.

Set CHUNK below, run, then "Save Version" so the output is downloadable.
"""

# ruff: noqa: E402 - the settings block stays at the top where it is edited;
# imports follow it because this is pasted into a notebook cell, not imported.

# ---------------------------------------------------------------- settings
CHUNK = "chunk01"  # chunk01 | chunk02 | chunk03 | chunk04, or "smoke" for 3
CHUNKS_DIR = "/kaggle/input/leakfree-boltz2-chunks/chunks"
PREVIOUS = "/kaggle/input/leakfree-boltz2-outputs"  # earlier sessions, if any
OUT = "/kaggle/working/outputs"
SMOKE_N = 3

# ---------------------------------------------------------------- setup
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

started = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "boltz>=2.0,<3.0"], check=True)

import torch

chunk_dir = Path(CHUNKS_DIR) / (CHUNK if CHUNK != "smoke" else "chunk01")
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)


def predicted_stems(*roots):
    """Row keys that already have an affinity prediction, from any root."""
    done = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for item in root.rglob("affinity*.json"):
            stem = item.parent.name
            if stem in (root.name, "predictions"):
                stem = item.stem.replace("affinity_", "")
            done.add(stem)
    return done


wanted = sorted(p.stem for p in chunk_dir.glob("*.yaml"))
if CHUNK == "smoke":
    wanted = wanted[:SMOKE_N]
done = predicted_stems(PREVIOUS, out)
todo = [s for s in wanted if s not in done]

print(f"chunk        {CHUNK}")
print(f"in chunk     {len(wanted)}")
print(f"already done {len(wanted) - len(todo)}")
print(f"to run       {len(todo)}")
print(f"GPU          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if torch.cuda.is_available():
    print(f"VRAM         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if not todo:
    print("\nNothing left in this chunk. Pick the next one.")
    raise SystemExit(0)

# Boltz takes a directory, so stage exactly what is left.
staged = Path("/kaggle/working/_todo")
if staged.exists():
    shutil.rmtree(staged)
staged.mkdir(parents=True)
for stem in todo:
    shutil.copy2(chunk_dir / f"{stem}.yaml", staged / f"{stem}.yaml")

# ---------------------------------------------------------------- run
cmd = [
    "boltz", "predict", str(staged),
    "--out_dir", str(out),
    "--use_msa_server",
    "--diffusion_samples", "1",
    "--output_format", "mmcif",
]
print("\n" + " ".join(cmd) + "\n", flush=True)
result = subprocess.run(cmd)
print(f"\nboltz exit code {result.returncode}", flush=True)

# ---------------------------------------------------------------- keep only affinity
kept = 0
affinity_dir = out / "affinity"
affinity_dir.mkdir(exist_ok=True)
for item in out.rglob("affinity*.json"):
    if item.parent == affinity_dir:
        continue
    stem = item.parent.name
    if stem in (out.name, "predictions"):
        stem = item.stem.replace("affinity_", "")
    shutil.copy2(item, affinity_dir / f"affinity_{stem}.json")
    kept += 1

for child in out.iterdir():
    if child != affinity_dir:
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
shutil.rmtree(staged, ignore_errors=True)

# ---------------------------------------------------------------- record it
elapsed = time.time() - started
scored = sorted(p.stem.replace("affinity_", "") for p in affinity_dir.glob("affinity_*.json"))
missing = [s for s in todo if s not in scored]

metadata = {
    "chunk": CHUNK,
    "boltz_version": subprocess.run(
        ["pip", "show", "boltz"], capture_output=True, text=True
    ).stdout.split("Version: ")[1].split("\n")[0]
    if shutil.which("pip")
    else "unknown",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "msa": "public MSA server (--use_msa_server)",
    "diffusion_samples": 1,
    "attempted": len(todo),
    "scored": len(scored),
    "failed": missing,
    "boltz_exit_code": result.returncode,
    "elapsed_seconds": round(elapsed, 1),
    "python": platform.python_version(),
}
(out / f"run_{CHUNK}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"\nscored {len(scored)} of {len(todo)} attempted, {elapsed / 3600:.2f} h")
if missing:
    print(f"failed ({len(missing)}): {missing[:5]}")
    print("Record these as failures with a reason; do not quietly drop them.")
print(f"\nOutput in {out}. Save Version, then add it as a dataset input next session.")
