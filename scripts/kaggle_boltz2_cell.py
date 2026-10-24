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
CHUNK = "smoke"  # smoke (3 complexes) | chunk01 | chunk02 | chunk03 | chunk04
CHUNKS_DIR = None  # auto-detected under /kaggle/input; set a path to override
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

# Say something before the install, which takes 5-15 minutes and used to run
# with -q: a cell that prints nothing for a quarter of an hour is
# indistinguishable from a cell that has hung.
print("=" * 60, flush=True)
print(f"Boltz-2 session, chunk={CHUNK}", flush=True)
print("=" * 60, flush=True)

print(f"notebook python {platform.python_version()}", flush=True)

# boltz declares requires-python ">=3.10,<3.13" and Kaggle's image is on 3.13,
# so boltz cannot be installed into the notebook's interpreter at all. pip
# responds by walking backwards through old boltz releases looking for one that
# fits, and those pull dm-tree==0.1.8, which has no wheel for 3.13 and fails to
# build. The dm-tree error is a symptom; the Python version is the cause.
#
# So boltz gets its own 3.11 environment, and this cell drives it as a
# subprocess. The venv lives in /kaggle/temp rather than /kaggle/working
# because everything under working is saved as the notebook's output, and a
# CUDA torch install is several gigabytes of it.
VENV = Path("/kaggle/temp/boltz-venv")
BOLTZ = VENV / "bin" / "boltz"


def run(cmd, label):
    print(f"-> {label}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        raise SystemExit(f"{label} failed ({result.returncode})")
    return result


if BOLTZ.exists():
    print(f"boltz env already present at {VENV}", flush=True)
else:
    run([sys.executable, "-m", "pip", "install", "-q", "uv"], "install uv")
    run([sys.executable, "-m", "uv", "venv", str(VENV), "--python", "3.11"],
        "create python 3.11 venv")
    print("   installing boltz + CUDA torch, several GB, 5-15 min", flush=True)
    run([sys.executable, "-m", "uv", "pip", "install", "--python", str(VENV / "bin" / "python"),
         "boltz>=2.0,<3.0"], "install boltz")
    print(f"   ready in {(time.time() - started) / 60:.1f} min", flush=True)

# Probe the venv once, here rather than after the run: if its torch cannot see
# the GPU, that is worth knowing before nine hours of CPU inference, and the
# same answer is what the attempt log needs afterwards.
probe = subprocess.run(
    [
        str(VENV / "bin" / "python"),
        "-c",
        "import json, platform, torch, importlib.metadata as m;"
        "print(json.dumps({'boltz': m.version('boltz'), 'torch': torch.__version__,"
        " 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),"
        " 'python': platform.python_version()}))",
    ],
    capture_output=True,
    text=True,
)
try:
    boltz_env = json.loads(probe.stdout.strip())
except ValueError:
    boltz_env = {"probe_failed": probe.stderr[-1000:]}
print(f"boltz env: {boltz_env}", flush=True)
if boltz_env.get("cuda_available") is False:
    raise SystemExit(
        "the boltz venv's torch cannot see the GPU. Check the accelerator is on, "
        "and do not run a chunk on CPU -- it will not finish."
    )

import torch


def find_chunks_root():
    """Locate the directory holding chunk01..chunk04 under /kaggle/input.

    Kaggle has moved input paths around -- a dataset has appeared at both
    /kaggle/input/<slug> and /kaggle/input/datasets/<user>/<slug> -- so the
    path is discovered rather than written down and corrected later.
    """
    if CHUNKS_DIR:
        return Path(CHUNKS_DIR)
    for candidate in sorted(Path("/kaggle/input").rglob("chunk01")):
        if candidate.is_dir() and any(candidate.glob("*.yaml")):
            return candidate.parent
    raise SystemExit(
        "no chunk01/ with YAML files found under /kaggle/input. Add the "
        "chunks dataset as an input, or set CHUNKS_DIR explicitly."
    )


chunks_root = find_chunks_root()
chunk_dir = chunks_root / (CHUNK if CHUNK != "smoke" else "chunk01")
print(f"chunks root  {chunks_root}")

# Earlier sessions' outputs, wherever they were attached.
previous = [p.parent for p in Path("/kaggle/input").rglob("affinity_*.json")]

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
done = predicted_stems(*previous, out)
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
    str(BOLTZ), "predict", str(staged),
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
    "boltz_environment": boltz_env,
    "notebook_python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "msa": "public MSA server (--use_msa_server)",
    "diffusion_samples": 1,
    "attempted": len(todo),
    "scored": len(scored),
    "failed": missing,
    "boltz_exit_code": result.returncode,
    "elapsed_seconds": round(elapsed, 1),
}
(out / f"run_{CHUNK}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"\nscored {len(scored)} of {len(todo)} attempted, {elapsed / 3600:.2f} h")
if missing:
    print(f"failed ({len(missing)}): {missing[:5]}")
    print("Record these as failures with a reason; do not quietly drop them.")
print(f"\nOutput in {out}. Save Version, then add it as a dataset input next session.")
