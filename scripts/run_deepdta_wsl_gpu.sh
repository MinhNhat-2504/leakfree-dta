#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/d/Project/AI Engineer/leakfree-dta"
SRC="$ROOT/data/external/chembl_3fam/leakfree/DeepDTA/repo/source"
DATA="$ROOT/data/external/chembl_3fam/leakfree/DeepDTA/data"
OUT="$ROOT/data/external/chembl_3fam/leakfree/DeepDTA/output-split-cuda"

mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_FORCE_GPU_ALLOW_GROWTH=true
cd "$SRC"
exec >> "$OUT/console.log" 2>&1
echo "DeepDTA start $(date -Is)"
exec /home/minh_nhat/.venvs/deepdta/bin/python \
  "$ROOT/scripts/run_deepdta_split.py" \
  --repo "$ROOT/data/external/chembl_3fam/leakfree/DeepDTA/repo" \
  --data "$ROOT/data/external/chembl_3fam/leakfree/GraphDTA/data" \
  --output "$OUT" \
  --batch-size 64 --epochs 100 --patience 15
