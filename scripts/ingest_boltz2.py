"""Verify and ingest Boltz-2 predictions from an external run.

An external run is the easiest place for a result to go quietly wrong: the
predictions come back from another machine, possibly in another order, possibly
covering only part of what was asked for. So this refuses anything it cannot
check, and the checks run before any metric is computed.

What is verified:

* every prediction maps to a row key in the frozen subset
* no row key appears twice
* the subset digest matches the one recorded when the subset was built, so the
  predictions cannot be scored against a subset that has since been rebuilt
* predictions are finite

Partial coverage is accepted and recorded as partial. A model scored on the
complexes that happened to fit in memory has not been scored on the subset, and
the count belongs in the result rather than in a footnote.

Pooled metrics are reported twice: as computed, and reweighted to the test
fold's family composition. The subset over-samples the rare families, so the
unweighted pooled figure is not comparable to the other models' pooled figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.eval.metrics import evaluate
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.hashing import digest_frame_keys
from leakfree_dta.utils.io import read_json, read_table, write_json, write_table

logger = get_logger(__name__)

#: How much of the subset must be scored before a pooled number is worth
#: reporting. Below this the per-family numbers stand on their own and the
#: coverage is the headline.
POOLED_COVERAGE_FLOOR = 0.667


def load_predictions(path: Path) -> pd.DataFrame:
    """Read predictions from a CSV, or from a directory of Boltz-2 outputs.

    Boltz writes one JSON of affinity values per complex, named after the input
    file, which is why the inputs were named by row key in the first place.
    """
    if path.is_file():
        frame = pd.read_csv(path)
        missing = {"row_key", "y_pred"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} is missing columns {sorted(missing)}")
        return frame[["row_key", "y_pred"]]

    records: list[dict[str, object]] = []
    for item in sorted(path.rglob("affinity*.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        value = payload.get("affinity_pred_value", payload.get("affinity_pred"))
        if value is None:
            logger.warning("no affinity value in %s; skipping", item.name)
            continue
        # Boltz names outputs after the input stem; the subset wrote inputs
        # named by row key with "||" replaced by "__".
        stem = item.parent.name if item.parent.name != path.name else item.stem
        records.append({"row_key": stem.replace("affinity_", ""), "y_pred": float(value)})

    if not records:
        raise SystemExit(
            f"no affinity JSON found under {path}. Expected either a CSV with "
            "row_key and y_pred, or a Boltz output directory."
        )
    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--split", default="leakfree")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    subset_dir = paths.EXTERNAL / args.dataset / args.split / "Boltz-2"
    manifest = read_json(subset_dir / "subset_manifest.json")
    subset = read_table(subset_dir / "subset.csv")

    recorded = manifest["subset_digest"]
    actual = digest_frame_keys(subset[schema.ROW_KEY].astype(str).tolist())
    if recorded != actual:
        raise SystemExit(
            f"subset digest mismatch: manifest says {recorded[:12]}..., the CSV "
            f"hashes to {actual[:12]}.... The subset has been rebuilt since the run."
        )

    pred = load_predictions(args.predictions)
    name_to_key = {str(k).replace("||", "__"): str(k) for k in subset[schema.ROW_KEY]}
    pred["row_key"] = pred["row_key"].map(lambda v: name_to_key.get(str(v), str(v)))

    known = set(subset[schema.ROW_KEY].astype(str))
    unknown = sorted(set(pred["row_key"]) - known)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} predictions do not correspond to a subset row, e.g. "
            f"{unknown[:3]}. Refusing to score."
        )
    duplicated = pred["row_key"][pred["row_key"].duplicated()].unique()
    if len(duplicated):
        raise SystemExit(f"{len(duplicated)} row keys predicted more than once, e.g. {duplicated[:3]}")

    pred = pred[np.isfinite(pred["y_pred"])]
    merged = subset.merge(pred, left_on=schema.ROW_KEY, right_on="row_key", how="inner")
    coverage = len(merged) / len(subset)
    logger.info("scored %d of %d subset complexes (%.1f%%)", len(merged), len(subset), 100 * coverage)
    if merged.empty:
        raise SystemExit("no predictions matched the subset")

    y_true = pd.to_numeric(merged[schema.Y], errors="coerce").to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)

    per_family: dict[str, dict] = {}
    for family, chunk in merged.groupby(schema.TARGET_FAMILY, observed=True):
        if len(chunk) < 10:
            logger.warning("family %s has only %d scored rows; reporting but unreliable", family, len(chunk))
        metrics = evaluate(
            pd.to_numeric(chunk[schema.Y], errors="coerce").to_numpy(float),
            chunk["y_pred"].to_numpy(float),
        )
        per_family[str(family)] = {**metrics.to_dict(), "n_scored": len(chunk),
                                   "n_in_subset": int((subset[schema.TARGET_FAMILY] == family).sum())}

    pooled = evaluate(y_true, y_pred)
    weight = manifest["weight_to_fold"]
    # Reweighting a pooled RMSE means recombining the per-family squared errors
    # under the fold's proportions, not averaging the per-family RMSEs.
    weighted_mse = sum(
        weight.get(f, 1.0) * per_family[f]["n_scored"] * per_family[f]["rmse"] ** 2
        for f in per_family
    )
    weighted_n = sum(weight.get(f, 1.0) * per_family[f]["n_scored"] for f in per_family)
    reweighted_rmse = float(np.sqrt(weighted_mse / weighted_n)) if weighted_n else float("nan")

    out = paths.RUNS / args.dataset / args.split / "Boltz-2"
    write_table(
        pd.DataFrame(
            {
                schema.ROW_KEY: merged[schema.ROW_KEY].to_numpy(),
                schema.TARGET_KEY: merged[schema.TARGET_KEY].to_numpy(),
                schema.LIGAND_KEY: merged[schema.LIGAND_KEY].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                schema.TARGET_FAMILY: merged[schema.TARGET_FAMILY].to_numpy(),
            }
        ),
        out / "predictions.parquet",
    )
    result = {
        **pooled.to_dict(),
        "coverage_of_subset": round(coverage, 4),
        "n_scored": len(merged),
        "n_in_subset": len(subset),
        "pooled_is_comparable": False,
        "rmse_reweighted_to_fold": round(reweighted_rmse, 5),
        "per_family": per_family,
        "note": (
            "The subset over-samples rare families; the pooled figure is not "
            "comparable to other models' pooled figures. Use the per-family "
            "numbers, or rmse_reweighted_to_fold."
        ),
    }
    write_json(result, out / "metrics.json")
    write_json(
        {
            "model": {"name": "Boltz-2", "source": "external", "evaluated_on": "frozen subset"},
            "subset": {k: v for k, v in manifest.items() if k != "row_keys"},
            "external_run": read_json(args.run_metadata) if args.run_metadata else None,
            "split": args.split,
        },
        out / "run.json",
    )

    print(f"\nBoltz-2 on the frozen subset -> {out}")
    print(f"  scored        {len(merged):,} of {len(subset):,} ({100 * coverage:.1f}%)")
    for family, metrics in sorted(per_family.items()):
        print(f"  {family:10s} n={metrics['n_scored']:4d}  rmse {metrics['rmse']:.3f}  "
              f"pearson {metrics['pearson']:.3f}")
    print(f"  pooled        rmse {pooled.get('rmse'):.3f} (NOT comparable across models)")
    print(f"  reweighted    rmse {reweighted_rmse:.3f} (to the fold's family mix)")
    if coverage < POOLED_COVERAGE_FLOOR:
        print(f"\n  Coverage is below {POOLED_COVERAGE_FLOOR:.0%}. Report the per-family")
        print("  numbers and the coverage; a pooled figure over whatever fitted in")
        print("  memory is not a result on this subset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
