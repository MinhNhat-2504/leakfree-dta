"""Compare Boltz-2 against every other model on the rows Boltz-2 has scored.

Boltz-2 is evaluated on a 600-complex subset, and the run arrives one chunk per
session, so at any moment it has scored some fraction of those. Every other
model was scored on the full 4,903-row leakage-free fold. Putting those numbers
side by side compares two different populations, and the difference between
them is larger than the difference between the models.

So this restricts every model to the row keys Boltz-2 has actually scored and
recomputes from per-row predictions. It then runs the paired cluster bootstrap
the rest of the paper uses, over protein groups rather than rows, because a
point estimate on a hundred rows from twenty-eight proteins looks far more
decisive than it is.

Run it after each chunk is ingested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta import paths
from leakfree_dta.data import schema
from leakfree_dta.eval.bootstrap import compare_models
from leakfree_dta.eval.metrics import evaluate
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.io import write_json

logger = get_logger(__name__)

#: Fewer groups than this and the paired interval is too wide to conclude
#: anything; the comparison is still printed, with the count beside it.
MIN_GROUPS_FOR_INFERENCE = 20


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="chembl_3fam")
    parser.add_argument("--split", default="leakfree")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    setup_logging("INFO")
    runs = paths.RUNS / args.dataset / args.split
    boltz_path = runs / "Boltz-2" / "predictions.parquet"
    if not boltz_path.exists():
        raise SystemExit(f"no Boltz-2 predictions at {boltz_path}; ingest a chunk first")

    boltz = pd.read_parquet(boltz_path).sort_values(schema.ROW_KEY).reset_index(drop=True)
    keys = set(boltz[schema.ROW_KEY])
    groups = boltz[schema.TARGET_KEY].to_numpy()
    n_groups = len(np.unique(groups))
    y_true = boltz["y_true"].to_numpy(float)

    by_family = boltz[schema.TARGET_FAMILY].value_counts().to_dict()
    print(f"\nBoltz-2 has scored {len(boltz)} rows over {n_groups} protein groups")
    print(f"family mix: {by_family}")
    if n_groups < MIN_GROUPS_FOR_INFERENCE:
        logger.warning(
            "only %d protein groups; the bootstrap resamples groups, so intervals "
            "will be very wide and a null result means little",
            n_groups,
        )

    rows = []
    for path in sorted(runs.glob("*/predictions.parquet")):
        name = path.parent.name
        if name == "Boltz-2":
            continue
        other = pd.read_parquet(path)
        other = other[other[schema.ROW_KEY].isin(keys)]
        if len(other) != len(boltz):
            logger.warning("%s covers %d of %d rows; skipping", name, len(other), len(boltz))
            continue
        other = other.sort_values(schema.ROW_KEY).reset_index(drop=True)
        if not (other[schema.ROW_KEY].to_numpy() == boltz[schema.ROW_KEY].to_numpy()).all():
            raise SystemExit(f"{name}: row keys do not align after sorting")

        metrics = evaluate(y_true, other["y_pred"].to_numpy(float)).to_dict()
        paired = compare_models(
            y_true,
            boltz["y_pred"].to_numpy(float),
            other["y_pred"].to_numpy(float),
            rmse,
            metric_name="rmse",
            groups=groups,
            n_bootstrap=args.n_bootstrap,
            higher_is_better=False,
        )
        rows.append(
            {
                "model": name,
                "rmse": metrics["rmse"],
                "pearson": metrics["pearson"],
                "boltz_minus_model": paired["difference"],
                "ci_lower": paired["ci_lower"],
                "ci_upper": paired["ci_upper"],
                "p_value": paired["p_value"],
                "boltz_better": bool(paired["significant"] and paired["difference"] < 0),
            }
        )

    boltz_metrics = evaluate(y_true, boltz["y_pred"].to_numpy(float)).to_dict()
    print(f"\n{'model':18s} {'RMSE':>7s} {'Pearson':>8s} {'vs Boltz-2':>11s} {'95% CI':>18s} {'p':>7s}")
    print("-" * 74)
    print(f"{'Boltz-2':18s} {boltz_metrics['rmse']:7.3f} {boltz_metrics['pearson']:8.3f}")
    for row in sorted(rows, key=lambda r: r["rmse"]):
        mark = "  *" if row["boltz_better"] else ""
        print(
            f"{row['model']:18s} {row['rmse']:7.3f} {row['pearson']:8.3f} "
            f"{row['boltz_minus_model']:+11.3f} "
            f"[{row['ci_lower']:+6.3f},{row['ci_upper']:+6.3f}] {row['p_value']:7.3f}{mark}"
        )
    beaten = [r["model"] for r in rows if r["boltz_better"]]
    print(f"\n* Boltz-2 significantly better than {len(beaten)} of {len(rows)}: {beaten}")
    print(
        "Models without a star are not distinguishable from Boltz-2 on these rows. "
        "That is a statement about the sample size, not about the models."
    )

    out = runs / "Boltz-2" / "comparison_on_scored_rows.json"
    write_json(
        {
            "n_rows": len(boltz),
            "n_protein_groups": int(n_groups),
            "family_mix": {str(k): int(v) for k, v in by_family.items()},
            "boltz2": boltz_metrics,
            "comparisons": rows,
            "note": (
                "Every model restricted to the row keys Boltz-2 has scored. "
                "Partial coverage of the 600-complex subset; not the full fold."
            ),
        },
        out,
    )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
