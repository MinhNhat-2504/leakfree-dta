# v1.0.0 — Journal of Cheminformatics submission

The state of the benchmark as described in the manuscript *How much of our
progress is leakage? A leakage-audited benchmark for drug–target affinity
prediction*.

## What the audit found

Benchmarks the drug–target affinity literature has reported on for a decade let
information cross the train/test boundary. This release measures how much,
through which channel, and what survives once the channels are closed.

**The label noise floor is 0.617 log units**, the pooled within-pair standard
deviation across replicate measurements. No model can be meaningfully scored
below it, and 23.5% of replicate pairs disagree by more than tenfold.

**Lookup leakage falls along the ladder** — the fraction of test rows whose
label is recoverable because the same ligand–target pair appears in training:

| split | test rows | recoverable | protein ≥40% identical in train |
| --- | --: | --: | --: |
| `random` | 145,856 | 14.87% | 100.00% |
| `ligand_scaffold` | 145,659 | 8.41% | 100.00% |
| `ligand_cluster` | 145,666 | 1.42% | 99.97% |
| `protein_identity` | 145,653 | 0.00% | **78.87%** |
| `both_cold` | 45,115 | 0.00% | 0.00% |
| `leakfree` | 4,903 | **0.00%** | **0.00%** |

The `protein_identity` rung is named for the thing it fails to control: it
closes pair lookup completely while leaving 78.87% of test rows with a protein
at least 40% identical to one in training.

**Every re-run model is beaten by a constant.** On the leakage-free split:

| model | RMSE |
| --- | --: |
| GraphDTA | 1.324 |
| DeepDTA | 1.328 |
| PerceiverCPI | 1.328 |
| predicting the training mean | **1.241** |

**The registered shift-robust hypothesis was not supported.** It is reported as
a negative result rather than dropped, which is the reason for registering it
before running it.

## What is in the repository

Extraction query, curation log with per-step row counts, the split index with
specifications and content digests, the frozen Boltz-2 subset definition, and
the re-run attempt log including failures and exclusions.

## What is in the archival deposit

Zenodo [10.5281/zenodo.23605646](https://doi.org/10.5281/zenodo.23605646) holds
what is too large for git: the full per-row fold assignment for every rung, the
audit reports in JSON and Markdown, and per-row predictions for every model.
Each archive is hashed in its `MANIFEST.md`.

## Reproducing it

The benchmark is reproducible from a ChEMBL release number and the committed
query. Applying a split manifest to a dataset whose digest does not match is
refused rather than warned about, because the failure it prevents is numbers
that look plausible and mean nothing.

```bash
python scripts/reproduce_all.py --dataset chembl_3fam
```

## Licensing

Code is MIT. Data in the Zenodo deposit is derived from ChEMBL and is
CC BY-SA 3.0; no BindingDB records are redistributed, only aggregate
concordance statistics.
