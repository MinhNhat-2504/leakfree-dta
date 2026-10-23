# Running Boltz-2 on the frozen subset

Boltz-2 predicts a structure per complex, so scoring the full 4,903-row
leakage-free test fold is out of reach on the hardware this project otherwise
uses. The evaluation is therefore on a frozen 600-complex subset, defined once
and archived, and run wherever a suitable GPU is available.

Nothing here is required for the rest of the paper. If the run never happens,
Boltz-2 stays an attempt-log row with a stated reason, exactly as MolTrans does.

## What the subset is, and what it is not

```bash
python scripts/build_boltz2_subset.py
```

Writes to `data/external/chembl_3fam/leakfree/Boltz-2/`:

| file | contents |
| --- | --- |
| `subset_manifest.json` | row keys, digests, family composition, reweighting factors |
| `subset.csv` | the rows themselves, with sequences and SMILES |
| `inputs/<row_key>.yaml` | one Boltz-2 input per complex |

600 complexes over 114 targets and 523 distinct ligands: 300 GPCR, 200 kinase,
100 protease.

**It over-samples the rare families on purpose, and is not representative of
the test fold.** The fold is 85% GPCR — 4,159 rows against 586 kinase and 158
protease — so a proportional sample of 600 would contain about 19 proteases.
That is enough to compute a number and not enough for it to mean anything.

The consequence has to be carried into any result. **Per-family metrics from
this subset are comparable to the other models' per-family metrics. A pooled
metric is not**, because the pooling weights differ. `weight_to_fold` in the
manifest gives the per-family factor for reweighting a pooled figure back to the
fold's composition:

```
gpcr 1.697   kinase 0.359   protease 0.193
```

**No sequence-length cap was applied.** Five targets exceed 1,000 residues and
are the most expensive complexes in the set; 73 of the 600 rows involve them.
Dropping them in advance would bias the subset towards short proteins, and
length correlates with family here — capping at 1,000 residues would remove 38%
of the kinase rows. If the hardware cannot fit one, record it as a failure with
a reason rather than quietly removing it.

## Compute

Roughly **35 GPU-hours** on one modern accelerator, against 20 if every complex
cost the 2-minute base figure. The difference is the long kinase targets, and
the estimate scales length to the power 1.5, which is a rough stand-in for
attention over residues. Treat it as an order of magnitude.

Boltz-2 wants considerably more VRAM than the 8 GB laptop GPU this project
develops on. Options, in rough order of convenience:

- a rented A100 or H100 for a day
- an institutional cluster allocation
- Kaggle or Colab, splitting the subset into chunks that fit a session

The run is embarrassingly parallel: each YAML is independent, so it can be split
across sessions or machines and recombined, provided every prediction keeps its
row key.

## Running it

```bash
pip install "boltz>=2.0,<3.0"

boltz predict data/external/chembl_3fam/leakfree/Boltz-2/inputs \
  --out_dir boltz_out \
  --use_msa_server \
  --diffusion_samples 1
```

`--use_msa_server` sends sequences to the public MSA server. If that is not
acceptable for your setting, generate MSAs locally and point Boltz at them; the
sequences here are public UniProt entries, but the decision is yours to make
rather than one this document should make silently.

Record, for the attempt log: the Boltz version, the model checkpoint, the
hardware, whether MSAs were generated locally or remotely, the wall-clock time,
and any complex that failed with its reason.

## Bringing the results back

```bash
python scripts/ingest_boltz2.py --predictions boltz_out --run-metadata run.json
```

The ingester refuses anything it cannot verify. It checks that each prediction
maps to a row key in the frozen subset, that no row appears twice, and that the
subset digest matches the one recorded when the subset was built. Partial runs
are accepted and recorded as partial: the number of complexes scored is part of
the result, not a footnote.

It then writes aligned predictions, per-family metrics, and a pooled metric
marked with the reweighting applied, under
`results/runs/chembl_3fam/leakfree/Boltz-2/`.

## What to claim

Report per-family RMSE and correlation, the number of complexes scored out of
600, and any failures. Do not report a pooled number without stating the
reweighting, and do not compare a pooled Boltz-2 figure against the other
models' pooled figures as though they were computed on the same population.

If fewer than about two thirds of the subset completes, report the per-family
numbers that survive and say the coverage was partial. A model evaluated on the
complexes that happened to fit in memory is not evaluated on the subset.
