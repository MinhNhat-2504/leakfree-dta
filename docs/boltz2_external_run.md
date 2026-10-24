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

**83 GPU-hours**, measured rather than estimated. A smoke run of three
complexes averaging 505 residues on a Kaggle T4 took 20.1 minutes: 13.9 for the
structure pass and 6.2 for the affinity pass.

The first figure in this document was 35 hours. It counted the structure pass
only. Boltz runs affinity as a second pass over the same inputs, and affinity is
what is being evaluated here, so the original number was low by a factor of
2.4 and every chunk sized from it would have overrun its session.

### Thirty-five hours was wrong, and eighty-three is not a sitting either

No free platform gives you one. Kaggle allows twelve hours per session and
around thirty a week; Colab is shorter and less predictable. The run has to
survive being stopped, which is a property of how the work is divided rather
than something to improvise when a session dies.

```bash
python scripts/chunk_boltz2_subset.py --session-hours 12
```

Chunks are packed by estimated minutes, not by complex count. Lengths run from
219 to 1,306 residues and cost grows faster than length, so equal-sized chunks
would differ several-fold in runtime. Each chunk is held to 75% of the session.

### The 16 GB problem

A T4 raised a CUDA OOM warning at 494 residues, recovering only because the
allocator freed its cache. It will not recover on the long targets, and
`--diffusion_samples` is already at its minimum of 1, so there is no memory
left to give back.

```bash
python scripts/chunk_boltz2_subset.py --session-hours 12 --max-sequence-length 900
```

This yields six chunks totalling 47.7 hours over 499 of the 600 complexes, and
it reports what the filter costs per family, because the cost is not uniform:

| family | in subset | over 900 aa | lost to the filter |
| --- | --: | --: | --: |
| GPCR | 300 | 0 | 0% |
| kinase | 200 | 87 | **43.5%** |
| protease | 100 | 14 | 14% |

**A GPCR result from this is complete; a kinase result is not.** The 113 kinase
complexes that fit are the short ones, and reporting their RMSE as a kinase
figure would be the same unexamined selection this paper is about. GPCR is 85%
of the leakage-free test fold, so the complete part is still the larger part.

The excluded 101 complexes are left for a larger GPU, not removed. The frozen
subset and its digest are unchanged, and `--status` continues to count against
all 600, so the coverage never silently redefines its denominator.

## Running it

```bash
pip install "boltz>=2.0,<3.0"

# One chunk per session; see the chunking section above. All chunks write to
# the same --out_dir, which is what makes the progress check work.
boltz predict data/external/chembl_3fam/leakfree/Boltz-2/chunks/chunk01 \
  --out_dir data/external/chembl_3fam/leakfree/Boltz-2/outputs \
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
python scripts/ingest_boltz2.py \
  --predictions data/external/chembl_3fam/leakfree/Boltz-2/outputs \
  --run-metadata run.json
```

Run it at any point, not only at the end: it accepts whatever is in the output
directory and records the coverage. Doing so after each session is a cheap way
to notice a problem three chunks earlier than you otherwise would.

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
