# leakfree-dta

**How much of our progress is leakage?** A leakage-audited benchmark and a
shift-robust baseline for drug–target affinity prediction.

Benchmarks that the drug–target affinity literature has reported on for a decade
contain information that crosses the train/test boundary. This repository measures
how much, through which channel, and what is left of reported performance once the
channels are closed.

---

## The claim, in two rows of one table

Running the pipeline on its calibration dataset — a synthetic benchmark whose
leakage is planted by construction — gives:

| split | recoverable by lookup | protein ≥40% identical in train | what the split constrains |
| --- | --: | --: | --- |
| `random` | **24.6%** | 100% | nothing |
| `ligand_scaffold` | **20.5%** | 100% | Bemis–Murcko scaffold |
| `ligand_cluster` | **0.4%** | 100% | ECFP4 cluster |
| `protein_identity` | 0.0% | **20.0%** | target accession |
| `protein_cluster30` | 0.0% | **0.0%** | 30% sequence identity |
| `temporal` | 18.4% | 100% | publication year |

Two rows carry the paper, one per axis.

**`ligand_scaffold`.** Scaffold splitting — the field's standard "hard" split —
removes almost none of the recoverable leakage, falling only from 24.6% to 20.5%.
The cause is a property of the representation, not of the data: a homologous
series (same core, a linker one or two carbons longer) has *different* InChIKeys
and *different* Bemis–Murcko scaffolds, so a scaffold split cheerfully separates
its members — but *identical* ECFP4 fingerprints, so no fingerprint-based model
can tell which side of the boundary it is looking at. The split achieves perfect
scaffold separation (0% scaffold overlap) and still leaks a fifth of its test set.

**`protein_identity`.** Holding out every target accession leaves 20% of test rows
with a training protein at 40% sequence identity or above. Accession-level
holdout is not protein-level holdout; clustering at 30% identity is what closes
the channel.

A test set built either way is, for the model, partly a copy of its training set.

---

## What this repository provides

**A leakage audit** (`leakfree_dta.audit`) that reports, per channel, what fraction
of a test set is reachable from training data. Channels are kept separate on
purpose: a benchmark can be clean on the ligand axis and badly compromised on the
protein axis, and a single summary number would hide exactly that.

**A shortcut probe** — the sharpest instrument here, and it needs no deep model.
Fit predictors that are *structurally incapable* of modelling interaction (the
per-target mean, the per-ligand mean, their additive combination) and see how far
they get. If `target_mean` reaches Pearson 0.70 where published models report 0.85,
then the modelling contribution is the gap, not the headline.

**Split manifests** (`leakfree_dta.splits`) keyed by content digest, so a reader
can apply the paper's splits to their own copy of the data and be told immediately
if that copy differs from the one the paper reports on.

The per-row manifests are 42 MB per rung, so the repository carries
`data/splits/<dataset>/_index.json` instead: 11 KB with each rung's
specification, fold counts and digests. Splits are a deterministic function of
the curated data and that specification, so the index is enough to rebuild them
and confirm you got the same answer. `scripts/export_split_artifacts.py` also
writes a single 17 MB table holding every rung's assignment, for the archival
deposit.

**A shift-robust baseline** (`leakfree_dta.models.robust`): group DRO over protein
clusters with a Mondrian conformal head. It is *not* claimed to be the most
accurate model — the worst-group objective trades average accuracy for uniformity,
and that trade is the point. It is claimed to have the lowest worst-group error and
the only intervals whose coverage holds *within* each protein group rather than
only on average.

**Cluster-bootstrap confidence intervals** (`leakfree_dta.eval`). The standard
bootstrap resamples rows, which assumes rows are independent. On chemically
redundant data they are not, so published intervals are too narrow — often enough
to make a reported improvement indistinguishable from its baseline once corrected.

---

## Install

```bash
git clone https://github.com/MinhNhat-2504/leakfree-dta
cd leakfree-dta
pip install -e ".[dev]"          # core pipeline
pip install -e ".[all]"          # plus torch, xgboost, ESM-2
```

Python 3.10+. The audit and split stages need only NumPy, pandas, SciPy,
scikit-learn and RDKit; PyTorch is required solely for the deep models, and the
pipeline reports an absent deep model as a missing table row rather than failing.

**MMseqs2 is required for publication-grade numbers.** Without it, protein
clustering falls back to an approximate k-mer + Smith–Waterman path. The fallback
exists so the pipeline and its tests run anywhere; it is far slower and not
identical to MMseqs2. Every audit report and split manifest records which backend
produced it, and `leakfree info` says which is active.

The binary is located in this order: the `LEAKFREE_MMSEQS` environment variable,
then `PATH`, then a few conventional install locations.

Pointing at it explicitly is the recommended route on Windows. The official
Windows build is a Cygwin executable surrounded by roughly 250 BusyBox shims —
`bash`, `awk`, `sed`, `find` and so on — which MMseqs2's shell workflows need
beside the binary. Putting that directory on `PATH` would shadow the system's own
tools, so don't.

```bash
# Linux / macOS
conda install -c bioconda mmseqs2        # or: brew install mmseqs2
```

```powershell
# Windows: unpack the release, then point at the binary rather than adding to PATH
# https://github.com/soedinglab/MMseqs2/releases -> mmseqs-win64.zip
& "$HOME\tools\mmseqs\bin\busybox.exe" --install -s "$HOME\tools\mmseqs\bin"
setx LEAKFREE_MMSEQS "$HOME\tools\mmseqs\bin\mmseqs.exe"
```

Confirm with `leakfree info`, which should report `identity backend  mmseqs2`.

Verify the installation:

```bash
leakfree info          # environment, dependencies, identity backend
leakfree demo          # the whole pipeline on synthetic data, about a minute
```

---

## Running it

```bash
# 0. Fetch ChEMBL. The EBI mirror throttles per connection, so this uses parallel
#    byte ranges; it is resumable, and the published checksum is verified.
python scripts/download_chembl.py --out D:/chembl_dl
python scripts/extract_chembl.py --tarball D:/chembl_dl/chembl_37_sqlite.tar.gz

# 1. Curate a source into the canonical schema. Writes two tables: one with
#    replicates (the audit needs them) and one aggregated (the models need that).
leakfree curate --source chembl --input data/raw/chembl_3fam.parquet --dataset chembl_3fam

# 2. Audit the data itself -- duplicates, conflicting labels, the noise floor.
leakfree audit --dataset chembl_3fam

# 3. Build every split on the ladder.
leakfree split --dataset chembl_3fam --ladder

# 4. Audit a split: what crosses the boundary, overall and per target family.
leakfree audit --dataset chembl_3fam --split leakfree

# 5. Train, compare, report.
leakfree train    --dataset chembl_3fam --split leakfree --model shift_robust
leakfree evaluate --dataset chembl_3fam --split leakfree
leakfree report   --dataset chembl_3fam
```

Stages are separate commands because the early ones are slow and the later ones
are iterated on constantly. `leakfree report` only reads what is already on disk,
so regenerating it never changes a number.

---

## How the audit is validated

An audit tool that reports "4.2% of test rows leak" is a measuring instrument, and
an instrument means nothing until it is calibrated against a known quantity. So
`leakfree_dta.data.synthetic` builds a dataset where the answer is known by
construction:

- *n* pairs measured twice, with a controlled disagreement.
- *n* ligands that are homologues of another ligand by design — distinct InChIKeys,
  distinct scaffolds, identical fingerprints.
- *n* targets that are 92%-identical paralogues of another target.
- Affinities generated from an explicit `target + ligand + interaction + noise`
  model, so the correlation a shortcut predictor *should* reach is known in closed
  form.

The test suite asserts that each check recovers its planted quantity. The shortcut
probe, for instance, reaches Pearson 0.853 against an analytic ceiling of 0.890 —
close enough to trust, and not above it, which would mean the probe was seeing
something it should not.

```bash
pytest                      # 88 tests
pytest -m "not torch"       # skip the deep-model tests
```

---

## Repository layout

```
src/leakfree_dta/
  data/        canonical schema, curation, source loaders, the calibration fixture
  chem/        RDKit standardisation, fingerprints, Tanimoto, clustering
  prot/        sequences, MMseqs2 + fallback identity, family assignment
  audit/       within-dataset checks, cross-split checks, shortcut probes, reports
  splits/      the three axes, their composition, frozen manifests
  models/      degenerate baselines, classical learners, group DRO + conformal
  eval/        metrics, cluster bootstrap, per-group, risk-coverage, progress curve
  report/      tables and figures, assembled from artifacts on disk
configs/       dataset, split ladder, and experiment definitions
paper/         outline, target journals, the argument
```

---

## Scope: three target families

The benchmark covers kinases, GPCRs and proteases. Not one family, and not all of
ChEMBL.

They were chosen because they fail differently, which is the finding rather than
the setup. Kinases leak mainly through the **protein** axis: ATP-site conservation
means a held-out accession often has a near-identical paralogue in training. GPCRs
leak through the **ligand** axis: medicinal chemistry programmes explore tight
analogue series. Proteases are dominated by **label inconsistency**: mixed
substrate-like and allosteric chemotypes measured in heterogeneous assays.

If one splitting rule were sufficient, all three would behave the same. Showing
that they do not is why the scope is three families and why the paper is one paper
rather than three.

---

## Honest limitations

**Exchangeability fails under the strict splits.** Conformal prediction's coverage
guarantee assumes exchangeable calibration and test data, which a temporal or
protein-cluster split deliberately breaks. The formal guarantee therefore does not
hold on those splits. What is reported is *empirical* coverage per split — the
honest claim being that conformal intervals degrade gracefully under shift where a
Gaussian error model degrades catastrophically, and the degradation is measured.

**Pooling IC50 with Ki and Kd is an approximation.** Results are reported with and
without IC50 so the cost is visible rather than defended.

**The re-evaluation runs other people's code as published.** Only the data loading
and the split are replaced. Re-tuning someone else's model and reporting a lower
number would measure our tuning effort, not their contribution. Models that would
not run are listed in the attempt log with a stated reason — dropping them would
bias the re-measured trend upward, since the hardest models to reproduce are
disproportionately the oldest.

**The fallback identity backend is approximate.** See Install.

---

## License

MIT. See [LICENSE](LICENSE).
