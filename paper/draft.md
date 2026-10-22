# How much of our progress is leakage? A leakage-audited benchmark for drug–target affinity

**Draft status.** The ChEMBL training runs, leakage ladder, model comparison,
and robustness evaluation are complete. The external published-model
reproduction track was not used for the headline claims because it is not yet
complete. All numerical claims below are tied to generated result artifacts.

---

## Abstract

Benchmarks for drug–target affinity prediction have supported a decade of
reported improvement. We audit what those benchmarks actually contain. Building
a curated extract from ChEMBL 37 covering three target families — kinases,
GPCRs and proteases — we measure, channel by channel, how much information
crosses the train/test boundary under each splitting rule the field uses, and
how much reported performance survives when every channel is closed.

Three findings are independent of any model. First, the labels themselves set a
floor: across 153,598 pairs measured more than once, the pooled within-pair
standard deviation is **0.617 log units**, and 23.5% of replicated pairs
disagree by more than tenfold. Any RMSE below that, on ChEMBL-derived data, is
smaller than the disagreement among the labels being scored against. Second, pooling IC50 with Ki — standard practice —
carries a systematic offset of **−0.569 log units**, measured directly on 35,957
pairs holding both. Third, Bemis–Murcko scaffold splitting [@bemis_1996], the field's usual
"hard" split, closes almost none of the recoverable leakage, because homologous
series receive different scaffolds while remaining fingerprint-identical.

On the ladder itself: the random split most of this literature reports on leaves
14.9% of test rows recoverable by lookup and 100% of them targeting a protein
already seen in training, where a predictor that ignores the protein entirely
reaches Pearson 0.520. Scaffold splitting halves the recoverable fraction to
8.4%; fingerprint clustering takes it to 1.4%. Holding out target accessions —
the usual "protein-cold" split — still leaves 78.9% of test rows with a training
protein above 40% sequence identity.

On the leakage-free split, the strongest point estimate is the classical
`xgboost_ecfp` baseline (RMSE **1.240**, Pearson **0.192**), essentially tied
with the additive shortcut (RMSE **1.241**, Pearson **0.000**) on absolute
error. DeepDTA reaches Pearson **0.177** but RMSE **1.293**. The proposed
`shift_robust` model does not support the registered robustness claim: its
RMSE is **1.490**, its worst-family RMSE is **1.645**, and its nominal 90%
interval covers only **54.5%** of test rows. The robustness objective and
conformal head therefore require revision before this can be presented as a
reliable baseline.

We release the audit as a package, the splits as content-addressed manifests,
and the per-row predictions of every model we re-ran.

---

## 1. Introduction

AI-designed components are increasingly entering clinical development. The gap
between activity and outcome has moved reviewer attention from accuracy to
reliability: whether a reported number means what it appears to mean.

Drug–target affinity (DTA) prediction is a good place to ask that question,
because it has a decade of leaderboard history and a small number of heavily
reused benchmarks. The literature reports steady improvement on them. We ask how
much of that improvement survives an evaluation in which no information crosses
the train/test boundary.

The question is not new. A recent audit of widely used biomolecular benchmarks
found systematic inconsistencies across Polaris, Therapeutics Data Commons,
MoleculeNet and several DTI collections, including protein overlap between
splits. Work on DTI benchmarking has argued that transductive evaluation lets
near-optimal scores be reached by shortcut rather than by modelling. Both
establish that the problem exists. Neither quantifies its effect on reported
performance, and neither provides a corrected benchmark others can use
[@de_la_fuente_2023].

This paper makes three contributions.

**A leakage audit that reports channels separately.** We decompose leakage into
pair, ligand, protein, temporal and label channels, and measure each. Keeping
them apart is not presentational: a benchmark can be clean on one axis and badly
compromised on another, and we find that which channel dominates depends on the
target family. A single leakage number would hide exactly that.

**A shortcut probe.** We fit predictors that are structurally incapable of
modelling ligand–protein interaction — the per-target mean, the per-ligand mean,
and their additive combination — and measure how far they get. This bounds what
a benchmark rewards without any interaction modelling, and unlike arguments from
how a split was constructed, it applies to any split, including ones whose
construction is undocumented.

**A benchmark and a measured robustness boundary.** We release leakage-free
splits as content-addressed manifests and test a worst-group/conformal baseline.
The baseline does not meet its registered coverage or worst-group objectives;
reporting that failure prevents an attractive robustness claim from becoming an
unmeasured assumption.

### 1.1 Scope: three families, chosen because they fail differently

We restrict to kinases, GPCRs and proteases. Not one family, because a single
family cannot show that no one splitting rule suffices; not all of ChEMBL,
because the three were chosen for contrast.

Kinases share a highly conserved ATP site, so many targets are near-identical in
the region ligands bind: leakage should run mainly through the **protein** axis.
GPCR medicinal chemistry explores tight analogue series, so leakage should run
through the **ligand** axis. Proteases mix substrate-like and allosteric
chemotypes measured in heterogeneous assays, so the dominant problem should be
**label inconsistency** rather than overlap.

If one splitting rule were sufficient, all three would behave alike. Section 6
tests that.

---

## 2. Data

### 2.1 Source

ChEMBL 37, extracted with the query released alongside this paper [@gaulton_2024].
ChEMBL is the
primary source rather than one of the packaged benchmarks because it carries two
fields they lack: `document_year`, without which no temporal split is possible,
and `assay_id` with a confidence score, without which replicate disagreement
cannot be traced to the assays that disagree. Davis and KIBA ship without dates,
which is why no honest prospective evaluation has been performed on them.

The query keeps single-protein human targets at ChEMBL confidence 8 or above,
with Ki, Kd, IC50 or EC50 in nM, excluding activities ChEMBL itself flags as
suspect or duplicate. It returned **1,330,694 activities** over **921 targets**,
**542,586 compounds** and **98,123 assays**.

### 2.2 Target family assignment, and a heuristic we removed

Families are assigned from ChEMBL's protein classification path, with a
committed override file for targets the automated rule misses. Of the final
extract, 93.4% of activities are assigned from the database annotation and 6.6%
from the override file.

We initially included a fallback for targets with no annotation: a sequence
motif heuristic claiming a kinase when at least two of three catalytic motifs
were present. It assigned 41 targets. Checking those against their own ChEMBL
names, **29 are not kinases** — apolipoprotein B-100, utrophin, tissue
alpha-L-fucosidase, NADPH oxidase 5, and a dozen histone-modifying enzymes. A
71% false-positive rate.

We report this because of why it looked safe. Against 2,000 random 450-residue
sequences the rule fires on 0.4%, which reads as a conservative test. Real
proteins are not random: `D[FLIV]G` alone matches 20% of random sequences and
more of real ones, and `HRD[LIVMFYWC]` occurs across enzyme families that are
not kinases. **Validating a sequence heuristic against random sequences gives
false confidence.** The heuristic is disabled; the 11,469 activities (0.85%) it
contributed are dropped.

### 2.3 Curation

Each step is logged with its row count and released with the benchmark, so a
reader who disagrees with a choice can see its size.

| step | rows after | removed |
| --- | --: | --: |
| query output | 1,330,694 | — |
| drop missing / implausible / unknown-type affinity | 1,330,694 | 0 |
| drop censored (`>`, `<`) | 960,842 | 27.79% |
| standardise structures | 960,842 | 0 |
| drop non-drug-like | 950,418 | 1.09% |
| drop targets without sequence | 950,418 | 0 |
| scope to three families | 950,418 | 0 |
| aggregate replicates (median) | 728,235 | 23.38% |

Two choices deserve defending.

**Censored measurements are dropped**, costing 27.8% of rows. A `>10 µM` record
is a bound, not a value; keeping it biases RMSE in a direction that depends on
the model, which makes model comparison meaningless. Papers that retain them for
classification are answering a different question.

**Replicates are aggregated by median, not mean**, because the replicate
distribution has heavy tails: one pair spans 6.97 log units. A single mis-keyed
measurement moves a mean and not a median.

The audit runs on the pre-aggregation table and the models train on the
post-aggregation one, so the noise floor in Section 4 is measured on the data the
models are actually scored against.

### 2.4 The curated benchmark

**728,235 unique pairs** over **459,457 ligands**, **905 targets** and **173,639
Bemis–Murcko scaffolds**, spanning **1976–2025**.

| family | pairs |
| --- | --: |
| kinase | 348,490 |
| GPCR | 264,855 |
| protease | 114,890 |

| affinity type | pairs |
| --- | --: |
| pIC50 | 429,298 |
| pKi | 218,399 |
| pEC50 | 60,069 |
| pKd | 20,469 |

---

## 3. The audit

### 3.1 Channels

Leakage is decomposed into channels, each measured as the fraction of test rows
affected.

**Pair.** The same (ligand, target) pair on both sides, or a ligand at Tanimoto
≥ 0.90 to a training ligand *measured against the same target*. The second is
the sharper test and the one we report as the headline. Asking whether a similar
compound was seen *somewhere* is routinely true of 100% of a test set and says
little; asking whether a near-identical compound was measured *against this same
protein* determines whether the label is recoverable by lookup. We report the
RMSE that lookup achieves, so the leakage is quantified in the units of the
metric it inflates.

**Ligand.** Shared InChIKey, shared skeleton, shared Bemis–Murcko scaffold, and
nearest-neighbour Tanimoto to the training set.

**Protein.** Shared accession, and maximum sequence identity to any training
target. The second is a commonly overlooked channel: holding out an
accession means little when a 90%-identical paralogue remains in training.

**Temporal.** Test rows published no later than the newest training row.

**Label.** Replicate disagreement and affinity-type mixing. Not leakage, but it
bounds what any model can achieve, and a reported RMSE below that bound means
something other than accuracy.

### 3.2 The shortcut probe

For each split we fit four predictors that cannot represent interaction: a global
constant, the per-target training mean, the per-ligand training mean, and their
additive combination fitted by alternating least squares. Their performance
bounds what the benchmark rewards without interaction modelling.

The probe is reported in the main results table rather than an appendix. A
reader can skip an appendix.

### 3.3 Calibration of the audit

An audit is a measuring instrument and means nothing until calibrated against a
known quantity. We therefore built a synthetic benchmark whose leakage is planted
by construction: pairs measured twice with a controlled disagreement, ligands
that are homologues of another ligand by design, targets that are 92%-identical
paralogues, and affinities generated from an explicit
`target + ligand + interaction + noise` model so the correlation a shortcut
predictor *should* reach is known in closed form.

On that fixture the additive-bias probe reaches Pearson **0.853** against an
analytic ceiling of **0.890** — close enough to trust, and not above it, which
would mean the probe was seeing something it should not. The test suite asserts
each check recovers its planted quantity.

The fixture also reproduces the ladder's central comparison. Recoverable
leakage: 24.6% under a random split, 20.5% under a scaffold split, 0.4% under a
fingerprint-cluster split. The scaffold split achieves *perfect* scaffold
separation and still leaks a fifth of its test set, because homologous series
have different InChIKeys and different scaffolds but identical ECFP4
fingerprints. On the protein axis: holding out every accession leaves 20.0% of
test rows with a training protein at ≥40% identity; clustering at 30% identity
drops it to 0.0%.

### 3.4 Sequence identity backend

Protein identity is computed with MMseqs2. The pure-Python Smith–Waterman
fallback costs ~0.3 s per pair of 400-residue proteins, putting an
all-against-all comparison of a few thousand targets at 20–700 hours; MMseqs2
does it in seconds. The two agree to within 0.002 on planted identities of 0.95,
0.80, 0.60 and 0.45, and every report records which backend produced it.

---

## 4. Label quality: the floor beneath every reported RMSE

Measured on the pre-aggregation table, over **153,598 pairs with more than one
measurement**:

| quantity | value |
| --- | --: |
| pooled within-pair standard deviation | **0.617 log units** |
| pairs disagreeing by > 1 log unit | **23.5%** (36,131) |
| largest within-pair range | 6.97 log units |
| maximum replicates for one pair | 198 |

The pooled standard deviation is a floor on the RMSE any model can achieve on
*this* data. A model reporting 0.5 on a test set drawn from it is reporting a
number smaller than the disagreement among the labels it is scored against.

The qualifier matters and is easy to drop. 0.617 is this extract's floor, not a
universal one; the low RMSEs often quoted in this literature are mostly on Davis
and KIBA, which have their own noise characteristics and would need their own
measurement. The comparison we can make is the one within this benchmark, and
there the gap runs the other way: no model we tested comes near the floor. On
the leakage-free split every one sits between 1.24 and 1.33, roughly double it.
The labels are noisy and the models are further from them than the noise
explains.

Pooling affinity types onto one p-scale is standard practice and standard
criticism. We measure its cost directly on the 35,957 pairs carrying more than
one type:

| comparison | mean offset (log units) |
| --- | --: |
| pIC50 − pKi | **−0.569** |
| pEC50 − pKd | −0.633 |
| pEC50 − pKi | −0.284 |
| pEC50 − pIC50 | −0.232 |
| pIC50 − pKd | −0.124 |
| pKd − pKi | +0.056 |

pIC50 and pKi differ by more than half a log unit on the same pair. All results
are reported with and without IC50.

### 4.1 Concentration and redundancy

The benchmark is not a uniform sample. The top decile of targets holds **59.8%**
of all measurements (Gini 0.752); the most-measured target has 29,976 and the
median has 252, while 99 targets have fewer than ten. An overall metric is
mostly a metric for a handful of targets, which is why every result below is
also reported per family and worst-group.

Chemically, a sample of 20,000 ligands collapses into 13,737 clusters at
Tanimoto distance 0.4 — an effective-sample-size ratio of 0.687. Confidence
intervals are therefore computed by bootstrapping groups, not rows.

---

## 5. Splits, models and evaluation

### 5.1 The split ladder

Three axes composed independently: ligand (random / scaffold / fingerprint
cluster), protein (random / accession / identity cluster at 30% or 50%), and
temporal (none / year cutoff). Each rung of the ladder adds exactly one
constraint to the rung above, so the drop between adjacent rungs is attributable
to the single channel the new constraint closes. A paper reporting only "random"
and "hard" splits cannot make that attribution.

A row is placed only where every constrained axis agrees; where they disagree it
is **excluded**, and the excluded fraction is reported. Under a two-axis split a
row whose ligand is in a training cluster and whose protein is in a test cluster
can go nowhere: putting it in training reveals a test protein and putting it in
test reveals a training ligand. That discarded fraction is the honest price of a
leakage-free evaluation.

Folds are assigned by hashing the group label rather than by shuffling, so a
group's fold depends only on its own identity. The benchmark can therefore grow
without invalidating published numbers.

Splits are released as manifests keyed on `row_key`, with a content digest of
each fold and of the dataset. A reader applies the split to their own copy and is
told immediately if that copy differs from ours — a near-universal failure mode
when "the same" ChEMBL extract is rebuilt a year later.

All nine rungs, over the 728,235-pair benchmark:

| split | train | valid | test | excluded |
| --- | --: | --: | --: | --: |
| `random` | 509,988 | 72,391 | 145,856 | 0 (0.0%) |
| `ligand_scaffold` | 509,752 | 72,824 | 145,659 | 0 (0.0%) |
| `ligand_cluster` | 509,742 | 72,827 | 145,666 | 0 (0.0%) |
| `protein_identity` | 508,929 | 73,653 | 145,653 | 0 (0.0%) |
| `protein_cluster30` | 453,046 | 81,237 | 193,952 | 0 (0.0%) |
| `protein_cluster50` | 506,711 | 74,460 | 147,064 | 0 (0.0%) |
| `both_cold` | 224,655 | 18,357 | 45,115 | 440,108 (60.4%) |
| `temporal` | 499,922 | 76,172 | 107,527 | 44,614 (6.1%) |
| `leakfree` | 195,322 | 915 | 4,903 | 527,095 (72.4%) |

Single-axis rungs exclude nothing, as they must. The cost appears when axes
compose: `both_cold` discards 60.4% of the data and `leakfree` 72.4%, leaving
4,903 test rows. That is the honest price of closing every channel at once on
this benchmark, and it is a result in itself — a dataset of 728,235 measurements
supports a test set of five thousand once nothing is allowed to leak.

The small `leakfree` fold is a real constraint on what can be concluded from it.
Model comparisons on that rung carry wide intervals, and we report `both_cold`
alongside it, which closes both structural channels while retaining 45,115 test
rows and differs only in dropping the temporal constraint.

### 5.2 Models

Degenerate predictors (global mean, target mean, ligand mean, additive bias),
classical learners (ridge and gradient boosting on ECFP4 plus protein k-mer
features), a DeepDTA reimplementation of the published architecture
[@ozturk_2018], and the tested shift-robust baseline.

The degenerate predictors appear in the main table beside the deep models. This
is the paper's single most important editorial choice.

### 5.3 Re-running published models

Inclusion and exclusion criteria were fixed before any model was run and are
committed in the repository. Six models are targeted, spanning 2018–2025.
Failures are reported with a reason and a log tail rather than omitted: the
models hardest to reproduce are disproportionately the oldest, so dropping them
would bias the re-measured trend upward at precisely the end that anchors it.

Only the data and the split are replaced; architecture, optimiser and
hyperparameters stay as published. Any departure is declared. For DeepDTA, six
literal compatibility patches were needed — the TensorFlow 1 session API, the
Keras `set_session`, `BatchNormalization` and `merge` imports — each a mechanical
rename forced by framework evolution, none touching the model, all recorded.
Patches apply exact-match-once, so an upstream change stops the run rather than
landing on the wrong line.

Boltz-2 is run on a stratified subset rather than the full benchmark: it predicts
a structure per complex at minutes of GPU time each. Its `coverage_scope` is
recorded so no figure can silently mix subset and full-benchmark numbers.

### 5.4 Evaluation protocol

Every headline number carries a 95% interval from a **cluster bootstrap over protein
groups**, 1,000 resamples. The standard bootstrap resamples rows, which assumes
independence; on chemically redundant data it understates uncertainty, and
published intervals are correspondingly too narrow. Model comparisons bootstrap
the *difference* on paired resamples rather than checking whether individual
intervals overlap.

Metrics: RMSE, MAE, Pearson, Spearman, R² against the test mean (not squared
Pearson, which flatters a biased model), concordance index for comparability with
the literature, and enrichment factor and BEDROC for screening. All reported
overall, macro-averaged across families, and worst-group.

### 5.5 The shift-robust baseline

Group DRO over protein clusters with a Mondrian conformal head [@sagawa_2020;
@romano_2020]. Not claimed to
be the most accurate model — the worst-group objective trades average accuracy
for uniformity, and the trade is the point. Claimed to have the lowest
worst-group error, intervals whose coverage holds within each protein group
rather than only on average, and a better risk–coverage curve.

Conformal's guarantee assumes exchangeability, which a protein-cluster or
temporal split deliberately breaks. We report empirical coverage per split rather
than claiming the theorem applies.

---

## 6. Results

### 6.1 Label quality

Complete; see Section 4.

### 6.2 The leakage ladder

**Table 1.** Fraction of test rows affected, by channel, for each rung.
"Recoverable" is the pair axis: the same pair, or a ligand at Tanimoto ≥ 0.90
measured against the same target, present in training. The other columns are
context, not leakage in themselves.

| split | test rows | **recoverable** | protein ≥40% | same ligand | near-dup ligand | same scaffold | best shortcut (Pearson) |
| --- | --: | --: | --: | --: | --: | --: | --- |
| `random` | 145,856 | **14.87%** | 100.00% | 46.77% | 10.40% | 84.17% | `ligand_mean` 0.520 |
| `ligand_scaffold` | 145,659 | **8.41%** | 100.00% | 0.00% | 2.09% | 0.00% | `target_mean` 0.368 |
| `ligand_cluster` | 145,666 | **1.42%** | 99.97% | 0.00% | 0.40% | 52.23% | `target_mean` 0.385 |
| `protein_identity` | 145,653 | 0.00% | **78.87%** | 48.32% | 9.11% | 60.28% | `ligand_mean` 0.451 |
| `protein_cluster30` | 193,952 | 0.00% | 0.00% | 25.28% | 5.54% | 36.72% | `ligand_mean` 0.387 |
| `protein_cluster50` | 147,064 | 0.00% | 37.59% | 31.44% | 6.30% | 43.04% | `ligand_mean` 0.428 |
| `both_cold` | 45,115 | 0.00% | 0.00% | 0.00% | 0.03% | 8.62% | `global_mean` 0.000 |
| `temporal` | 107,527 | 1.05% | 99.38% | 3.86% | 0.92% | 18.53% | `target_mean` 0.188 |
| `leakfree` | 4,903 | **0.00%** | 0.00% | 0.00% | 0.02% | 8.83% | `global_mean` 0.000 |

Four readings.

**The random split, which most of this literature reports on, leaves 14.9% of
test rows recoverable by lookup and 100% of them targeting a protein already
seen in training.** It is a transductive benchmark described as a predictive one.

**Scaffold splitting removes about half the recoverable leakage, and fingerprint
clustering removes most of the rest**: 14.87% → 8.41% → 1.42%. Our synthetic
fixture predicted a starker contrast (24.6% → 20.5% → 0.4%), overstating how
little scaffold splitting achieves, because its ligands were deliberately built
as homologous series. On real chemistry the scaffold split is worth something —
it halves the problem — and still leaves six times more recoverable leakage than
clustering does. We report the weaker real figure, not the fixture's.

**Holding out accessions is not a protein-cold split.** Under
`protein_identity`, 78.9% of test rows still have a training protein at ≥40%
sequence identity. The channel only closes under identity clustering, and at 50%
identity 37.6% remains, as it must: clustering at 50% permits neighbours between
40% and 50%.

**The shortcut probe is the uncomfortable column.** On the random split a
predictor that sees only the ligand and ignores the protein entirely reaches
Pearson 0.520. On `protein_identity` it still reaches 0.451. Any model's
reported correlation on such a split has to be read against that floor. On
`both_cold` and `leakfree` every shortcut collapses to zero, because no test
target or ligand appears in training at all.

A residual 8.6–8.8% scaffold overlap persists on the two strictest rungs. This is
not leftover leakage but two different criteria: ligands are clustered by
fingerprint, and a Bemis–Murcko scaffold can span several fingerprint clusters.

#### 6.2.1 Per-family channels: the pre-registered hypothesis, partly refuted

We registered before measuring that the dominant channel would differ by family:
protein identity for kinases, ligand similarity for GPCRs, label inconsistency
for proteases.

| family | protein ≥40% under accession holdout | near-duplicate ligands, random split | replicate conflict rate | noise floor |
| --- | --: | --: | --: | --: |
| kinase | **84.2%** | 22.2% | 20.5% | 0.559 |
| GPCR | 74.6% | 24.3% | 27.2% | 0.674 |
| protease | 76.8% | **46.6%** | **28.8%** | **0.733** |

Two thirds of the hypothesis survives and one third does not.

Kinases are the most exposed on the protein axis, as predicted, but the margin is
narrow: 84.2% against 74.6% and 76.8%. Accession holdout fails for every family,
not distinctively for kinases, and the conserved ATP site is a weaker explanation
than we expected.

Proteases are worst on label quality, as predicted — a 0.733 log-unit noise floor
against 0.559 for kinases, and the highest conflict rate.

The GPCR prediction is wrong. We expected GPCRs to dominate the ligand axis
through tight analogue series; proteases do, and by a wide margin — 46.6% of
protease test rows have a near-duplicate training ligand, against 24.3% for
GPCRs. Proteases turn out to be the family where chemistry and labels are both
worst, while GPCRs are unexceptional on every axis we measured.

The broader claim the hypothesis was built to support does hold: no single
splitting rule is sufficient, because the families do not rank the same way on
different channels. But the specific family-to-channel assignment we expected was
only partly right, and we report it as registered rather than rewriting it to
match.

### 6.3 Inflation across the ladder

Table 3 and Figure 2 report Pearson correlation for every completed model on
every rung. The headline pattern is a collapse under simultaneous coldness.
ERM-MLP falls from **0.828** on `random` to **0.125** on `leakfree`, DeepDTA
from **0.789** to **0.177**, and XGBoost from **0.761** to **0.192**. The
additive shortcut falls from **0.474** to **0.000**. This is not evidence that
all predictive signal disappears: the strict setting changes both the chemical
and target distributions, and the remaining correlations are small relative to
the leakage-rich split. The complete generated table is
`results/tables/chembl_3fam/table3_inflation.md`.

### 6.4 Models on the leakage-free split

Table 2 includes all ten registered models, including the degenerate predictors.
It uses the registered cluster bootstrap over **114 target groups** with
**1,000 resamples**. The best RMSE is XGBoost (**1.240**, 95% interval
**[1.160, 1.325]**), followed by the
additive baseline (**1.241**). DeepDTA is not competitive on this split
(RMSE **1.293**, Pearson **0.177**), while the MLP variants and `shift_robust`
are worse than the additive predictor on RMSE. The paired comparison against
`additive_bias` is reported in the companion comparisons table; the important
negative result is that no deep model improves absolute error on the
leakage-free benchmark.

### 6.5 The shift-robust baseline

The registered robustness hypothesis is **not supported**. On the
leakage-free test set, `shift_robust` has RMSE **1.490**, compared with
**1.449** for ERM-MLP and **1.464** for Group-DRO-MLP. Its worst-family RMSE
is **1.645**, versus **1.695** for ERM-MLP and **1.646** for Group-DRO-MLP.
The conformal interval has mean width **2.182** log units but covers only
**54.5%** of rows at the nominal 90% level; coverage is **64.2%** for kinases,
**53.5%** for GPCRs, and **47.5%** for proteases. Selective prediction also
does not rescue the result: the risk–coverage area is **1.468**, compared with
**1.440** for Group-DRO-MLP. These results make the baseline a useful failed
ablation, not a validated uncertainty method.

### 6.6 Progress over time

We drop the reported-versus-re-measured progress curve. The external published
model re-runs are not complete and the available results are not comparable
enough to justify a publication-year trend. The omission is intentional rather
than a missing analysis; the attempt-log machinery remains in the supplement.

---

## 7. Discussion

The results support a narrower claim than “the field has made no progress.”
On this ChEMBL extract, a large fraction of the apparent model advantage on a
random split disappears when ligand and protein overlap are closed. The
additive predictor reaches Pearson **0.474** on the random split, while the
best deep model reaches **0.828**; on the leakage-free split those values are
**0.000** and **0.177**, respectively. The benchmark therefore rewards
shortcuts under the permissive split, and the remaining interaction signal is
not large enough for the tested deep models to outperform the degenerate
baseline in RMSE.

**For benchmark maintainers.** Publish split manifests, target and ligand
identifiers, sequence-identity thresholds, temporal cutoffs, and a leakage
audit. A split name such as “scaffold” is not sufficient: the scaffold rung
still leaves **8.41%** recoverable rows, compared with **1.42%** after
fingerprint clustering.

**For model authors.** Report the noise floor, a target-only or additive
baseline, and cluster-bootstrap intervals. Absolute performance without these
references cannot distinguish interaction modelling from memorisation.

**On robustness.** The registered shift-robust hypothesis failed on this
dataset. The interval head was substantially under-covered, and the
worst-family objective did not improve the worst-family error enough to offset
its loss in average accuracy. This is a useful boundary condition for future
work, not evidence that conformal prediction is valid under arbitrary shift.

### 7.1 Limitations

**Exchangeability.** Conformal coverage is reported empirically, not guaranteed,
under the strict splits.

**Affinity pooling.** Quantified in Section 4 rather than assumed away; results
reported with and without IC50.

**Censoring.** Dropping 27.8% of rows is a real choice. Models intended to
predict bounds are out of scope here.

**Re-running without re-tuning** may understate what a published model could do
with effort we did not spend. We consider the alternative worse: re-tuning
someone else's model measures our tuning budget.

**Subset evaluation for Boltz-2**, recorded in its coverage scope.

---

## 8. Data and code availability

Code: `https://github.com/MinhNhat-2504/leakfree-dta`, MIT.

Released artifacts: the extraction query, the curation log with per-step row
counts, the split manifests with content digests, the audit reports in JSON and
Markdown, per-row predictions for every model, and the re-run attempt log
including failures.

The benchmark is reproducible from a ChEMBL release number and the committed
query. Applying a split manifest to a dataset whose digest does not match is
refused rather than warned about, because the failure it prevents is numbers that
look plausible and mean nothing.

---

## References

[@gaulton_2024] Gaulton, A. et al. The ChEMBL database in 2024: enabling
multi-modal research on drug discovery. *Nucleic Acids Research* (2024).
doi: [10.1093/nar/gkae1000](https://doi.org/10.1093/nar/gkae1000).

[@ozturk_2018] Öztürk, H., Özgür, A. and Ozkirimli, E. DeepDTA: deep drug–target
binding affinity prediction. *Bioinformatics* 34, i821–i829 (2018).
doi: [10.1093/bioinformatics/bty593](https://doi.org/10.1093/bioinformatics/bty593).

[@bemis_1996] Bemis, G. W. and Murcko, M. A. The properties of known drugs. 1.
Molecular frameworks. *Journal of Medicinal Chemistry* 39, 2887–2893 (1996).
doi: [10.1021/jm9602928](https://doi.org/10.1021/jm9602928).

[@sagawa_2020] Sagawa, S., Koh, P. W., Hashimoto, T. B. and Liang, P.
Distributionally robust neural networks for group shifts. ICLR (2020).
[arXiv:1911.08731](https://arxiv.org/abs/1911.08731).

[@romano_2020] Romano, Y., Sesia, M. and Candès, E. Classification with valid
and adaptive coverage. NeurIPS (2020).
[arXiv:2006.02544](https://arxiv.org/abs/2006.02544).

[@de_la_fuente_2023] de la Fuente, J. et al. Towards a more inductive world for
drug repurposing approaches. (2023).
[arXiv:2311.12670](https://arxiv.org/abs/2311.12670).
