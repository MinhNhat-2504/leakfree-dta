# Paper outline

**Working title.** How much of our progress is leakage? A leakage-audited
benchmark and shift-robust baseline for drug–target affinity prediction.

**Target journals**, in submission order:

1. *Journal of Cheminformatics* — open access, rewards resource papers with
   working code, reviewers who will actually run it. Best fit.
2. *Briefings in Bioinformatics* — favours systematic comparisons; good second
   choice.
3. *Scientific Data* — only if the benchmark itself becomes the contribution and
   the baseline is cut. Accepts data descriptors, so the framing would change.
4. *Bioinformatics* (Application Note) — fallback if the scope has to shrink.

**Not** *JCIM* for this paper. Its reviewers expect a methodological contribution
to a model; this paper's contribution is an evaluation protocol. Save *JCIM* for
the follow-up (Topic 1 in the proposal document: conformal calibration of Boltz-2,
which reuses this benchmark as its substrate).

---

## The argument in five sentences

1. Drug–target affinity benchmarks that the field has reported on for a decade let
   information cross the train/test boundary.
2. The standard mitigation — Bemis–Murcko scaffold splitting — closes almost none
   of it, because homologous series get different scaffolds but identical
   fingerprints.
3. Predictors that are structurally incapable of modelling interaction recover much
   of the reported performance, which bounds how much interaction modelling the
   benchmarks actually reward.
4. Which channel dominates depends on the target family, so no single splitting
   rule is sufficient and the field's search for "the" hard split is misdirected.
5. Once every channel is closed, a deliberately plain model with a worst-group
   objective and group-conditional intervals is more *usable* than a more accurate
   one, because it knows where it does not work.

---

## Sections

### 1. Introduction

Open with the gap between activity and outcome: over 170 AI-designed drug
programmes have entered the clinic and none has an FDA approval. Reviewers are
accordingly more interested in reliability than in another point of AUC. Then the
specific question: the field reports steady year-on-year improvement on these
benchmarks — how much of it survives an evaluation that does not leak?

Position against the 2026 *Chemical Science* audit (which found the inconsistencies
but did not quantify their effect on reported performance) and arXiv 2311.12670
(which identified the transductive shortcut but inferred it from how splits were
constructed, rather than measuring it). This paper measures it directly, which
means it applies to any split, including ones whose construction is undocumented.

### 2. The benchmark

Canonical schema, and why InChIKey rather than SMILES. Curation decisions as a
published log with row counts, so a reader who disagrees with one can see its size.
Three families and why they were chosen — the differential-failure argument, stated
here as a hypothesis and tested in Section 4.

Cross-source consistency: pairs present in both ChEMBL and BindingDB, and how far
apart the two curated databases put them. This sets a floor on achievable accuracy
that is independent of any model, and it has not been published.

### 3. The audit

The channels, one subsection each. Then the shortcut probe, which deserves its own
subsection and probably its own figure — it is the result most likely to be quoted.

**Calibration.** The synthetic fixture and the planted quantities. Lead with this
rather than appendixing it: an audit is a measuring instrument, and showing it
recovers a known answer is what licenses every number that follows. The probe
reaching 0.853 against an analytic ceiling of 0.890 is the sentence to put in the
methods.

**The noise floor.** Pooled within-pair standard deviation over replicated pairs.
Quote it beside every RMSE in the paper. An RMSE below the floor is not an
achievement — it indicates the model is fitting assay artifacts shared between the
training and test labels.

### 4. Results

**Table 1 / Figure 1 — the leakage ladder.** Affected fraction per channel per
split. The `random` → `ligand_scaffold` → `ligand_cluster` progression is the
headline.

**Table 2 — models on the leakage-free split**, with the degenerate predictors in
the *same table*. This is the single most important editorial decision in the
paper. Putting `additive_bias` in an appendix would let a reader skip it; putting
it beside the deep models makes the comparison unavoidable.

**Table 3 / Figure 2 — inflation.** Each model's score across the ladder. Lines
that fall steeply were measuring leakage; flat lines were measuring something real.
Both outcomes are findings, and the paper should say so before showing which
happened.

**Figure 3 — per-family dominant channel.** The differential-failure result. One
word per family in the abstract.

**Table 4 — the shift-robust baseline.** Worst-group error, group-conditional
coverage, risk-coverage area. State plainly that it loses on average RMSE, and why
that is the expected consequence of the objective rather than a disappointment.

**Figure 4 — the progress curve** (optional, and only if enough models re-run).
Reported versus re-measured performance by publication year. High-risk, high-reward:
it is the most quotable figure in the paper and the most likely to be attacked on
comparability grounds. Include it only with per-point provenance and the attempt
log, and drop it rather than weaken the comparability rules.

### 5. Discussion

What a benchmark maintainer should change. What a model author should report. The
limitations, stated before a reviewer raises them — exchangeability under shift,
IC50 pooling, the approximate fallback backend, and the fact that re-running
published code without re-tuning may understate what those models could do.

Resist the temptation to claim the field has made no progress. The defensible claim
is narrower and stronger: *a specific fraction* of reported improvement does not
survive a leakage-free evaluation, and the field has no way of knowing which
fraction because nobody has measured it.

---

## Figures and tables

| artifact | produced by | status |
| --- | --- | --- |
| T1 leakage ladder | `leakfree report` | pipeline works on synthetic data |
| T2 model comparison | `leakfree evaluate` | pipeline works |
| T3 inflation | `leakfree report` | pipeline works |
| T4 robustness | `leakfree evaluate` | needs real data |
| F1 leakage by channel | `report.figures` | works |
| F2 inflation curve | `report.figures` | works |
| F3 per-family channels | `audit.runner` | works |
| F4 coverage by group | `report.figures` | works |
| F5 progress curve | `eval.progress_curve` | needs the re-runs |
| S1 curation log | `leakfree curate` | works |
| S2 inclusion criteria | `models.external` | written, pre-registered |
| S3 attempt log | `models.external` | needs the re-runs |

---

## Plan

**Phase 1 — real data (3–4 weeks).** Build the ChEMBL extract. Install MMseqs2 and
re-run everything that currently uses the fallback. Fetch BindingDB for the
cross-source table. The pipeline is finished; this phase is data work.

**Phase 2 — the re-runs (4–6 weeks).** The critical path and the main risk. Run
the five models in `models/external.py` on the frozen splits. Fix the inclusion
criteria *before* starting — they are already committed in
`configs/experiment/main.yaml` and `models/external.py`, which is the point.

**Phase 3 — the baseline (2–3 weeks).** Group DRO and conformal are implemented.
This phase is hyperparameters, the ablation (ERM versus DRO, with and without the
conformal head), and the per-family breakdown.

**Phase 4 — writing (3–4 weeks).** Draft, then preprint immediately. Topics 3 and 5
in the proposal document were both flagged as competitive; the date stamp matters.

Total 3–4 months, consistent with the 3–5 months the proposal estimated.

**Before writing a word, re-check novelty.** Search arXiv, bioRxiv and ChemRxiv for
"DTI benchmark leakage", "drug target affinity data leakage audit" and
"scaffold split insufficient". If a group has published the ladder, pivot to the
shortcut probe and the shift-robust baseline, which are the more original
contributions anyway.

---

## What would make this stronger

**A prospective check.** Hold out everything after a cutoff, predict, then compare
against measurements published since. Costs nothing but patience and converts a
retrospective argument into a prospective one.

**A pharmacology co-author.** Not required for the computational result, but needed
to write the "so what" paragraph credibly, and to choose the case study in Section 4
that makes the leakage concrete for a medicinal chemist.

**One experimental data point.** Even a single enzyme inhibition assay on a top
prediction from the leakage-free model would lift the paper's standing
considerably. Any paper with experimental validation outranks a purely
computational one.
