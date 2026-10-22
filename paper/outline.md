# Paper outline

**Working title.** How much of our progress is leakage? A leakage-audited
benchmark for drug–target affinity prediction.

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
5. Once every channel is closed, the tested worst-group objective and conformal
   head do not yet deliver the promised robustness; that negative result defines
   a useful boundary for future work.

---

## Sections

### 1. Introduction

Open with the gap between activity and outcome as AI-designed drug programmes
enter clinical development without a corresponding assurance of translation.
Reviewers are
accordingly more interested in reliability than in another point of AUC. Then the
specific question: the field reports steady year-on-year improvement on these
benchmarks — how much of it survives an evaluation that does not leak?

Position against recent audits of benchmark consistency (which found the inconsistencies
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

**Table 4 — the robustness ablation.** Worst-group error, empirical coverage,
and risk-coverage area. State plainly that the registered robustness hypothesis
was not supported.

**Figure 4 — risk-coverage.** The external progress curve is dropped because
published-model re-runs are incomplete and not comparable enough.

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
| T1 leakage ladder | `leakfree report` | complete on ChEMBL 37 |
| T2 model comparison | `leakfree evaluate` | complete on ChEMBL 37 |
| T3 inflation | `leakfree report` | complete on ChEMBL 37 |
| T4 robustness | `leakfree evaluate` | complete; hypothesis not supported |
| F1 leakage by channel | `report.figures` | works |
| F2 inflation curve | `report.figures` | works |
| F3 coverage by family | `report.figures` | complete on ChEMBL 37 |
| F4 risk-coverage | `report.figures` | complete on ChEMBL 37 |
| F5 progress curve | `eval.progress_curve` | intentionally dropped |
| S1 curation log | `leakfree curate` | works |
| S2 inclusion criteria | `models/external.py` | written, pre-registered |
| S3 attempt log | `models.external` | needs the re-runs |

---

## Plan and current state

**Phase 1 — real data.** *Complete for the released ChEMBL 37 extract.*

- MMseqs2 installed and wired. `leakfree info` reports `identity backend
  mmseqs2`; every audit report and split manifest records which backend produced
  it, so nothing already computed with the fallback can be mistaken for a
  publication number.
- `scripts/download_chembl.py` and `scripts/extract_chembl.py` are written and
  linted. The SQL in `sql/chembl_activities.sql` has not yet run against a real
  release.
- The extract was downloaded, queried, curated, split with MMseqs2 clustering,
  and re-audited. Released artefacts record the source release and content
  digests. BindingDB cross-source validation is not part of this paper.

**Phase 2 — the re-runs.** *Harness complete; external published-model track
deferred.*

- `leakfree_dta.external` clones at a pinned commit, lays out data, runs,
  collects, scores, and writes the attempt log. Verified end to end against the
  real DeepDTA repository: it cloned, ran, failed, and produced a correctly
  populated attempt-log row.
- The DeepDTA adapter carries six declared compatibility patches and a converter
  into DeepDTA's native matrix format, with a round-trip check that runs before
  the model starts.
- **Known gap:** `run_experiments.py` logs metrics but writes no per-row
  predictions, so a further patch is needed before DeepDTA can enter the per-row
  comparison. The adapter's `collect` raises with that exact reason rather than
  inventing numbers.
- The GraphDTA, MolTrans and PerceiverCPI adapters are placeholders: the
  repositories and entry points are right, but neither the data layout nor the
  compatibility work has been done. Expect each to need its own DeepDTA-sized
  effort.
- `deepdta_reimpl` is available now as an interim comparison point, and is named
  so that no table can present it as a re-run of the published code.

**Phase 3 — the baseline.** Group DRO and conformal are implemented and tested.
The ChEMBL ablation is complete, and the registered robustness claim is reported
as a negative result: `shift_robust` is under-covered and does not improve
worst-family RMSE enough to justify its loss in average accuracy.

**Phase 4 — writing.** Draft and generated tables are complete. Remaining work is
citation verification, author review, journal formatting, and an optional decision
about the external re-run track.

The external re-run track remains the schedule risk. It is explicitly excluded
from headline claims rather than allowed to block the benchmark paper.

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
