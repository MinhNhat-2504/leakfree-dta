# Validation and submission status

Updated: 2026-10-24 (Asia/Ho_Chi_Minh)

## External published-model re-runs — completed where technically eligible

The harness, durable outcomes, consolidated attempt log, DeepDTA matrix encoding,
and PerceiverCPI input layout are implemented and tested. DeepDTA, GraphDTA and
PerceiverCPI have all produced full comparable prediction vectors on the
leakage-free split from their upstream sources. Three of six registered
candidates therefore completed; the other three are attempt-log rows with stated
reasons.

| candidate | status | evidence / next requirement |
| --- | --- | --- |
| DeepDTA | completed on CUDA | Upstream source at `a546a8433a68` run under WSL2 on the RTX 4060 through the split-aware wrapper. Best validation epoch 3, 18 epochs, 12.6 minutes. Test RMSE 1.3278, Pearson r 0.1701, Spearman 0.1529, 4,903 predictions. Two declared departures: selection on the designated validation fold rather than upstream's test-as-validation protocol, and a `random.randint(1, 1e9)` coercion for tf_keras 2.16 on Python 3.12, which touches seed generation only. Row keys and labels verified against the split manifest; predictions SHA-256 recorded. Artifacts under `data/external/chembl_3fam/leakfree/DeepDTA/output-cuda-v2/` and `results/runs/chembl_3fam/leakfree/DeepDTA/`. |
| GraphDTA | completed on CUDA | Upstream GINConvNet head `6162245c6776684b391d30610850fe7077478435`; RTX 4060 Laptop GPU; best validation epoch 9, early-stopped after 34 epochs. Test RMSE 1.3237, Pearson r 0.1565, 4,903 predictions. Checkpoint and run metadata are under `data/external/chembl_3fam/leakfree/GraphDTA/output/`; aligned predictions are in `results/runs/chembl_3fam/leakfree/GraphDTA/`. |
| MolTrans | excluded | Released code is binary CPI classification with sigmoid/BCELoss and has no regression head; conversion would be a reimplementation. |
| PerceiverCPI | completed on CUDA | RTX 4060 Laptop GPU; published CLI run with batch 16, explicit validation/test paths, and only infrastructure shims for Windows pipes/RAM. Test RMSE 1.3281, Pearson r 0.1079, Spearman rho 0.0815, 4,903 predictions; elapsed 417.21 minutes. Checkpoint, metadata, and aligned predictions are under `data/external/chembl_3fam/leakfree/PerceiverCPI/output/` and `results/runs/chembl_3fam/leakfree/PerceiverCPI/`. |
| ChemBERTa-2 + ESM-2 | invalidated/excluded | The local script used ECFP4 plus dipeptide frequencies and was moved out of the reportable run tree. It was neither a foundation-model run nor a published DTA model. |
| Boltz-2 | not run | Requires a separately defined, frozen subset and external compute budget; no result should appear in this paper until both exist. |

The progress-over-time curve remains omitted. Failed, timed-out, skipped, and
invalidated candidates must not be converted into successful results by renaming
reimplementations.

## BindingDB cross-source analysis — complete as concordance

- 10,521 endpoint-matched ligand-target-endpoint combinations overlap.
- Pearson r = 0.9493; RMSE = 0.4016 log units; 2.86% differ by more than one log unit.
- PMID provenance is available in both sources for 9,163 combinations; 90.31%
  share at least one publication.
- Therefore this is database concordance/provenance overlap, not independent
  cross-source assay validation and not a physical accuracy floor.

Artifacts: `results/tables/chembl_3fam/table_cross_source_bindingdb.csv`,
`table_cross_source_by_family.csv`, `table_cross_source_provenance.csv`, and
`cross_source_summary.md`.

## Prospective / experimental validation — not executed

The completed analysis is a retrospective temporal holdout: train through 2018,
validate on 2019--2020, and test on 107,527 records from 2021--2025. All labels
already existed when the analysis was designed. The generated prospective protocol
states what must be frozen before future labels are acquired.

A true prospective study additionally requires serialized model weights, a lock
date, pre-enrolled candidates and endpoints, a timestamped analysis plan, and
future labels. A wet-lab study additionally requires compounds, assay design,
controls, replicates, safety review, budget, an independent laboratory, and author
approval.

## Author review / target-journal format — prepared, awaiting authors

The target is Journal of Cheminformatics, Research Article. The abstract is 244
words and includes the required Contribution heading. Submission front matter,
journal checklist, and author-review checklist are prepared. Authorship, ORCIDs,
affiliations, funding, competing interests, contributions, acknowledgements,
data-release DOI, and final scientific sign-off require author input and cannot be
completed from repository evidence.
