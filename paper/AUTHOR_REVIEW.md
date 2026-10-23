# Author review required

The computational audit can prepare evidence, but it cannot approve scientific claims or
make authorship and disclosure decisions. Each item below needs a named author and date.

## Scientific sign-off

- Confirm the headline interpretation: the paper quantifies leakage inflation; it does not
  claim that the DTA field has made no progress.
- Confirm that the registered shift-robust hypothesis is reported as unsupported.
- Approve the endpoint-matched BindingDB result and its provenance caveat: 90.31% of
  PMID-comparable shared records have at least one publication in common.
- Approve the description of 2021--2025 as a retrospective temporal holdout, not a
  prospective study or experimental assay.
- Decide whether the incomplete external-model section remains in the supplement or is
  removed entirely before submission.
- Decide whether Boltz-2 remains in the pre-registered candidate table despite having no
  completed result in this repository.

## Authorship and disclosures

Settled 24 October 2026 and written into `submission_frontmatter.md`:

- [x] Author list and order: Nhat Trinh Ngoc Minh; Linh Nguyen Thi My.
- [x] Corresponding author: Linh Nguyen Thi My, linh.ntm@vlu.edu.vn.
- [x] ORCIDs: 0009-0006-9477-9642; 0000-0003-3990-312X.
- [x] Affiliation: one shared affiliation, Faculty of Information Technology,
      Van Lang School of Technology, Van Lang University, Ho Chi Minh City.
- [x] Funding: none.
- [x] Competing interests: none.

Settled 24 October 2026:

- [x] Author contributions: NTNM carried out the study; NTML supervised,
      administered the project, and reviewed and approved the manuscript.
- [x] Acknowledgements: none, recorded as "Not applicable".
- [x] Ethics and consent: "Not applicable". The section itself is mandatory in
      a BMC Declarations block even when it does not apply, so it stays with
      that content rather than being removed.
- [x] AI-assistance disclosure: wording is in `submission_frontmatter.md`.

Still open:

- [ ] Confirm the AI-assistance wording against the journal's policy as it
      stands on the submission date. Publisher requirements here change often
      enough that a statement written weeks earlier can be out of date.

## Data and release decisions

See `RELEASE_CHECKLIST.md` for the DOI procedure; the short version is Zenodo,
with the DOI reserved before submission and the deposit published at submission.

- [x] Zenodo deposit created and DOI reserved: 10.5281/zenodo.23605646.
      Draft only -- it is published at submission, not before.
- Confirm which ChEMBL-derived artifacts may legally be redistributed.
- Confirm whether the BindingDB input can be redistributed or must be downloaded by readers.
- Freeze a release commit and rerun the full reproduction command from a clean environment.
- Approve all final tables, figures, captions, and supplementary files.

## Optional validation decision

- For a real prospective study, approve a lock date, serialized model checkpoint, candidate
  enrollment rule, endpoints, primary metrics, and timestamped protocol before labels arrive.
- For a wet-lab assay, identify the laboratory, compounds, target construct, assay format,
  controls, replicate plan, blinding, budget, safety review, and data-release agreement.
