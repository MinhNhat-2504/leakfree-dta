# Journal of Cheminformatics submission checklist

Target article type: **Research Article**.

Checked against the journal-specific Research Article guidance on 2026-10-23:
https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/researchArticle.html

## Satisfied in the current draft

- Abstract is 244 words, below the 350-word limit.
- Abstract contains an explicit **Contribution** heading.
- Source code is MIT licensed and the computational pipeline has automated tests.
- Main numerical claims point to machine-readable result artifacts.
- The manuscript distinguishes the DeepDTA reimplementation from an original-code rerun.
- BindingDB overlap is described as database concordance, not independent validation.
- Temporal evaluation is described as retrospective, not prospective or experimental.

## Required before submission

- Add all author names, affiliations, email addresses, ORCID identifiers, and mark the corresponding author.
- Obtain every author's approval of the final manuscript and author order.
- Add 3--10 keywords.
- Complete every declaration: ethics, consent for publication, data/materials availability,
  competing interests, funding, author contributions, acknowledgements, and optional author information.
- Deposit a versioned code release and result bundle in a persistent repository and cite its DOI.
- Resolve the ChEMBL redistribution/licensing question. The journal requires third-party
  reproducibility without registration or restrictive data terms; a reproducible extraction
  recipe may not by itself satisfy the journal's stronger accessibility policy.
- Verify every reference, DOI, in-text citation, figure number, table number, and supplement link.
- Decide whether to include a graphical abstract. If included: 920 x 300 pixels, white
  background, PNG/JPEG/SVG, at most 150 KB.
- Prepare a cover letter explaining fit, policy exceptions, competing interests, author
  approval, and confirmation that the manuscript is not under review elsewhere.

## Editorial blockers

- The external published-model track is incomplete and cannot support a progress-over-time curve.
- No genuinely prospective or experimental validation has been performed.
- The current training pipeline does not serialize fitted model weights, which blocks a
  pre-registered future prediction set.
- Author-specific declarations and CRediT-style contributions cannot be inferred from code.

## Reproducibility package to archive

- ChEMBL release number, SQL extraction query, raw-extract metadata, and checksum.
- BindingDB release filename/date and checksum; endpoint-matched concordance tables and provenance table.
- Curated-table digests and all split manifests.
- Environment lock file, package versions, MMseqs2 version/backend, random seeds, and commands.
- Per-row predictions, metrics, bootstrap comparisons, figures, tables, and external attempt logs.
- A release tag matching the manuscript revision.
