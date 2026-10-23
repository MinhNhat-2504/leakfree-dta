# Archival release and DOI

Journal of Cheminformatics expects code and data to be archived somewhere
permanent with a DOI. A GitHub URL is not enough on its own: repositories can be
renamed, made private, or deleted, and a reviewer two years from now needs the
exact state the paper describes.

## Where the DOI comes from

**Zenodo** (https://zenodo.org), run by CERN, is the usual choice for this and is
free. Alternatives are figshare, Dryad (data-oriented), and Software Heritage
(archives code but does not mint a DOI for a release in the same way). Zenodo is
recommended here because it integrates directly with GitHub and issues two DOIs:

- a **version DOI**, pointing at one specific release — this is what the paper cites
- a **concept DOI**, always resolving to the newest version — useful in the README

## The ordering problem, and how to avoid it

The manuscript must contain the DOI, but the DOI normally appears only after the
deposit is published. Zenodo solves this: a draft deposit can **reserve** a DOI
before publication. Reserve it, write it into the manuscript, then publish the
deposit when the paper is submitted.

Do not publish the Zenodo record and then keep changing the repository. The
deposit should correspond to a tagged commit, and that tag should not move.

## Steps

1. **Connect the repository.** Sign in to Zenodo with GitHub, open
   https://zenodo.org/account/settings/github/, and switch on
   `MinhNhat-2504/leakfree-dta`. Zenodo only sees public repositories, so push
   first.

2. **Add `.zenodo.json`** at the repository root so the deposit carries correct
   metadata rather than whatever Zenodo infers. A template is below.

3. **Reserve the DOI.** In Zenodo, start a new upload for the repository and use
   "Reserve DOI" in the Basic information panel. Copy it into
   `paper/submission_frontmatter.md` and into Section 8 of `paper/draft.md`.

4. **Freeze and tag.** Confirm the working tree is clean, the full test suite
   passes, and `scripts/reproduce_all.py` runs from a clean checkout. Then:

   ```bash
   git tag -a v1.0.0 -m "Release accompanying the Journal of Cheminformatics submission"
   git push origin v1.0.0
   ```

5. **Create the GitHub Release** from that tag. Zenodo picks it up automatically
   and publishes the archive against the reserved DOI.

6. **Check what was archived.** Zenodo stores the repository as a source
   snapshot, which excludes anything in `.gitignore` — and that excludes
   `data/` and `results/`. Decide deliberately what else to attach:

   - split manifests (small, and the core reusable artifact)
   - audit reports in JSON and Markdown
   - per-row predictions for every model
   - curation log and dataset summary

   Attach these as additional files on the Zenodo record if they are not tracked
   in git. They are the evidence behind the paper's tables.

## Redistribution

ChEMBL is CC BY-SA 3.0 and BindingDB has its own terms. Before attaching any
derived data, confirm what may be redistributed; when in doubt, archive the
extraction code, release identifiers and content digests rather than the records
themselves, which is what the repository is already built to support.

## `.zenodo.json` template

```json
{
  "title": "leakfree-dta: a leakage-audited benchmark for drug-target affinity prediction",
  "upload_type": "software",
  "license": "MIT",
  "creators": [
    {
      "name": "Trinh Ngoc Minh Nhat",
      "orcid": "0009-0006-9477-9642",
      "affiliation": "Faculty of Information Technology, Van Lang School of Technology, Van Lang University, Ho Chi Minh City, Vietnam"
    },
    {
      "name": "Nguyen Thi My Linh",
      "orcid": "0000-0003-3990-312X",
      "affiliation": "Faculty of Information Technology, Van Lang School of Technology, Van Lang University, Ho Chi Minh City, Vietnam"
    }
  ],
  "keywords": [
    "drug-target affinity", "data leakage", "benchmark auditing",
    "scaffold split", "sequence identity", "reproducibility"
  ]
}
```

Check the name order against Zenodo's rendering before publishing: Zenodo
displays `creators[].name` verbatim, and Vietnamese names are often entered
family-name-first in one system and given-name-first in another. The two should
agree with how the authors appear on the manuscript.

## Before tagging

- [ ] Working tree clean, all tests pass, ruff clean
- [ ] `scripts/reproduce_all.py` completes from a clean checkout
- [ ] Every number in `paper/draft.md` re-verified against artifacts
- [ ] `[AUTHOR INPUT REQUIRED]` markers resolved in `submission_frontmatter.md`
- [ ] Reserved DOI written into the manuscript and this repository's README
- [ ] Redistribution of derived data confirmed against ChEMBL and BindingDB terms
