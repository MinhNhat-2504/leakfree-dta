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

## Two routes, and why this one

Zenodo can archive a GitHub repository automatically, and that is the route most
guides describe. It is the wrong route here.

**The GitHub integration archives only what git tracks.** `data/` and `results/`
are gitignored, so the automatic snapshot would contain the source code and none
of the evidence: no split manifests, no audit reports, no per-row predictions.
Those artifacts are what the paper's tables rest on, and a deposit without them
does not discharge the journal's requirement.

It also mints the DOI only when a release is created, whereas the manuscript
needs the DOI before submission.

So the deposit is created **manually**, and the GitHub release exists alongside
it to fix the exact commit. `.zenodo.json` is only read by the GitHub
integration; on this route it does nothing, and the template below is kept only
in case the integration is used later for a future version.

## Steps

1. **Reserve the DOI first.** On Zenodo, *New upload* -> upload type `Software`
   -> *Reserve DOI* in the Basic information panel. Do not publish. Then replace
   every `[RESERVED DOI — insert before submission]` marker:

   ```bash
   grep -rn "RESERVED DOI" paper/
   ```

   Currently two, in `submission_frontmatter.md` (Availability of data and
   materials) and `draft.md` (Section 8). The grep is the check, not the list --
   it stays correct if a third appears.

2. **Fill the metadata** by uploading `paper/zenodo_creators.json` in the
   creators panel, which is why that file exists: the ORCIDs and the affiliation
   are then typed once rather than per deposit.

   The name split follows each author's own ORCID record, not the usual
   Vietnamese reading of the name. ORCID has given `Nhat` with family
   `Trinh Ngoc Minh`, and given `Linh` with family `Nguyen Thi My`, so Zenodo
   renders "Trinh Ngoc Minh, Nhat" -- matching the credit name on the ORCID
   record. That is what Crossref and Scholar match against, so ORCID is the
   authority here and the manuscript byline already agrees with it.

3. **Freeze and tag.** Working tree clean, full test suite passing, ruff clean,
   and `scripts/reproduce_all.py` completing from a clean checkout. Then:

   ```bash
   git tag -a v1.0.0 -m "Release accompanying the Journal of Cheminformatics submission"
   git push origin v1.0.0
   ```

4. **Create the GitHub Release** from that tag, and download its source archive.
   That archive is one of the files attached to the deposit, so the deposit and
   the tag describe the same commit.

5. **Attach the evidence.** The source archive alone is not the deposit. Add, at
   minimum:

   - `data/splits/<dataset>/_index.json` -- fold counts, specifications and
     per-fold digests, enough to verify a rebuilt split
   - `results/splits/<dataset>_assignment.parquet` -- the full fold assignment,
     so a reader applies the splits without rerunning the clustering
   - per-row predictions for every model that was run
   - audit reports and the curation log
   - the tables and figures the paper reproduces

   `results/` is roughly 1 GB, most of it per-row detail inside the audit
   reports. Zenodo allows 50 GB per record, so size is not the constraint;
   deliberateness is. Archive what the paper's claims rest on, not everything the
   pipeline happened to write.

6. **Publish when the paper is submitted**, not before, and do not move the tag
   afterwards. A published Zenodo record is permanent by design.

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
      "name": "Trinh Ngoc Minh, Nhat",
      "orcid": "0009-0006-9477-9642",
      "affiliation": "Faculty of Information Technology, Van Lang School of Technology, Van Lang University, Ho Chi Minh City, Vietnam"
    },
    {
      "name": "Nguyen Thi My, Linh",
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

## Before tagging

- [x] Working tree clean, all tests pass (146 passed, 1 skipped), ruff clean
- [ ] `scripts/reproduce_all.py` completes from a clean checkout
- [x] Every number in `paper/draft.md` re-verified against artifacts
- [x] `[AUTHOR INPUT REQUIRED]` markers resolved in `submission_frontmatter.md`
- [x] Reserved DOI written into the manuscript and this repository's README
- [x] Redistribution of derived data confirmed: ChEMBL is CC BY-SA 3.0 and the
      derived files are attributed as such; BindingDB appears only as aggregate
      concordance statistics, with no records redistributed

Tagged as v1.0.0 at commit 62ef9bd with `reproduce_all.py` unverified from a
clean checkout. That is the one item above that was taken on trust rather than
checked, and it is recorded here rather than ticked.
