# Data directory

Nothing here is committed except this file and the split manifests. Raw data is
large, licensed by its providers, and regenerable; splits are small and *are* the
published artifact.

```
raw/        source downloads, exactly as obtained
interim/    caches: fingerprints, sequences, identity graphs, ESM embeddings
processed/  curated tables in the canonical schema
splits/     frozen split manifests  <-- these are committed
external/   third-party benchmark releases (Davis, KIBA, PDBbind)
```

## Obtaining the sources

**ChEMBL** (the primary source — the only one with publication years, which is
what makes a temporal split possible).

Download the SQLite release from
<https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/> and extract
activities with the query in `sql/chembl_activities.sql`. Record the release
number: "ChEMBL" without a version is not a reproducible input, and the same
extract rebuilt a year later will differ.

**BindingDB** (cross-source consistency). `BindingDB_All.tsv` from
<https://www.bindingdb.org/rwd/bind/downloads>. Several GB; the loader streams it
in chunks.

**UniProt** sequences are fetched on demand and cached in `interim/sequences/`.

**Davis and KIBA** from the DeepDTA release
(<https://github.com/hkmztrk/DeepDTA>), placed in `external/davis/` and
`external/kiba/`. Loaded so the audit can be applied to the benchmarks the field
actually reports on — not because they are good.

**PDBbind** requires registration at <http://www.pdbbind.org.cn/>. Only the index
file is used; SMILES and sequences must be supplied separately.

## A note on provenance

Treat every download as untrusted input: put each in its own directory under
`raw/` or `external/` and pass paths as arguments. Do not run scripts from inside
a download directory — a Python interpreter loads modules from the current
directory, so a stray `json.py` in an extracted archive would be imported in place
of the standard library.

## Reproducibility

Every curated table carries a content digest, and every split manifest records the
digest of the dataset it was built on. Applying a manifest to a different dataset
is refused rather than warned about, because the failure mode it prevents is the
worst kind: numbers that look entirely plausible and mean nothing.
