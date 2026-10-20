-- Extract activities for the three target families from a ChEMBL release.
--
-- Run against the SQLite distribution:
--   sqlite3 chembl_35.db < sql/chembl_activities.sql > data/raw/chembl_3fam.csv
--
-- Decisions made here rather than in Python, because they are cheap in SQL and
-- expensive afterwards:
--
--   * Single-protein targets only. A protein complex or a cell line cannot be
--     given one sequence, so it cannot enter a sequence-based split.
--   * Confidence >= 8. ChEMBL's 8 is "homologous single protein" and 9 is "direct
--     single protein"; below 8 the activity is not reliably attributable to one
--     target, and the pair key would be wrong.
--   * Human only, by default. Cross-species orthologues are a leakage channel in
--     their own right: rat and human EGFR are ~90% identical, so letting both in
--     would mean a "held-out" target has a near-copy in training. Relax the
--     organism filter only after deciding how orthologues should be clustered.
--   * No aggregation. Replicates are kept, because the audit measures their
--     disagreement and that is one of the paper's results.
--
-- The protein class path is selected rather than inferred, since the sequence
-- motif fallback is far less reliable than ChEMBL's own classification.

.headers on
.mode csv

SELECT
    md.chembl_id                      AS molecule_chembl_id,
    cs.canonical_smiles               AS canonical_smiles,
    cseq.accession                    AS target_uniprot,
    cseq.sequence                     AS target_sequence,
    td.chembl_id                      AS target_chembl_id,
    td.pref_name                      AS target_name,
    act.standard_type                 AS standard_type,
    act.standard_relation             AS standard_relation,
    act.standard_value                AS standard_value,
    act.standard_units                AS standard_units,
    act.pchembl_value                 AS pchembl_value,
    act.activity_comment              AS activity_comment,
    act.assay_id                      AS assay_id,
    a.confidence_score                AS confidence_score,
    a.assay_type                      AS assay_type,
    docs.year                         AS document_year,
    docs.doi                          AS doi,
    docs.pubmed_id                    AS pubmed_id,
    -- The full classification path, root first, as in
    -- "Enzyme/Kinase/Protein Kinase/TK protein kinase group".
    --
    -- The whole path, not one level of it. component_class links to the most
    -- specific class and pc2..pc5 walk up through its ancestors, so a COALESCE
    -- over them returns whichever ancestor happens to be highest -- "Enzyme" for
    -- a kinase, "Membrane receptor" for a GPCR. The family regexes in
    -- prot/families.py look for "kinase", "protease" and "7tm", none of which
    -- appear at that level, so almost every target would have fallen through to
    -- the sequence-motif heuristic and then been dropped as unassigned.
    -- Concatenating the path lets a regex match whichever level carries the word.
    TRIM(
        COALESCE(pc5.pref_name || '/', '') ||
        COALESCE(pc4.pref_name || '/', '') ||
        COALESCE(pc3.pref_name || '/', '') ||
        COALESCE(pc2.pref_name || '/', '') ||
        COALESCE(pc1.pref_name, ''),
        '/'
    )                                 AS target_protein_class
FROM activities              AS act
JOIN assays                  AS a     ON a.assay_id          = act.assay_id
JOIN target_dictionary       AS td    ON td.tid              = a.tid
JOIN target_components       AS tc    ON tc.tid              = td.tid
JOIN component_sequences     AS cseq  ON cseq.component_id   = tc.component_id
JOIN molecule_dictionary     AS md    ON md.molregno         = act.molregno
JOIN compound_structures     AS cs    ON cs.molregno         = act.molregno
JOIN docs                            ON docs.doc_id         = act.doc_id
LEFT JOIN component_class    AS cc    ON cc.component_id     = tc.component_id
LEFT JOIN protein_classification AS pc1 ON pc1.protein_class_id = cc.protein_class_id
LEFT JOIN protein_classification AS pc2 ON pc2.protein_class_id = pc1.parent_id
LEFT JOIN protein_classification AS pc3 ON pc3.protein_class_id = pc2.parent_id
LEFT JOIN protein_classification AS pc4 ON pc4.protein_class_id = pc3.parent_id
LEFT JOIN protein_classification AS pc5 ON pc5.protein_class_id = pc4.parent_id
WHERE
    td.target_type   = 'SINGLE PROTEIN'
    AND td.organism  = 'Homo sapiens'
    AND a.confidence_score >= 8
    AND act.standard_type IN ('Ki', 'Kd', 'IC50', 'EC50')
    AND act.standard_units = 'nM'
    AND act.standard_value IS NOT NULL
    AND act.standard_value > 0
    -- Exclude activities ChEMBL itself flagged as unreliable.
    AND (act.data_validity_comment IS NULL OR act.data_validity_comment = '')
    AND act.potential_duplicate = 0
    AND cs.canonical_smiles IS NOT NULL
    AND cseq.sequence IS NOT NULL
    -- One component per target: a multi-component target has no single sequence.
    AND (SELECT COUNT(*) FROM target_components WHERE tid = td.tid) = 1
ORDER BY cseq.accession, md.chembl_id, act.assay_id;
