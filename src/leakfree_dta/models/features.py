"""Feature construction shared by the non-deep models.

Kept in one place so that every baseline sees identical features. When baselines
differ only in their learner, a difference between them is attributable to the
learner -- which is the point of having them.

Protein features are intentionally weak by default: amino-acid composition and
dipeptide frequencies. A stronger protein representation (ESM embeddings, enabled
with ``protein="esm"``) is available, and the paper reports both, because the gap
between them is informative. If a model gains nothing from a 650M-parameter
protein language model over k-mer counts, the benchmark is not testing protein
understanding.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class FeatureSpec:
    """How to featurise a curated table."""

    ligand: str = "ecfp4"  # "ecfp4" | "ecfp6" | "descriptors" | "ecfp4+descriptors"
    protein: str = "composition"  # "composition" | "dipeptide" | "esm" | "none"
    n_bits: int = 2048
    include_cross: bool = False

    def describe(self) -> dict[str, object]:
        return {
            "ligand": self.ligand,
            "protein": self.protein,
            "n_bits": self.n_bits,
            "include_cross": self.include_cross,
        }


def ligand_features(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Ligand block of the feature matrix, one row per table row."""
    from leakfree_dta.chem.fingerprints import descriptor_frame, ecfp_matrix

    unique = df[[schema.LIGAND_KEY, schema.SMILES]].drop_duplicates(subset=[schema.LIGAND_KEY])
    position = {key: index for index, key in enumerate(unique[schema.LIGAND_KEY])}
    rows = df[schema.LIGAND_KEY].map(position).to_numpy()

    blocks: list[np.ndarray] = []
    if "ecfp" in spec.ligand:
        radius = 3 if "ecfp6" in spec.ligand else 2
        blocks.append(ecfp_matrix(unique[schema.SMILES], radius=radius, n_bits=spec.n_bits))
    if "descriptors" in spec.ligand:
        descriptors = descriptor_frame(unique[schema.SMILES].tolist()).to_numpy(dtype=np.float32)
        # Median-impute, since a handful of molecules fail individual descriptors.
        medians = np.nanmedian(descriptors, axis=0)
        descriptors = np.where(np.isfinite(descriptors), descriptors, medians)
        blocks.append(descriptors)
    if not blocks:
        raise ValueError(f"unknown ligand feature {spec.ligand!r}")

    unique_features = np.hstack(blocks)
    return unique_features[rows]


def protein_features(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Protein block of the feature matrix, one row per table row."""
    if spec.protein == "none":
        return np.zeros((len(df), 0), dtype=np.float32)

    unique = df[[schema.TARGET_KEY, schema.TARGET_SEQUENCE]].drop_duplicates(
        subset=[schema.TARGET_KEY]
    )
    position = {key: index for index, key in enumerate(unique[schema.TARGET_KEY])}
    rows = df[schema.TARGET_KEY].map(position).to_numpy()

    if spec.protein == "composition":
        features = np.vstack([_composition(str(seq)) for seq in unique[schema.TARGET_SEQUENCE]])
    elif spec.protein == "dipeptide":
        features = np.vstack([_dipeptide(str(seq)) for seq in unique[schema.TARGET_SEQUENCE]])
    elif spec.protein == "esm":
        features = _esm_embeddings(
            dict(zip(unique[schema.TARGET_KEY], unique[schema.TARGET_SEQUENCE].astype(str), strict=True))
        )
    else:
        raise ValueError(f"unknown protein feature {spec.protein!r}")

    return features[rows]


@lru_cache(maxsize=4096)
def _composition(sequence: str) -> np.ndarray:
    """Amino-acid composition: 20 frequencies."""
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    for residue in sequence.upper():
        index = AMINO_ACIDS.find(residue)
        if index >= 0:
            counts[index] += 1
    total = counts.sum()
    return counts / total if total else counts


@lru_cache(maxsize=2048)
def _dipeptide(sequence: str) -> np.ndarray:
    """Dipeptide frequencies: 400 features, concatenated with composition."""
    sequence = sequence.upper()
    counts = np.zeros(len(AMINO_ACIDS) ** 2, dtype=np.float32)
    for left, right in itertools.pairwise(sequence):
        i, j = AMINO_ACIDS.find(left), AMINO_ACIDS.find(right)
        if i >= 0 and j >= 0:
            counts[i * len(AMINO_ACIDS) + j] += 1
    total = counts.sum()
    if total:
        counts /= total
    return np.concatenate([_composition(sequence), counts])


def _esm_embeddings(sequences: dict[str, str]) -> np.ndarray:
    """Mean-pooled ESM-2 embeddings, cached to disk.

    Requires the ``plm`` extra. Falls back to dipeptide features with a loud
    warning rather than failing, so that a run on a machine without the model
    still produces a complete results table -- with the fallback recorded.
    """
    from leakfree_dta import paths
    from leakfree_dta.utils.io import read_json, write_json

    cache_file = paths.cache_dir("esm") / "esm2_t12_mean.json"
    cache: dict[str, list[float]] = read_json(cache_file) if cache_file.exists() else {}
    missing = {key: value for key, value in sequences.items() if key not in cache}

    if missing:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.warning(
                "ESM features requested but transformers/torch are unavailable; "
                "falling back to dipeptide features. Install the 'plm' extra for the real thing."
            )
            return np.vstack([_dipeptide(sequences[key]) for key in sequences])

        model_id = "facebook/esm2_t12_35M_UR50D"
        logger.info("embedding %d sequences with %s", len(missing), model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        with torch.no_grad():
            for key, sequence in missing.items():
                encoded = tokenizer(
                    sequence[:1022], return_tensors="pt", truncation=True, max_length=1022
                ).to(device)
                output = model(**encoded).last_hidden_state
                # Mean over real tokens only; padding would dilute the embedding.
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
                cache[key] = pooled.squeeze(0).cpu().numpy().astype(float).tolist()

        write_json(cache, cache_file)

    return np.vstack([np.asarray(cache[key], dtype=np.float32) for key in sequences])


def build_features(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """The full feature matrix for a curated table.

    ``include_cross=True`` appends a low-rank interaction block -- the outer
    product of the two blocks' leading principal components. Without it, a linear
    model literally cannot represent an interaction, which makes it a useful
    control: its performance is the ceiling for a pure additive model, and
    comparing it to the shortcut probe shows whether a benchmark rewards anything
    beyond additive effects.
    """
    ligand = ligand_features(df, spec)
    protein = protein_features(df, spec)
    blocks = [ligand, protein]

    if spec.include_cross and protein.shape[1] > 0:
        blocks.append(_cross_block(ligand, protein))

    return np.hstack(blocks).astype(np.float32)


def _cross_block(ligand: np.ndarray, protein: np.ndarray, *, n_components: int = 8) -> np.ndarray:
    """Outer product of the leading components of each block."""
    from sklearn.decomposition import TruncatedSVD

    n_ligand = min(n_components, max(ligand.shape[1] - 1, 1))
    n_protein = min(n_components, max(protein.shape[1] - 1, 1))
    ligand_reduced = TruncatedSVD(n_components=n_ligand, random_state=0).fit_transform(ligand)
    protein_reduced = TruncatedSVD(n_components=n_protein, random_state=0).fit_transform(protein)
    return (ligand_reduced[:, :, None] * protein_reduced[:, None, :]).reshape(len(ligand), -1)
