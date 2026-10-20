"""DeepDTA, reimplemented in PyTorch.

**This is a reimplementation, not the published code, and the paper must say so
wherever this model's numbers appear.** The distinction is not pedantry. The
original (Ozturk et al., Bioinformatics 2018) is Keras on TensorFlow 1.x under
Python 2.7/3.6, which does not run in any environment that also runs the rest of
this pipeline. Attempting the original is still worth doing and is recorded in the
attempt log; this module is what makes a 2018-vintage architecture available as a
comparison point in the meantime.

What is reproduced, from the paper and the reference implementation:

* Character-level integer encoding of SMILES (64 symbols, truncated at 100) and of
  the protein sequence (25 symbols, truncated at 1000).
* Two parallel CNN towers, three convolutions each with 32/64/96 filters, kernel
  widths 4/6/8 for the drug and 4/8/12 for the protein, ReLU, then global max
  pooling.
* Concatenation into 1024-1024-512 fully connected layers with dropout 0.1, and a
  linear output.
* Adam at 1e-3, batch 256, MSE loss.

What differs, and why each difference is unavoidable rather than chosen:

* PyTorch instead of Keras/TF1. Weight initialisation defaults differ between the
  frameworks, so individual runs will not match the original numerically.
* Early stopping on our validation fold. The original trains a fixed 100 epochs
  on a split that has no validation fold. Training to a fixed epoch count on a
  leakage-free split would be the less faithful choice, not the more faithful one,
  because the original's epoch count was itself selected against its own data.

Neither difference touches the architecture. A reader who wants the published
numbers should read the published paper; what this model answers is "how does a
2018 architecture do on a split that does not leak".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.models.base import DTAModel, FitResult, Prediction

logger = get_logger(__name__)

#: The SMILES alphabet from the reference implementation, in its original order.
#: Index 0 is reserved for padding, so a symbol's code is its position plus one.
CHARISOSMISET = (
    "#%)(+*-/.1032547698:=@[]\\acbedgfihmlonsrutyBACEDGFIHKMLONPSRUTWVY"
)

#: The protein alphabet, likewise.
CHARPROTSET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

MAX_SMILES_LEN = 100
MAX_PROTEIN_LEN = 1000


def _encode(text: str, alphabet: str, max_len: int) -> np.ndarray:
    """Integer-encode a string, left-aligned and zero-padded.

    Unknown symbols become 0, the same code as padding. That is what the
    reference implementation does, and changing it here would be a silent
    deviation; it is noted rather than fixed.
    """
    lookup = {symbol: index + 1 for index, symbol in enumerate(alphabet)}
    encoded = np.zeros(max_len, dtype=np.int64)
    for position, symbol in enumerate(text[:max_len]):
        encoded[position] = lookup.get(symbol, 0)
    return encoded


@dataclass
class DeepDTAConfig:
    """Hyperparameters, at the published values unless noted."""

    embedding_dim: int = 128
    n_filters: int = 32
    drug_kernels: tuple[int, int, int] = (4, 6, 8)
    protein_kernels: tuple[int, int, int] = (4, 8, 12)
    hidden_sizes: tuple[int, int, int] = (1024, 1024, 512)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15  # not in the original; see the module docstring
    seed: int = 0
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def _build_network(torch, nn, config: DeepDTAConfig):
    class Tower(nn.Module):
        """One CNN tower: embed, three widening convolutions, global max pool."""

        def __init__(self, vocab_size: int, kernels: tuple[int, int, int]) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
            channels = [config.embedding_dim, config.n_filters, config.n_filters * 2, config.n_filters * 3]
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernels[i])
                    for i in range(3)
                ]
            )
            self.out_features = config.n_filters * 3

        def forward(self, tokens):
            # (batch, length) -> (batch, embedding, length) for Conv1d.
            x = self.embedding(tokens).transpose(1, 2)
            for convolution in self.convolutions:
                x = torch.relu(convolution(x))
            return x.max(dim=2).values

    class DeepDTANet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drug = Tower(len(CHARISOSMISET) + 1, config.drug_kernels)
            self.protein = Tower(len(CHARPROTSET) + 1, config.protein_kernels)

            layers: list[Any] = []
            size = self.drug.out_features + self.protein.out_features
            for index, hidden in enumerate(config.hidden_sizes):
                layers.append(nn.Linear(size, hidden))
                layers.append(nn.ReLU())
                # The original applies dropout after the first two blocks only.
                if index < 2:
                    layers.append(nn.Dropout(config.dropout))
                size = hidden
            layers.append(nn.Linear(size, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, drug_tokens, protein_tokens):
            joined = torch.cat([self.drug(drug_tokens), self.protein(protein_tokens)], dim=1)
            return self.head(joined).squeeze(-1)

    return DeepDTANet()


class DeepDTAModel(DTAModel):
    """DeepDTA's architecture, trained on our splits."""

    name = "deepdta_reimpl"
    reports_uncertainty = False

    def __init__(self, config: DeepDTAConfig | None = None, **kwargs: Any) -> None:
        self.config = config or DeepDTAConfig(**kwargs)
        super().__init__(**self.config.__dict__)
        self._model = None

    # --- encoding -----------------------------------------------------------

    def _encode_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode each distinct ligand and target once, and return index arrays.

        Returns ``(drug_codes, drug_index, protein_codes, protein_index)``, where
        the code arrays hold one row per *distinct* molecule or protein and the
        index arrays map dataset rows onto them.

        Materialising one encoded protein per row is the obvious implementation
        and does not fit: at 1,000 residues in int64, the 509,988-row training
        fold is 4.08 GB for the protein side alone, against 7.2 MB for the 905
        distinct sequences plus 2 MB of indices. The batches gather from the
        small arrays instead, which also keeps the host-to-device copy small.
        """
        ligands = df[[schema.LIGAND_KEY, schema.SMILES]].drop_duplicates(subset=[schema.LIGAND_KEY])
        targets = df[[schema.TARGET_KEY, schema.TARGET_SEQUENCE]].drop_duplicates(
            subset=[schema.TARGET_KEY]
        )

        drug_codes = np.vstack(
            [_encode(str(value), CHARISOSMISET, MAX_SMILES_LEN) for value in ligands[schema.SMILES]]
        )
        protein_codes = np.vstack(
            [
                _encode(str(value), CHARPROTSET, MAX_PROTEIN_LEN)
                for value in targets[schema.TARGET_SEQUENCE]
            ]
        )

        drug_position = {key: i for i, key in enumerate(ligands[schema.LIGAND_KEY])}
        protein_position = {key: i for i, key in enumerate(targets[schema.TARGET_KEY])}
        return (
            drug_codes,
            df[schema.LIGAND_KEY].map(drug_position).to_numpy(dtype=np.int64),
            protein_codes,
            df[schema.TARGET_KEY].map(protein_position).to_numpy(dtype=np.int64),
        )

    # --- interface ----------------------------------------------------------

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("DeepDTAModel needs PyTorch; install the 'deep' extra") from exc

        config = self.config
        device = torch.device(config.resolve_device())
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        drug_codes, drug_index, protein_codes, protein_index = self._encode_frame(train)
        y = self.targets(train)

        model = _build_network(torch, nn, config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_fn = nn.MSELoss()

        # The code tables live on the device; the indices select rows per batch.
        drug_table = torch.tensor(drug_codes, device=device)
        protein_table = torch.tensor(protein_codes, device=device)
        drug_idx = torch.tensor(drug_index, device=device)
        protein_idx = torch.tensor(protein_index, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        valid_tensors = None
        if valid is not None and len(valid) >= 10:
            v_dc, v_di, v_pc, v_pi = self._encode_frame(valid)
            valid_tensors = (
                torch.tensor(v_dc, device=device),
                torch.tensor(v_di, device=device),
                torch.tensor(v_pc, device=device),
                torch.tensor(v_pi, device=device),
                torch.tensor(self.targets(valid), dtype=torch.float32, device=device),
            )

        n = len(y)
        batch_size = min(config.batch_size, max(2, n))
        history: list[dict[str, float]] = []
        best_score, best_epoch, best_state = np.inf, 0, None

        logger.info(
            "deepdta: %d train rows on %s, batch %d, up to %d epochs",
            n,
            device,
            batch_size,
            config.max_epochs,
        )

        for epoch in range(config.max_epochs):
            model.train()
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                batch = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                rows = torch.as_tensor(batch, device=device)
                loss = loss_fn(
                    model(drug_table[drug_idx[rows]], protein_table[protein_idx[rows]]),
                    y_t[rows],
                )
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)

            record = {"epoch": float(epoch), "train_mse": total / n}
            score = record["train_mse"]
            if valid_tensors is not None:
                model.eval()
                with torch.no_grad():
                    v_dc, v_di, v_pc, v_pi, v_y = valid_tensors
                    chunks = []
                    for start in range(0, len(v_y), 4096):
                        stop = start + 4096
                        chunks.append(
                            model(v_dc[v_di[start:stop]], v_pc[v_pi[start:stop]])
                        )
                    record["valid_rmse"] = float(
                        torch.sqrt(loss_fn(torch.cat(chunks), v_y)).item()
                    )
                score = record["valid_rmse"]
            history.append(record)

            if score < best_score - 1e-5:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= config.patience:
                logger.info("early stopping at epoch %d (best %d, %.4f)", epoch, best_epoch, best_score)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        self._model = model
        self._device = device
        self._mark_fitted()
        return FitResult(
            n_train=n,
            n_valid=len(valid) if valid is not None else 0,
            history=history,
            best_epoch=best_epoch,
            notes={
                "device": str(device),
                "implementation": "reimplementation in PyTorch, not the published code",
                "selection_criterion": "valid_rmse" if valid_tensors else "train_mse",
            },
        )

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        import torch

        drug_codes, drug_index, protein_codes, protein_index = self._encode_frame(test)
        drug_table = torch.tensor(drug_codes, device=self._device)
        protein_table = torch.tensor(protein_codes, device=self._device)
        drug_idx = torch.tensor(drug_index, device=self._device)
        protein_idx = torch.tensor(protein_index, device=self._device)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(test), 4096):
                stop = start + 4096
                outputs.append(
                    self._model(
                        drug_table[drug_idx[start:stop]],
                        protein_table[protein_idx[start:stop]],
                    )
                    .cpu()
                    .numpy()
                )

        return Prediction(
            mean=np.concatenate(outputs),
            metadata={"implementation": "reimplementation", "citation": "Ozturk et al. 2018"},
        )
