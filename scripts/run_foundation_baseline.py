"""Handcrafted-feature MLP baseline: ECFP4 + dipeptide frequencies.

Despite the historical filename, this implementation does not load ChemBERTa-2
or ESM-2 and must not be reported as a foundation-model result. It is retained as
an optional ablation until a real pretrained-embedding pipeline is implemented.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakfree_dta.data import schema
from leakfree_dta.eval.metrics import evaluate
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.models.features import FeatureSpec, build_features
from leakfree_dta.utils.io import write_json, write_table

logger = get_logger(__name__)


class InteractionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing train.csv, valid.csv, test.csv")
    parser.add_argument("--output", required=True, help="Directory to save predictions.csv and metrics.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=100_000)
    args = parser.parse_args()

    setup_logging("INFO")
    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    test_path = data_dir / "test.csv"

    if not (train_path.exists() and test_path.exists()):
        raise SystemExit(f"train.csv and test.csv must exist in {data_dir}")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path) if valid_path.exists() and valid_path.stat().st_size > 0 else pd.DataFrame()
    test_df = pd.read_csv(test_path)

    for df in (train_df, valid_df, test_df):
        if not df.empty and schema.LIGAND_KEY not in df.columns:
            df[schema.LIGAND_KEY] = df[schema.SMILES]

    logger.info(
        "ECFP4 + dipeptide MLP baseline: train=%d, valid=%d, test=%d",
        len(train_df), len(valid_df), len(test_df)
    )

    # Compute high-capacity combined embeddings (ECFP4-1024 + 400-dim dipeptide protein representation)
    spec = FeatureSpec(ligand="ecfp4", protein="dipeptide", n_bits=1024)
    logger.info("Extracting feature embeddings: %s", spec.describe())

    x_train = build_features(train_df, spec)
    y_train = pd.to_numeric(train_df[schema.Y], errors="coerce").to_numpy(dtype=float)
    x_test = build_features(test_df, spec)
    y_test = pd.to_numeric(test_df[schema.Y], errors="coerce").to_numpy(dtype=float)

    if not valid_df.empty and len(valid_df) > 10:
        x_val = build_features(valid_df, spec)
        y_val = pd.to_numeric(valid_df[schema.Y], errors="coerce").to_numpy(dtype=float)
    else:
        # Use 10% of train for validation
        split_idx = int(0.9 * len(x_train))
        x_val, y_val = x_train[split_idx:], y_train[split_idx:]
        x_train, y_train = x_train[:split_idx], y_train[:split_idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    in_dim = x_train.shape[1]
    model = InteractionHead(in_features=in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None

    start_time = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(bx)
        val_loss /= len(val_loader.dataset)

        logger.info("Epoch %2d/%2d - Train Loss: %.4f, Val Loss: %.4f", epoch, args.epochs, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Inference on test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, _by in test_loader:
            bx = bx.to(device)
            preds = model(bx)
            test_preds.extend(preds.cpu().numpy().tolist())

    y_pred = np.array(test_preds, dtype=float)
    metrics = evaluate(y_test, y_pred)
    elapsed = time.monotonic() - start_time

    logger.info(
        "ECFP4 + dipeptide MLP results: RMSE=%.4f, Pearson=%.4f, Spearman=%.4f (%.1f min)",
        metrics.get("rmse"), metrics.get("pearson"), metrics.get("spearman"), elapsed / 60
    )

    pred_out = pd.DataFrame({
        schema.ROW_KEY: test_df[schema.ROW_KEY].to_numpy() if schema.ROW_KEY in test_df.columns else range(len(test_df)),
        "y_true": y_test,
        "y_pred": y_pred,
    })
    pred_path = out_dir / "predictions.csv"
    pred_out.to_csv(pred_path, index=False)
    write_json(metrics.to_dict(), out_dir / "metrics.json")
    write_table(pred_out, out_dir / "predictions.parquet")

    logger.info("Predictions saved to %s", pred_path)
    return 0


if __name__ == "__main__":
    main()
