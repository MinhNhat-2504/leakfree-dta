"""Run the upstream GraphDTA GINConvNet on this project's explicit split.

The model and molecular/protein feature definitions are copied mechanically from
the pinned GraphDTA repository; only its Davis/KIBA-only data/training wrapper is
replaced. Test labels are never used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def one_hot(value, choices, unknown=False):
    if unknown and value not in choices:
        value = choices[-1]
    if value not in choices:
        raise ValueError(f"{value!r} not in allowable GraphDTA feature values")
    return [value == choice for choice in choices]


def atom_features(atom):
    return np.asarray(
        one_hot(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
                "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge",
                "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg",
                "Pb", "Unknown",
            ],
            unknown=True,
        )
        + one_hot(atom.GetDegree(), list(range(11)))
        + one_hot(atom.GetTotalNumHs(), list(range(11)), unknown=True)
        + one_hot(atom.GetImplicitValence(), list(range(11)), unknown=True)
        + [atom.GetIsAromatic()]
    )


def smiles_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse a curated SMILES: {smiles!r}")
    features = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    features /= features.sum(axis=1, keepdims=True)
    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((start, end), (end, start)))
    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    return torch.from_numpy(features), torch.from_numpy(edge_index)


SEQ_DICT = {letter: i + 1 for i, letter in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")}


def encode_sequence(sequence):
    encoded = np.zeros(1000, dtype=np.int64)
    for i, letter in enumerate(sequence[:1000]):
        if letter not in SEQ_DICT:
            raise ValueError(f"Unrecognized amino-acid character {letter!r}")
        encoded[i] = SEQ_DICT[letter]
    return torch.from_numpy(encoded)


class GraphDTASplit(Dataset):
    def __init__(self, path: Path, graph_cache: dict):
        self.frame = pd.read_csv(path)
        required = {"row_key", "smiles", "target_sequence", "y"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        self.frame["y"] = pd.to_numeric(self.frame["y"], errors="raise")
        self.row_keys = self.frame["row_key"].to_numpy()
        self.smiles_values = self.frame["smiles"].to_numpy()
        self.sequence_values = self.frame["target_sequence"].to_numpy()
        self.labels = self.frame["y"].to_numpy(dtype=np.float32)
        self.graphs = graph_cache
        self.sequences = {}
        smiles = self.frame["smiles"].drop_duplicates()
        for position, value in enumerate(smiles, start=1):
            self.graphs[value] = smiles_graph(value)
            if position % 10000 == 0:
                print(f"molecular graphs cached: {position}/{len(smiles)}", flush=True)
        for value in self.frame["target_sequence"].drop_duplicates():
            self.sequences[value] = encode_sequence(value)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        features, edge_index = self.graphs[self.smiles_values[index]]
        return Data(
            x=features,
            edge_index=edge_index,
            target=self.sequences[self.sequence_values[index]].unsqueeze(0),
            y=torch.tensor([self.labels[index]], dtype=torch.float32),
            row_id=torch.tensor([index], dtype=torch.long),
        )


def predict(model, loader, device):
    model.eval()
    output, ids = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output.extend(model(batch).reshape(-1).detach().cpu().numpy().tolist())
            ids.extend(batch.row_id.reshape(-1).cpu().numpy().tolist())
    order = np.argsort(ids)
    return np.asarray(output, dtype=np.float32)[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from models.ginconv import GINConvNet

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GraphDTA GPU run requested, but CUDA is unavailable")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    graph_cache = {}
    train_set = GraphDTASplit(args.data / "train.csv", graph_cache)
    valid_set = GraphDTASplit(args.data / "valid.csv", graph_cache)
    test_set = GraphDTASplit(args.data / "test.csv", graph_cache)
    loader_args = {"batch_size": args.batch_size, "num_workers": 0}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = GINConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = torch.nn.MSELoss()
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    checkpoint = args.output / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_rows = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = loss_fn(prediction, batch.y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            n_rows += batch.num_graphs

        valid_prediction = predict(model, valid_loader, device)
        valid_truth = valid_set.labels
        rmse = float(np.sqrt(np.mean((valid_prediction - valid_truth) ** 2)))
        print(
            f"epoch={epoch} train_mse={total_loss / max(1, n_rows):.6f} "
            f"valid_rmse={rmse:.6f} gpu_mem_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}",
            flush=True,
        )
        if rmse < best_rmse:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {args.patience} epochs without validation improvement", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_prediction = predict(model, test_loader, device)
    result = test_set.frame[["row_key", "y"]].rename(columns={"y": "y_true"})
    result["y_pred"] = test_prediction
    result.to_csv(args.output / "predictions.csv", index=False)
    metadata = {
        "model": "GraphDTA GINConvNet",
        "upstream_repository": "https://github.com/thinng/GraphDTA",
        "upstream_commit": __import__("subprocess").check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "epochs_max": args.epochs,
        "best_epoch_by_validation": best_epoch,
        "best_validation_rmse": best_rmse,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "train_rows": len(train_set),
        "valid_rows": len(valid_set),
        "test_rows": len(test_set),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
