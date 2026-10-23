"""CUDA DeepDTA upstream builder on explicit train/validation/test CSVs.

Training is resumable at epoch boundaries. Only validation loss selects epochs;
the test CSV is opened after training/selection. Character encoders, architecture,
Adam and MSE come from the pinned upstream source, not a reimplementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import runpy
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def next_state(previous, epoch, loss, train_loss, seconds):
    """Pure selection transition: no test information enters this function."""
    if not math.isfinite(loss) or not math.isfinite(train_loss):
        raise ValueError("Nonfinite training/validation loss; refusing checkpoint")
    improved = previous["best_loss"] is None or loss < previous["best_loss"]
    return {
        "epoch": epoch,
        "best_loss": loss if improved else previous["best_loss"],
        "best_epoch": epoch if improved else previous["best_epoch"],
        "stale": 0 if improved else previous["stale"] + 1,
        "history": previous["history"] + [{
            "epoch": epoch, "loss": train_loss, "val_loss": loss,
            "seconds": seconds,
        }],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch size and patience must be positive")
    args.repo, args.data, args.output = (p.resolve() for p in (args.repo, args.data, args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    # WSL/Linux advisory lock releases automatically even on forced termination.
    import fcntl
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tf_keras as keras
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("DeepDTA CUDA run requested, but no TensorFlow GPU is available")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.modules["keras"] = keras
    source = args.repo / "source" / "run_experiments.py"
    sys.path.insert(0, str(source.parent))
    upstream = runpy.run_path(str(source), run_name="deepdta_upstream")
    print("GPU:", devices, flush=True)
    configuration = {
        "protocol": "explicit_split_validation_selection_v1",
        "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
        "num_filters": 32, "smi_kernel": 4, "seq_kernel": 8,
        "max_smi_len": 100, "max_seq_len": 1000, "seed": 0,
        "train_sha256": sha256(args.data / "train.csv"),
        "valid_sha256": sha256(args.data / "valid.csv"),
        "source_sha256": sha256(source),
        "encoder_sha256": sha256(source.parent / "datahelper.py"),
        "runner_sha256": sha256(__file__),
    }
    config_path = args.output / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Run configuration/input/source changed; use a new output directory")
    else:
        atomic_json(config_path, configuration)

    class PairSequence(keras.utils.Sequence):
        def __init__(self, path):
            frame = pd.read_csv(path)
            required = {"row_key", "smiles", "target_sequence", "y"}
            if not required.issubset(frame.columns) or frame.empty:
                raise ValueError(f"Missing columns or empty split: {path}")
            if frame[list(required)].isna().any().any() or frame.row_key.duplicated().any():
                raise ValueError(f"Missing values or duplicate row keys: {path}")
            self.row_keys = frame.row_key.to_numpy()
            self.labels = frame.y.to_numpy(dtype=np.float32)
            if not np.isfinite(self.labels).all():
                raise ValueError(f"Nonfinite labels: {path}")
            self.drug_ids, smiles = pd.factorize(frame.smiles, sort=False)
            self.protein_ids, sequences = pd.factorize(frame.target_sequence, sort=False)
            self.drugs = np.asarray([
                upstream["label_smiles"](s, 100, upstream["CHARISOSMISET"]) for s in smiles
            ], dtype=np.int32)
            self.proteins = np.asarray([
                upstream["label_sequence"](s, 1000, upstream["CHARPROTSET"]) for s in sequences
            ], dtype=np.int32)
            print(f"{path.name}: {len(self.labels)} rows, {len(smiles)} drugs, "
                  f"{len(sequences)} proteins; unique tokens "
                  f"{(self.drugs.nbytes + self.proteins.nbytes) / 2**20:.1f} MiB", flush=True)

        def __len__(self):
            return math.ceil(len(self.labels) / args.batch_size)

        def __getitem__(self, index):
            selection = slice(index * args.batch_size, (index + 1) * args.batch_size)
            return (self.drugs[self.drug_ids[selection]],
                    self.proteins[self.protein_ids[selection]]), self.labels[selection]

    train = PairSequence(args.data / "train.csv")
    valid = PairSequence(args.data / "valid.csv")
    if set(train.row_keys) & set(valid.row_keys):
        raise ValueError("Train/validation row-key overlap")
    flags = SimpleNamespace(max_smi_len=100, max_seq_len=1000,
                            charsmiset_size=upstream["CHARISOSMILEN"],
                            charseqset_size=upstream["CHARPROTLEN"])
    keras.utils.set_random_seed(0)
    model = upstream["build_combined_categorical"](flags, 32, 4, 8)
    # Materialize optimizer slots before restoring, preserving Adam iterations/moments.
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    elif hasattr(model.optimizer, "_create_all_weights"):
        model.optimizer._create_all_weights(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    _ = checkpoint.save_counter
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    complete = sorted(checkpoints.glob("epoch-[0-9][0-9][0-9][0-9]"))
    state = {"epoch": 0, "best_loss": None, "best_epoch": None, "stale": 0, "history": []}
    if complete:
        latest = complete[-1]
        state = json.loads((latest / "state.json").read_text())
        checkpoint.restore(str(latest / "model")).assert_existing_objects_matched()
        print(f"Resuming epoch {state['epoch'] + 1}; best epoch {state['best_epoch']}, "
              f"validation RMSE {math.sqrt(state['best_loss']):.6f}", flush=True)
    started = time.time()
    while state["epoch"] < args.epochs and state["stale"] < args.patience:
        epoch = state["epoch"] + 1
        # Reset RNGs per epoch so recovery does not reuse the initial epoch seed.
        random.seed(epoch)
        np.random.seed(epoch)
        tf.random.set_seed(epoch)
        epoch_started = time.time()
        history = model.fit(train, validation_data=valid,
                            initial_epoch=epoch - 1, epochs=epoch, shuffle=False,
                            workers=0, use_multiprocessing=False, max_queue_size=1, verbose=2)
        state = next_state(state, epoch, float(history.history["val_loss"][-1]),
                           float(history.history["loss"][-1]), time.time() - epoch_started)
        temporary = checkpoints / ("pending-" + uuid.uuid4().hex)
        temporary.mkdir()
        checkpoint.write(str(temporary / "model"))
        atomic_json(temporary / "state.json", state)
        os.replace(temporary, checkpoints / f"epoch-{epoch:04d}")
        atomic_json(args.output / "progress.json", state)
        print(f"Checkpoint epoch={epoch} best_epoch={state['best_epoch']} "
              f"valid_rmse={math.sqrt(state['best_loss']):.6f} stale={state['stale']}", flush=True)

    # Test labels/features are first loaded after the selected checkpoint is fixed.
    checkpoint.restore(str(checkpoints / f"epoch-{state['best_epoch']:04d}" / "model")).assert_existing_objects_matched()
    test = PairSequence(args.data / "test.csv")
    if set(test.row_keys) & (set(train.row_keys) | set(valid.row_keys)):
        raise ValueError("Test/training or test/validation row-key overlap")
    predictions = []
    for index in range(len(test)):
        features, _ = test[index]
        predictions.append(np.asarray(model(features, training=False)).reshape(-1))
    predictions = np.concatenate(predictions)
    if predictions.shape != test.labels.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid prediction shape/nonfinite predictions")
    prediction_path = args.output / "predictions.csv"
    pd.DataFrame({"row_key": test.row_keys, "y_true": test.labels,
                  "y_pred": predictions}).to_csv(prediction_path.with_suffix(".tmp"), index=False)
    os.replace(prediction_path.with_suffix(".tmp"), prediction_path)
    import subprocess
    metadata = {
        **configuration,
        "model": "DeepDTA upstream build_combined_categorical",
        "upstream_repository": "https://github.com/hkmztrk/DeepDTA",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "tensorflow": tf.__version__, "tf_keras": keras.__version__,
        "device": "cuda", "gpu": tf.config.experimental.get_device_details(devices[0]),
        "train_rows": len(train.labels), "valid_rows": len(valid.labels), "test_rows": len(test.labels),
        "completed_epochs": state["epoch"], "best_epoch_by_validation": state["best_epoch"],
        "best_validation_rmse": math.sqrt(state["best_loss"]),
        "test_sha256": sha256(args.data / "test.csv"), "predictions_sha256": sha256(prediction_path),
        "session_elapsed_seconds": time.time() - started,
        "training_elapsed_seconds": sum(item["seconds"] for item in state["history"]),
        "adaptations": ["Explicit train/valid/test wrapper; validation-only epoch selection",
                        "Unique int32 tokens gathered per batch; batch size 64 by default",
                        "Atomic epoch checkpoints preserve model, Adam, selection and history",
                        "Per-epoch RNG seed; cross-process bitwise determinism is not guaranteed"],
    }
    atomic_json(args.output / "run_metadata.json", metadata)
    atomic_json(args.output / "COMPLETED.json", {
        "status": "completed", "predictions": str(prediction_path),
        "best_epoch_by_validation": state["best_epoch"], "completed_epochs": state["epoch"],
        "predictions_sha256": metadata["predictions_sha256"],
    })
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
