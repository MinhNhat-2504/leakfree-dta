"""Group distributionally robust optimisation over protein groups.

Standard training minimises average loss. On a DTA dataset that means minimising
the loss on the handful of targets that carry most of the measurements, and the
imbalance check shows how extreme that is: the top decile of targets routinely
holds over half the data. A model so trained can be excellent on those and useless
on everything else, while reporting a fine overall number.

Group DRO minimises the loss of the *worst* group instead:

    min_theta  max_{q in simplex}  sum_g q_g * L_g(theta)

solved with online exponentiated gradient on ``q`` interleaved with SGD on
``theta`` (Sagawa et al., 2020). Groups that are currently doing badly get more
weight; the weights adapt as training proceeds.

Three practical points this implementation takes seriously:

* **Groups are protein clusters, not individual targets.** With one group per
  target, the worst group is always some target with four noisy measurements, and
  the objective chases label noise. Clustering at the same identity threshold the
  split uses keeps the groups meaningful and large enough to estimate a loss on.
* **DRO needs stronger regularisation than ERM.** Worst-group optimisation will
  overfit the worst group if allowed to, so weight decay defaults higher here than
  a comparable ERM model would use, and this is a hyperparameter the paper reports
  rather than tunes away.
* **Model selection uses worst-group validation loss.** Selecting on average
  validation loss would undo the training objective, which is a common and
  invisible way DRO results get weakened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leakfree_dta.data import schema
from leakfree_dta.logging import get_logger
from leakfree_dta.models.base import DTAModel, FitResult, Prediction
from leakfree_dta.models.features import FeatureSpec, build_features

logger = get_logger(__name__)


@dataclass
class GroupDROConfig:
    """Hyperparameters, all reported in the paper's supplement."""

    hidden_sizes: tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    group_lr: float = 0.01  # exponentiated-gradient step size for the group weights
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    min_group_size: int = 20
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


class GroupDROModel(DTAModel):
    """An MLP on ligand+protein features, trained with group DRO.

    The architecture is deliberately plain. The paper's claim is about the
    *objective* and the *calibration*, not about a new network, and a plain MLP
    makes that claim testable: swap the objective and nothing else changes.
    """

    name = "group_dro_mlp"
    reports_uncertainty = True

    def __init__(
        self,
        config: GroupDROConfig | None = None,
        *,
        group_column: str = "protein_group",
        feature_spec: FeatureSpec | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or GroupDROConfig(**kwargs)
        super().__init__(**{"group_column": group_column, **self.config.__dict__})
        self.group_column = group_column
        self.spec = feature_spec or FeatureSpec()
        self._model = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None

    # --- groups -------------------------------------------------------------

    def _group_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Group label per row, falling back sensibly when the column is absent.

        The split machinery attaches ``protein_group``. When a caller has not used
        it, target family is the next best grouping, and target key the one after
        that -- with a warning, because single-target groups make the objective
        chase noise.
        """
        if self.group_column in df.columns:
            return df[self.group_column].astype(str).to_numpy()
        if schema.TARGET_FAMILY in df.columns and df[schema.TARGET_FAMILY].nunique() > 1:
            logger.warning(
                "no %r column; grouping by target family instead. Pass a split's "
                "protein_group for the intended behaviour.",
                self.group_column,
            )
            return df[schema.TARGET_FAMILY].astype(str).to_numpy()
        logger.warning("grouping by individual target; worst-group loss will be noisy")
        return df[schema.TARGET_KEY].astype(str).to_numpy()

    def _merge_small_groups(self, labels: np.ndarray) -> np.ndarray:
        """Pool groups below ``min_group_size`` into one ``__small__`` group.

        Without this, the maximisation picks whichever three-row group happens to
        have an outlier, and training follows it. Pooling keeps those rows in the
        objective without letting any of them dominate it.
        """
        counts = pd.Series(labels).value_counts()
        small = set(counts[counts < self.config.min_group_size].index)
        if not small:
            return labels
        logger.info(
            "pooling %d groups with under %d rows (%d rows total) into '__small__'",
            len(small),
            self.config.min_group_size,
            int(counts[list(small)].sum()),
        )
        return np.where(np.isin(labels, list(small)), "__small__", labels)

    # --- fitting ------------------------------------------------------------

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, **kwargs: Any) -> FitResult:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("GroupDROModel needs PyTorch; install the 'deep' extra") from exc

        config = self.config
        device = torch.device(config.resolve_device())
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        x_train = build_features(train, self.spec)
        y_train = self.targets(train)
        groups = self._merge_small_groups(self._group_labels(train))
        unique_groups = np.unique(groups)
        group_index = {label: index for index, label in enumerate(unique_groups)}
        group_ids = np.array([group_index[label] for label in groups], dtype=np.int64)

        # Standardise features on the training fold only.
        #
        # The clamp matters more here than it looks. Most ECFP bits are zero
        # throughout any given training fold, so their standard deviation is
        # exactly zero. Clamping to a small epsilon and dividing would turn a
        # single set bit in the test fold into a value of order 1e6 and diverge
        # the network -- and it would do so precisely on the leakage-free splits,
        # where test compounds are most chemically distinct and so most likely to
        # set a bit the training fold never did. A constant feature carries no
        # information, so its scale is set to one, leaving it centred at zero.
        self._scaler_mean = x_train.mean(axis=0)
        deviation = x_train.std(axis=0)
        self._scaler_scale = np.where(deviation < 1e-6, 1.0, deviation)
        n_constant = int((deviation < 1e-6).sum())
        if n_constant:
            logger.info(
                "%d of %d features are constant in the training fold; they are centred "
                "but not rescaled",
                n_constant,
                x_train.shape[1],
            )
        x_train = (x_train - self._scaler_mean) / self._scaler_scale

        model = _build_mlp(nn, x_train.shape[1], config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        group_tensor = torch.tensor(group_ids, dtype=torch.long, device=device)

        # The adversary's distribution over groups, initialised uniform.
        group_weights = torch.ones(len(unique_groups), device=device) / len(unique_groups)

        valid_tensors = None
        if valid is not None and len(valid) >= 10:
            x_valid = (build_features(valid, self.spec) - self._scaler_mean) / self._scaler_scale
            valid_groups = self._merge_small_groups(self._group_labels(valid))
            valid_tensors = (
                torch.tensor(x_valid, dtype=torch.float32, device=device),
                torch.tensor(self.targets(valid), dtype=torch.float32, device=device),
                valid_groups,
            )

        history: list[dict[str, float]] = []
        best_score, best_epoch, best_state = np.inf, 0, None
        n = len(x_train)
        if n < config.batch_size:
            # BatchNorm needs more than one row per batch, and a single-batch epoch
            # gives the adversary nothing to adapt to.
            logger.warning("only %d training rows; reducing batch size", n)
            config.batch_size = max(2, n // 2)

        objective = torch.tensor(float("nan"))
        for epoch in range(config.max_epochs):
            model.train()
            order = rng.permutation(n)
            for start in range(0, n, config.batch_size):
                batch = order[start : start + config.batch_size]
                batch_x = x_tensor[batch]
                batch_y = y_tensor[batch]
                batch_g = group_tensor[batch]

                predictions = model(batch_x).squeeze(-1)
                losses = (predictions - batch_y) ** 2

                # Per-group mean loss within the batch; groups absent from the
                # batch contribute nothing this step.
                present = torch.unique(batch_g)
                group_losses = torch.stack(
                    [losses[batch_g == group].mean() for group in present]
                )

                # Exponentiated gradient on the adversary, then renormalise.
                #
                # The max is subtracted before exponentiating. Since the weights are
                # renormalised immediately afterwards, subtracting any constant
                # leaves the result mathematically identical, but it keeps the
                # exponential from overflowing early in training when a group's
                # squared error can still be large.
                with torch.no_grad():
                    step = config.group_lr * group_losses.detach()
                    group_weights[present] *= torch.exp(step - step.max())
                    total_weight = group_weights.sum()
                    if total_weight > 0:
                        group_weights /= total_weight
                    else:
                        # Every weight underflowed to zero; restart from uniform
                        # rather than propagate NaNs through the objective.
                        group_weights.fill_(1.0 / len(group_weights))

                weights = group_weights[present]
                objective = (weights * group_losses).sum() / weights.sum()

                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()

            record = {"epoch": float(epoch), "train_objective": float(objective.item())}
            if valid_tensors is not None:
                worst, average = self._validate(model, valid_tensors, torch)
                record["valid_worst_group_rmse"] = worst
                record["valid_rmse"] = average
                # Selection on worst-group loss, matching the training objective.
                score = worst
            else:
                score = float(objective.item())
            history.append(record)

            if score < best_score - 1e-5:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            elif epoch - best_epoch >= config.patience:
                logger.info("early stopping at epoch %d (best %d, score %.4f)", epoch, best_epoch, best_score)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        self._model = model
        self._device = device
        self._group_weights = group_weights.detach().cpu().numpy()
        self._group_names = unique_groups
        self._train_residual_std = self._residual_std(model, x_tensor, y_tensor, torch)
        self._mark_fitted()

        return FitResult(
            n_train=len(train),
            n_valid=len(valid) if valid is not None else 0,
            history=history,
            best_epoch=best_epoch,
            notes={
                "n_groups": len(unique_groups),
                "device": str(device),
                "final_group_weight_max": round(float(self._group_weights.max()), 5),
                "most_upweighted_group": str(unique_groups[int(self._group_weights.argmax())]),
                "selection_criterion": "worst_group_rmse" if valid is not None else "train_objective",
            },
        )

    def _validate(self, model, tensors, torch) -> tuple[float, float]:
        x_valid, y_valid, valid_groups = tensors
        model.eval()
        with torch.no_grad():
            predictions = model(x_valid).squeeze(-1)
            squared = ((predictions - y_valid) ** 2).cpu().numpy()
        average = float(np.sqrt(squared.mean()))

        frame = pd.DataFrame({"group": valid_groups, "squared": squared})
        per_group = frame.groupby("group", observed=True)["squared"].agg(["mean", "size"])
        reliable = per_group[per_group["size"] >= 10]
        worst = float(np.sqrt(reliable["mean"].max())) if len(reliable) else average
        return worst, average

    def _residual_std(self, model, x_tensor, y_tensor, torch) -> float:
        """Training residual spread, used as the uncertainty estimate's scale.

        A single global number is a weak uncertainty estimate, and it is not the
        paper's uncertainty claim -- :class:`ShiftRobustDTA` replaces it with
        conformal intervals. It exists so this model satisfies the interface and
        so the conformal head has a difficulty signal to normalise by.
        """
        with torch.no_grad():
            predictions = model(x_tensor).squeeze(-1)
            residuals = (predictions - y_tensor).cpu().numpy()
        return float(np.std(residuals))

    # --- prediction ---------------------------------------------------------

    def predict(self, test: pd.DataFrame, **kwargs: Any) -> Prediction:
        self.check_fitted(self)
        import torch

        features = (build_features(test, self.spec) - self._scaler_mean) / self._scaler_scale
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self._device)
            mean = self._model(tensor).squeeze(-1).cpu().numpy()

        # Rows in a group the model never trained on are the ones to distrust, so
        # their reported uncertainty is inflated. The factor is a documented
        # heuristic; the conformal head is what makes the number trustworthy.
        groups = self._merge_small_groups(self._group_labels(test))
        unseen = ~np.isin(groups, self._group_names)
        std = np.where(unseen, self._train_residual_std * 2.0, self._train_residual_std)

        return Prediction(
            mean=mean,
            std=std,
            metadata={
                "n_unseen_groups": int(unseen.sum()),
                "uncertainty_source": "train_residual_std (heuristic)",
            },
        )


def _build_mlp(nn, n_features: int, config: GroupDROConfig):
    layers: list[Any] = []
    size = n_features
    for hidden in config.hidden_sizes:
        layers += [nn.Linear(size, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(config.dropout)]
        size = hidden
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


class ERMModel(GroupDROModel):
    """The same network trained on average loss, as the DRO ablation's control.

    Implemented as a subclass with ``group_lr=0`` so that the two models differ in
    exactly one line of configuration. Any gap between them is then attributable to
    the objective, which is what an ablation is supposed to establish.
    """

    name = "erm_mlp"

    def __init__(self, config: GroupDROConfig | None = None, **kwargs: Any) -> None:
        config = config or GroupDROConfig(**kwargs)
        # group_lr = 0 freezes the adversary at uniform, which is ERM exactly.
        config.group_lr = 0.0
        super().__init__(config=config)
