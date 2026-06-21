"""Robustness experiment: Model 2 augmented with synthetic signals from Model 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.metrics import macro_accuracy
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


@dataclass
class RobustnessConfig:
    synthetic_path: str
    held_out: tuple[str, ...]
    model: TemporalCNNConfig
    pdw: PDWConfig
    augment_held_out_only: bool = True
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/gan_signals/robustness"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobustnessConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        model_raw = raw.pop("model")
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        pdw_raw = raw.pop("pdw")
        for key in ("emitters", "modes"):
            if pdw_raw.get(key) is not None:
                pdw_raw[key] = tuple(pdw_raw[key])
        raw["held_out"] = tuple(raw["held_out"])
        return cls(
            model=TemporalCNNConfig(**model_raw),
            pdw=PDWConfig(**pdw_raw),
            **raw,
        )


def _classifier_loss(
    type_logits: torch.Tensor,
    mode_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_mode: torch.Tensor,
) -> torch.Tensor:
    """Compute combined type and mode classification loss.

    Mode loss is computed only over rows with y_mode >= 0 (real signals).
    Synthetic signals with y_mode == -1 do not contribute to mode loss or gradients.
    """
    type_loss = F.cross_entropy(type_logits, y_type)
    valid = y_mode >= 0
    if bool(valid.any()):
        mode_loss = F.cross_entropy(mode_logits[valid], y_mode[valid])
    else:
        mode_loss = type_logits.new_zeros(())
    return type_loss + mode_loss


@torch.no_grad()
def evaluate_type_accuracy(
    model: TemporalCNN,
    dataset: Dataset[Any],
    n_types: int,
    device: str,
) -> float:
    """Compute macro-accuracy of type head over a dataset."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=256)
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y_type, _, _ in loader:
        type_pred, _, _ = model.predict(x.to(dev))
        preds.append(type_pred.cpu())
        targets.append(y_type)
    return macro_accuracy(torch.cat(preds), torch.cat(targets), n_types)


def _fit_classifier(
    model_config: TemporalCNNConfig,
    train_ds: Dataset[Any],
    val_ds: Dataset[Any],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> TemporalCNN:
    """Train a TemporalCNN classifier with early stopping by validation accuracy.

    Returns the model with weights from the epoch with highest type accuracy on val_ds.
    """
    dev = torch.device(device)
    model = TemporalCNN(model_config).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_acc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        for x, y_type, y_mode, _ in train_loader:
            x = x.to(dev)
            y_type = y_type.to(dev)
            y_mode = y_mode.to(dev)
            type_logits, mode_logits = model(x)
            loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        acc = evaluate_type_accuracy(model, val_ds, model_config.n_types, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
