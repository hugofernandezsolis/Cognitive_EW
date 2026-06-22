"""Script de entrenamiento de la Temporal CNN para clasificación ELINT."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.temporal_cnn_elint.metrics import (
    confusion_matrix,
    lpi_accuracy,
    macro_accuracy,
    profile_latency,
)
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


@dataclass
class TrainConfig:
    data: PDWConfig
    model: TemporalCNNConfig
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_weights: tuple[float, float] = (1.0, 1.0)
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/temporal_cnn_elint"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        data_raw = raw.pop("data")
        for key in ("emitters", "modes"):
            if data_raw.get(key) is not None:
                data_raw[key] = tuple(data_raw[key])
        model_raw = raw.pop("model")
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        for key in ("splits", "loss_weights"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(data=PDWConfig(**data_raw), model=TemporalCNNConfig(**model_raw), **raw)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect_preds(
    model: TemporalCNN, loader: DataLoader[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_preds, type_targets, mode_preds, mode_targets = [], [], [], []
    model.eval()
    for x, y_type, y_mode, _ in loader:
        tp, mp, _ = model.predict(x.to(device))
        type_preds.append(tp.cpu())
        mode_preds.append(mp.cpu())
        type_targets.append(y_type)
        mode_targets.append(y_mode)
    return (
        torch.cat(type_preds),
        torch.cat(type_targets),
        torch.cat(mode_preds),
        torch.cat(mode_targets),
    )


def _init_tracking(config: TrainConfig) -> Any:
    import trackio

    return trackio.init(project="temporal_cnn_elint", config=vars(config))


def _run_metadata(config: TrainConfig) -> dict[str, Any]:
    hyperparameters = asdict(config)
    data_blob = json.dumps(asdict(config.data), sort_keys=True).encode()
    return {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "data_config_hash": hashlib.sha256(data_blob).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    device = torch.device(config.device)

    dataset = PDWSyntheticDataset(config.data)
    train_ds, val_ds, test_ds = split_dataset(dataset, config.splits, config.seed)
    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader: DataLoader[Any] = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader: DataLoader[Any] = DataLoader(test_ds, batch_size=config.batch_size)

    model = TemporalCNN(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    w_type, w_mode = config.loss_weights

    run = _init_tracking(config) if config.tracking else None
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))
    best_path = out_dir / "best.pt"
    best_val = -1.0
    history: list[float] = []

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for x, y_type, y_mode, _ in train_loader:
            x = x.to(device)
            y_type = y_type.to(device)
            y_mode = y_mode.to(device)
            type_logits, mode_logits = model(x)
            loss = w_type * F.cross_entropy(type_logits, y_type) + w_mode * F.cross_entropy(
                mode_logits, y_mode
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        history.append(train_loss)

        tp, tt, mp, mt = _collect_preds(model, val_loader, device)
        val_type = macro_accuracy(tp, tt, config.model.n_types)
        val_mode = macro_accuracy(mp, mt, config.model.n_modes)
        val_score = 0.5 * (val_type + val_mode)
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_macro_acc_type": val_type,
                    "val_macro_acc_mode": val_mode,
                }
            )
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, weights_only=True))
    library = EmitterLibrary.from_yaml(config.data.library_path)
    tp, tt, mp, mt = _collect_preds(model, test_loader, device)
    test_metrics: dict[str, Any] = {
        "macro_acc_type": macro_accuracy(tp, tt, config.model.n_types),
        "macro_acc_mode": macro_accuracy(mp, mt, config.model.n_modes),
        "lpi_accuracy": lpi_accuracy(tp, tt, library.lpi_indices()),
        "confusion_type": confusion_matrix(tp, tt, config.model.n_types).tolist(),
        "confusion_mode": confusion_matrix(mp, mt, config.model.n_modes).tolist(),
    }
    sample = next(iter(test_loader))[0][:1].to(device)
    mean_ms, p99_ms = profile_latency(model, sample, n_warmup=5, n_iter=50, device=config.device)
    test_metrics["latency_mean_ms"] = mean_ms
    test_metrics["latency_p99_ms"] = p99_ms

    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if run is not None:
        run.finish()
    return {"test": test_metrics, "train_loss_history": history}
