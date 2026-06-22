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
    strict_elint_score,
)
from cog_ew.temporal_cnn_elint.model import (
    TemporalCNN,
    TemporalCNNConfig,
    TemporalCNNV2,
    TemporalCNNV2Config,
)

ModelConfig = TemporalCNNConfig | TemporalCNNV2Config


@dataclass
class TrainConfig:
    data: PDWConfig
    model: ModelConfig
    architecture: str = "v1"
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_weights: tuple[float, ...] = (1.0, 1.0)
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
        architecture = raw.pop("architecture", "v1")
        model_raw = raw.pop("model")
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        for key in ("splits", "loss_weights"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if architecture == "v2":
            model: ModelConfig = TemporalCNNV2Config(**model_raw)
        elif architecture == "v1":
            model = TemporalCNNConfig(**model_raw)
        else:
            raise ValueError("architecture must be 'v1' or 'v2'")
        return cls(data=PDWConfig(**data_raw), model=model, architecture=architecture, **raw)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(config: TrainConfig) -> TemporalCNN | TemporalCNNV2:
    if config.architecture == "v2":
        if not isinstance(config.model, TemporalCNNV2Config):
            raise TypeError("v2 architecture requires TemporalCNNV2Config")
        return TemporalCNNV2(config.model)
    if not isinstance(config.model, TemporalCNNConfig):
        raise TypeError("v1 architecture requires TemporalCNNConfig")
    return TemporalCNN(config.model)


def _collect_preds(
    model: TemporalCNN | TemporalCNNV2,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_preds, type_targets, mode_preds, mode_targets = [], [], [], []
    threat_preds, threat_targets = [], []
    model.eval()
    with torch.inference_mode():
        for x, y_type, y_mode, y_threat in loader:
            tp, mp, thp = model.predict(x.to(device))
            type_preds.append(tp.cpu())
            mode_preds.append(mp.cpu())
            threat_preds.append(thp.cpu())
            type_targets.append(y_type)
            mode_targets.append(y_mode)
            threat_targets.append(y_threat)
    return (
        torch.cat(type_preds),
        torch.cat(type_targets),
        torch.cat(mode_preds),
        torch.cat(mode_targets),
        torch.cat(threat_preds),
        torch.cat(threat_targets),
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

    model = _build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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
        for x, y_type, y_mode, y_threat in train_loader:
            x = x.to(device)
            y_type = y_type.to(device)
            y_mode = y_mode.to(device)
            y_threat = y_threat.to(device)
            if config.architecture == "v2":
                w_type, w_mode, w_threat = config.loss_weights[:3]
                type_logits, mode_logits, threat_logits = model(x)
                loss = (
                    w_type * F.cross_entropy(type_logits, y_type)
                    + w_mode * F.cross_entropy(mode_logits, y_mode)
                    + w_threat * F.cross_entropy(threat_logits, y_threat)
                )
            else:
                w_type, w_mode = config.loss_weights[:2]
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

        tp, tt, mp, mt, thp, tht = _collect_preds(model, val_loader, device)
        val_type = macro_accuracy(tp, tt, config.model.n_types)
        val_mode = macro_accuracy(mp, mt, config.model.n_modes)
        n_threats = getattr(config.model, "n_threats", 4)
        val_threat = macro_accuracy(thp, tht, n_threats)
        if config.architecture == "v2":
            val_score = (val_type + val_mode + val_threat) / 3.0
        else:
            val_score = 0.5 * (val_type + val_mode)
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_macro_acc_type": val_type,
                    "val_macro_acc_mode": val_mode,
                    "val_macro_acc_threat": val_threat,
                }
            )
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, weights_only=True))
    library = EmitterLibrary.from_yaml(config.data.library_path)
    tp, tt, mp, mt, thp, tht = _collect_preds(model, test_loader, device)
    n_threats = getattr(config.model, "n_threats", 4)
    test_metrics: dict[str, Any] = {
        "macro_acc_type": macro_accuracy(tp, tt, config.model.n_types),
        "macro_acc_mode": macro_accuracy(mp, mt, config.model.n_modes),
        "macro_acc_threat": macro_accuracy(thp, tht, n_threats),
        "lpi_accuracy": lpi_accuracy(tp, tt, library.lpi_indices()),
        "confusion_type": confusion_matrix(tp, tt, config.model.n_types).tolist(),
        "confusion_mode": confusion_matrix(mp, mt, config.model.n_modes).tolist(),
        "confusion_threat": confusion_matrix(thp, tht, n_threats).tolist(),
    }
    sample = next(iter(test_loader))[0][:1].to(device)
    mean_ms, p99_ms = profile_latency(model, sample, n_warmup=5, n_iter=50, device=config.device)
    test_metrics["latency_mean_ms"] = mean_ms
    test_metrics["latency_p99_ms"] = p99_ms
    test_metrics["strict_elint_score"] = strict_elint_score(test_metrics)

    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if run is not None:
        run.finish()
    return {"test": test_metrics, "train_loss_history": history}
