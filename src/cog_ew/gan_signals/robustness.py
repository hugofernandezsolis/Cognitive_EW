"""Robustness experiment: Model 2 augmented with synthetic signals from Model 4."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.data.synthetic_loader import SyntheticPDWDataset
from cog_ew.temporal_cnn_elint.metrics import macro_accuracy, profile_latency
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


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robustness_improvement_score(baseline: float, augmented: float) -> float:
    """Return a finite robustness score for the M4 anchor.

    Use relative improvement when the baseline has support. If the baseline is zero,
    the relative ratio is undefined, so the absolute gain is the meaningful anchor.
    """
    delta = augmented - baseline
    if baseline > 0:
        return delta / baseline
    return delta


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_robustness_experiment(config: RobustnessConfig) -> dict[str, Any]:
    """Train baseline and augmented classifiers; compare macro type accuracy on held-out emitters.

    The only difference between baseline and augmented is that the augmented training set
    is extended with synthetic signals from the held-out emitters (produced by the GAN).
    Both fits start from the same seed so any accuracy difference is attributable to the data.
    """
    library = EmitterLibrary.from_yaml(config.pdw.library_path)
    names = library.emitter_names()
    held_out_ids = tuple(names.index(n) for n in config.held_out)
    catalogued = tuple(n for n in names if n not in config.held_out)

    cat_ds = PDWSyntheticDataset(replace(config.pdw, emitters=catalogued))
    train_ds, val_ds = split_dataset(cat_ds, (0.85, 0.15), config.seed)
    held_ds = PDWSyntheticDataset(replace(config.pdw, emitters=config.held_out))

    aug_emitters = held_out_ids if config.augment_held_out_only else None
    synth_ds = SyntheticPDWDataset(config.synthetic_path, emitters=aug_emitters, known_only=True)

    n_types = config.model.n_types
    fit_kwargs: dict[str, Any] = dict(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
    )

    _set_seeds(config.seed)
    base_model = _fit_classifier(config.model, train_ds, val_ds, **fit_kwargs)
    base_acc = evaluate_type_accuracy(base_model, held_ds, n_types, config.device)

    _set_seeds(config.seed)
    aug_model = _fit_classifier(
        config.model, ConcatDataset([train_ds, synth_ds]), val_ds, **fit_kwargs
    )
    aug_acc = evaluate_type_accuracy(aug_model, held_ds, n_types, config.device)

    global_eval: Dataset[Any] = ConcatDataset([val_ds, held_ds])
    delta = aug_acc - base_acc
    rel = robustness_improvement_score(base_acc, aug_acc)
    metrics: dict[str, Any] = {
        "baseline": base_acc,
        "augmented": aug_acc,
        "delta": delta,
        "relative_improvement": rel,
        "global": {
            "baseline": evaluate_type_accuracy(base_model, global_eval, n_types, config.device),
            "augmented": evaluate_type_accuracy(aug_model, global_eval, n_types, config.device),
        },
    }

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = held_ds[0][0].unsqueeze(0)
    mean_ms, p99_ms = profile_latency(
        aug_model, sample, n_warmup=5, n_iter=50, device=config.device
    )
    # Latency goes to disk only — not in the returned dict
    on_disk = {**metrics, "latency_mean_ms": mean_ms, "latency_p99_ms": p99_ms}
    (out_dir / "metrics.json").write_text(json.dumps(on_disk, indent=2))

    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    run_meta: dict[str, Any] = {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "synthetic_hash": _file_sha256(config.synthetic_path),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    return metrics
