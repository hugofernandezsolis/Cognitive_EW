"""Script de entrenamiento de la Temporal CNN para clasificación ELINT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig


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
