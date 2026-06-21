"""Robustness experiment: Model 2 augmented with synthetic signals from Model 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig


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
