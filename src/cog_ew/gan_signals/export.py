"""Export masivo de señales PDW sintéticas a HDF5 (Modelo 4, sub-pieza B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExportConfig:
    checkpoint: str
    z_dim: int = 64
    e_dim: int = 16
    channels: int = 64
    n_emitters: int = 8
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75)
    extrapolate: bool = False
    samples_per_type: int = 2500
    out_path: str = "data/synthetic/wgan_gp.h5"
    library_path: str = "configs/temporal_cnn_elint/emitters.yaml"
    n_real_compare: int = 4000
    seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExportConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("alphas") is not None:
            raw["alphas"] = tuple(raw["alphas"])
        return cls(**raw)
