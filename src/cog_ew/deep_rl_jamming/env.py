"""Entorno RL que simula el ciclo radar amenaza (PRI, frecuencia, modos ECCM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RadarEnvConfig:
    library_path: str
    effectiveness: dict[str, dict[str, float]]
    emitters: tuple[str, ...] | None = None
    history_k: int = 8
    horizon_t: int = 64
    power_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    burnthrough: float = 15.0
    eff_threshold: float = 0.5
    js_scale: float = 20.0
    lock_gain: float = 0.15
    lock_decay: float = 0.15
    n_eccm: int = 3
    w_eff: float = 1.0
    lambda_power: float = 0.5
    r_win: float = 10.0
    r_lose: float = 10.0
    obs_noise_std: float = 0.0
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RadarEnvConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("emitters") is not None:
            raw["emitters"] = tuple(raw["emitters"])
        if "power_levels" in raw:
            raw["power_levels"] = tuple(float(p) for p in raw["power_levels"])
        return cls(**raw)
