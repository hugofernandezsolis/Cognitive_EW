"""Dataset sintético de secuencias PDW etiquetadas (tipo / modo / amenaza)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from cog_ew.data.pdw_generator import generate_pulse_train
from cog_ew.data.pdw_library import (
    CONTINUOUS_RANGES,
    INTRA_PULSE_MODS,
    MODES,
    EmitterLibrary,
    mode_to_threat,
)
from cog_ew.data.preprocessing import (
    normalize_pdw,
    one_hot_intra_pulse,
    toa_to_pri,
    window_sequence,
)


@dataclass
class PDWConfig:
    library_path: str
    emitters: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None
    window: int = 64
    n_pulses: int = 256
    n_trains: int = 8
    normalize: bool = True
    noise_std: float = 0.0
    drop_prob: float = 0.0
    spurious_prob: float = 0.0
    seed: int = 0
    feature_set: str = "base"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PDWConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        for key in ("emitters", "modes"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)


class PDWSyntheticDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    def __init__(self, config: PDWConfig) -> None:
        if config.feature_set not in {"base", "v2"}:
            raise ValueError("feature_set must be 'base' or 'v2'")

        self.config = config
        library = EmitterLibrary.from_yaml(config.library_path)
        names = library.emitter_names()
        rng = np.random.default_rng(config.seed)

        windows: list[np.ndarray] = []
        types: list[int] = []
        modes: list[int] = []
        threats: list[int] = []

        for emitter in library.emitters:
            if config.emitters is not None and emitter.name not in config.emitters:
                continue
            type_idx = names.index(emitter.name)
            for mode_name, mode_spec in emitter.modes.items():
                if config.modes is not None and mode_name not in config.modes:
                    continue
                mode_idx = MODES.index(mode_name)
                threat_idx = mode_to_threat(mode_name)
                for _ in range(config.n_trains):
                    train = generate_pulse_train(
                        mode_spec,
                        config.n_pulses,
                        rng,
                        noise_std=config.noise_std,
                        drop_prob=config.drop_prob,
                        spurious_prob=config.spurious_prob,
                    )
                    if train.toa.shape[0] < config.window:
                        continue
                    feat = (
                        self._v2_features(train, mode_name)
                        if config.feature_set == "v2"
                        else self._base_features(train)
                    )
                    for window in window_sequence(feat, config.window):
                        windows.append(window)
                        types.append(type_idx)
                        modes.append(mode_idx)
                        threats.append(threat_idx)

        self._x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        self._type = torch.tensor(types, dtype=torch.int64)
        self._mode = torch.tensor(modes, dtype=torch.int64)
        self._threat = torch.tensor(threats, dtype=torch.int64)

    def _base_features(self, train) -> np.ndarray:
        pri = toa_to_pri(train.toa)
        cont = np.stack([train.rf, train.pw, train.pa, train.aoa, pri], axis=1)
        if self.config.normalize:
            cont = normalize_pdw(cont, CONTINUOUS_RANGES)
        onehot = one_hot_intra_pulse(train.intra_pulse_mod, len(INTRA_PULSE_MODS))
        return np.concatenate([cont, onehot], axis=1).astype(np.float32)

    def _v2_features(self, train, mode_name: str) -> np.ndarray:
        base = self._base_features(train)
        rf = base[:, 0]
        pw = base[:, 1]
        pri = base[:, 4]
        delta_rf = _prepend_zero(np.diff(rf))
        delta_pri = _prepend_zero(np.diff(pri))
        rolling_pri_std = _rolling_std(pri, width=8)
        rolling_rf_std = _rolling_std(rf, width=8)
        rolling_pw_mean = _rolling_mean(pw, width=8)
        pulse_progression = _pulse_progression(base.shape[0])
        lpi_hint = _lpi_hint(train)
        freq_hopping_hint = _freq_hopping_hint(rf)
        extra = np.stack(
            [
                delta_rf,
                delta_pri,
                rolling_pri_std,
                rolling_rf_std,
                rolling_pw_mean,
                pulse_progression,
                lpi_hint,
                freq_hopping_hint,
            ],
            axis=1,
        )
        return np.concatenate([base, extra], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        return (
            self._x[index],
            int(self._type[index]),
            int(self._mode[index]),
            int(self._threat[index]),
        )


def _prepend_zero(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=np.float32), values.astype(np.float32)])


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[0]):
        start = max(0, i + 1 - width)
        out[i] = float(np.mean(values[start : i + 1]))
    return out


def _rolling_std(values: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[0]):
        start = max(0, i + 1 - width)
        out[i] = float(np.std(values[start : i + 1]))
    return out


def _pulse_progression(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _lpi_hint(train) -> np.ndarray:
    bandwidth = float(np.std(train.rf))
    duty_hint = float(np.mean(train.pw) / max(np.mean(toa_to_pri(train.toa)), 1e-6))
    raw = 0.5 * np.tanh(bandwidth / 120.0) + 0.5 * np.tanh(duty_hint * 20.0)
    return np.full(train.rf.shape[0], np.clip(raw, 0.0, 1.0), dtype=np.float32)


def _freq_hopping_hint(rf: np.ndarray) -> np.ndarray:
    return np.tanh(np.abs(_prepend_zero(np.diff(rf))) * 4.0).astype(np.float32)
