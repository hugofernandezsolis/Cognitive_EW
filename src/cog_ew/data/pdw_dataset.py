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
                    pri = toa_to_pri(train.toa)
                    cont = np.stack([train.rf, train.pw, train.pa, train.aoa, pri], axis=1)
                    if config.normalize:
                        cont = normalize_pdw(cont, CONTINUOUS_RANGES)
                    onehot = one_hot_intra_pulse(train.intra_pulse_mod, len(INTRA_PULSE_MODS))
                    feat = np.concatenate([cont, onehot], axis=1)
                    for window in window_sequence(feat, config.window):
                        windows.append(window)
                        types.append(type_idx)
                        modes.append(mode_idx)
                        threats.append(threat_idx)

        self._x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        self._type = torch.tensor(types, dtype=torch.int64)
        self._mode = torch.tensor(modes, dtype=torch.int64)
        self._threat = torch.tensor(threats, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        return (
            self._x[index],
            int(self._type[index]),
            int(self._mode[index]),
            int(self._threat[index]),
        )
