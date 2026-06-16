"""Entorno RL que simula el ciclo radar amenaza (PRI, frecuencia, modos ECCM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from numpy.typing import NDArray

from cog_ew.data.pdw_library import CONTINUOUS_RANGES, MODES, EmitterLibrary, EmitterSpec
from cog_ew.deep_rl_jamming.threat import RadarState
from cog_ew.ew_library.library import JammingTechnique

_SCAN_MAX = 15.0
_N_FEATURES = 5


def _normalize(value: float, value_range: NDArray[np.float64]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


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


class RadarJammingEnv(gym.Env[NDArray[np.float32], int]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: RadarEnvConfig) -> None:
        super().__init__()
        self.config = config
        library = EmitterLibrary.from_yaml(config.library_path)
        if config.emitters is not None:
            self._candidates = tuple(e for e in library.emitters if e.name in config.emitters)
        else:
            self._candidates = library.emitters
        self._techniques = list(JammingTechnique)
        self._n_power = len(config.power_levels)
        self.action_space = gym.spaces.Discrete(len(self._techniques) * self._n_power)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(config.history_k, _N_FEATURES), dtype=np.float32
        )
        self._emitter: EmitterSpec = self._candidates[0]
        self._ladder: tuple[str, ...] = ()
        self._state = RadarState()
        self._t = 0
        self._history = np.zeros((config.history_k, _N_FEATURES), dtype=np.float32)

    def encode_action(self, technique: JammingTechnique, power_level: int) -> int:
        return self._techniques.index(technique) * self._n_power + power_level

    def _decode_action(self, action: int) -> tuple[JammingTechnique, int]:
        technique_idx, power_level = divmod(int(action), self._n_power)
        return self._techniques[technique_idx], power_level

    def _emitted_features(self) -> NDArray[np.float32]:
        spec = self._emitter.modes[self._ladder[self._state.mode_idx]]
        rf = 0.5 * (spec.rf_band[0] + spec.rf_band[1])
        pri = 0.5 * (spec.pri_range[0] + spec.pri_range[1])
        pw = 0.5 * (spec.pw_range[0] + spec.pw_range[1])
        rf_n = _normalize(rf, CONTINUOUS_RANGES[0])
        pw_n = _normalize(pw, CONTINUOUS_RANGES[1])
        pri_n = _normalize(pri, CONTINUOUS_RANGES[4])
        scan_n = min(1.0, spec.scan_period / _SCAN_MAX)
        eccm = 1.0 if self._state.eccm_active else 0.0
        feat = np.array([rf_n, pri_n, pw_n, scan_n, eccm], dtype=np.float32)
        if self.config.obs_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.config.obs_noise_std, _N_FEATURES)
            feat = (feat + noise).astype(np.float32)
        return np.clip(feat, 0.0, 1.0).astype(np.float32)

    def _push_obs(self) -> NDArray[np.float32]:
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = self._emitted_features()
        return self._history.copy()

    def _info(self, outcome: str, j_s: float) -> dict[str, Any]:
        return {
            "real_mode": self._ladder[self._state.mode_idx],
            "j_s": j_s,
            "eccm_active": self._state.eccm_active,
            "outcome": outcome,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._candidates)))
        self._emitter = self._candidates[idx]
        self._ladder = tuple(m for m in MODES if m in self._emitter.modes)
        self._state = RadarState()
        self._t = 0
        self._history = np.zeros((self.config.history_k, _N_FEATURES), dtype=np.float32)
        obs = self._push_obs()
        return obs, self._info("ongoing", 0.0)
