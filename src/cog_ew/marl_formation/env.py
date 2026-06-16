"""Entorno multi-agente que simula el IADS adversario y la formación de aeronaves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from cog_ew.data.pdw_library import CONTINUOUS_RANGES, MODES, EmitterLibrary, EmitterSpec
from cog_ew.deep_rl_jamming.threat import RadarState
from cog_ew.ew_library.library import JammingTechnique

_SCAN_MAX = 15.0
_N_RADAR_FEATURES = 5
_N_MODES = len(MODES)


def _normalize(value: float, value_range: NDArray[np.float64]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


@dataclass(frozen=True)
class IADSEnvConfig:
    library_path: str
    effectiveness: dict[str, dict[str, float]]
    suppression_techniques: tuple[str, ...] = (
        "noise",
        "drfm_repeater",
        "vgpo",
        "rgpo",
        "cross_eye",
        "chaff",
    )
    emitters: tuple[str, ...] | None = None
    n_agents: int = 4
    n_radars: int = 4
    power_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    burnthrough: float = 15.0
    eff_threshold: float = 0.5
    js_scale: float = 20.0
    lock_gain: float = 0.15
    lock_decay: float = 0.15
    n_eccm: int = 3
    w_lock: float = 1.0
    lambda_power: float = 0.5
    w_supp: float = 1.0
    r_win: float = 10.0
    r_lose: float = 10.0
    horizon_t: int = 64
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> IADSEnvConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("emitters") is not None:
            raw["emitters"] = tuple(raw["emitters"])
        if "power_levels" in raw:
            raw["power_levels"] = tuple(float(p) for p in raw["power_levels"])
        if "suppression_techniques" in raw:
            raw["suppression_techniques"] = tuple(raw["suppression_techniques"])
        return cls(**raw)


class IADSFormationEnv:
    def __init__(self, config: IADSEnvConfig) -> None:
        self.config = config
        library = EmitterLibrary.from_yaml(config.library_path)
        if config.emitters is not None:
            self._candidates = tuple(e for e in library.emitters if e.name in config.emitters)
        else:
            self._candidates = library.emitters
        self._techniques = list(JammingTechnique)
        self._none_idx = self._techniques.index(JammingTechnique.NONE)
        self._n_power = len(config.power_levels)
        self.n_agents = config.n_agents
        self.n_radars = config.n_radars
        self.action_dim = config.n_radars * 3 * self._n_power
        self.obs_dim = config.n_radars * _N_RADAR_FEATURES + config.n_agents
        self.state_dim = config.n_radars * (_N_MODES + 2) + config.n_agents
        self._rng = np.random.default_rng(config.seed)
        self._emitters: list[EmitterSpec] = []
        self._ladders: list[tuple[str, ...]] = []
        self._states: list[RadarState] = []
        self._last_actions = [0] * config.n_agents
        self._t = 0

    def encode_action(self, target: int, jam_type: int, power_level: int) -> int:
        return target * (3 * self._n_power) + jam_type * self._n_power + power_level

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        power_level = action % self._n_power
        jam_type = (action // self._n_power) % 3
        target = action // (3 * self._n_power)
        return target, jam_type, power_level

    def _radar_features(self, idx: int) -> NDArray[np.float32]:
        spec = self._emitters[idx].modes[self._ladders[idx][self._states[idx].mode_idx]]
        rf = 0.5 * (spec.rf_band[0] + spec.rf_band[1])
        pri = 0.5 * (spec.pri_range[0] + spec.pri_range[1])
        pw = 0.5 * (spec.pw_range[0] + spec.pw_range[1])
        eccm = 1.0 if self._states[idx].eccm_active else 0.0
        return np.array(
            [
                _normalize(rf, CONTINUOUS_RANGES[0]),
                _normalize(pri, CONTINUOUS_RANGES[4]),
                _normalize(pw, CONTINUOUS_RANGES[1]),
                min(1.0, spec.scan_period / _SCAN_MAX),
                eccm,
            ],
            dtype=np.float32,
        )

    def _obs(self) -> dict[int, NDArray[np.float32]]:
        radar_feats = np.concatenate([self._radar_features(i) for i in range(self.n_radars)])
        obs: dict[int, NDArray[np.float32]] = {}
        for a in range(self.n_agents):
            agent_onehot = np.zeros(self.n_agents, dtype=np.float32)
            agent_onehot[a] = 1.0
            obs[a] = np.concatenate([radar_feats, agent_onehot]).astype(np.float32)
        return obs

    def _global_state(self) -> NDArray[np.float32]:
        parts: list[NDArray[np.float32]] = []
        for i in range(self.n_radars):
            state = self._states[i]
            mode_oh = np.zeros(_N_MODES, dtype=np.float32)
            mode_oh[MODES.index(self._ladders[i][state.mode_idx])] = 1.0
            parts.append(mode_oh)
            parts.append(
                np.array([state.lock_energy, 1.0 if state.eccm_active else 0.0], dtype=np.float32)
            )
        last = np.array([a / (self.action_dim - 1) for a in self._last_actions], dtype=np.float32)
        parts.append(last)
        return np.concatenate(parts).astype(np.float32)

    def _info(self, outcome: str, suppressed_count: int) -> dict[str, Any]:
        return {"outcome": outcome, "suppressed_fraction": suppressed_count / self.n_radars}

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[int, NDArray[np.float32]], NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idxs = self._rng.integers(0, len(self._candidates), size=self.n_radars)
        self._emitters = [self._candidates[int(i)] for i in idxs]
        self._ladders = [tuple(m for m in MODES if m in e.modes) for e in self._emitters]
        self._states = [RadarState() for _ in range(self.n_radars)]
        self._last_actions = [0] * self.n_agents
        self._t = 0
        return self._obs(), self._global_state(), self._info("ongoing", 0)
