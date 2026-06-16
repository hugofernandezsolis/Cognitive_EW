"""Comparación entre la política cognitiva (D3QN) y el baseline rule-based (Modelo 5)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cog_ew.deep_rl_jamming.agent import D3QNAgent
from cog_ew.deep_rl_jamming.env import RadarJammingEnv
from cog_ew.ew_library.library import EWResponseLibrary


class Policy(Protocol):
    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int: ...


class AgentPolicy:
    def __init__(self, agent: D3QNAgent) -> None:
        self.agent = agent

    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int:
        return self.agent.select_action(obs, epsilon=0.0)


class BaselinePolicy:
    def __init__(self, library: EWResponseLibrary, env: RadarJammingEnv) -> None:
        self.library = library
        self.env = env

    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int:
        techniques = self.library.select(info["emitter"], info["real_mode"])
        return self.env.encode_action(techniques[0], self.env.n_power_levels - 1)
