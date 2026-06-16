"""Script de entrenamiento del agente Deep RL de jamming adaptativo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.deep_rl_jamming.agent import D3QNConfig
from cog_ew.deep_rl_jamming.env import RadarEnvConfig


@dataclass
class TrainConfig:
    env: RadarEnvConfig
    agent: D3QNConfig
    total_steps: int = 20000
    eval_episodes: int = 50
    eval_every: int = 2000
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/deep_rl_jamming"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        env = RadarEnvConfig.from_yaml(raw.pop("env_config"))
        agent = D3QNConfig(**raw.pop("agent"))
        return cls(env=env, agent=agent, **raw)
