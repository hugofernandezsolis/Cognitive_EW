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


def evaluate_policy(
    env: RadarJammingEnv, policy: Policy, episodes: int, seed: int
) -> dict[str, float]:
    wins = 0
    total_reward = 0.0
    total_steps = 0
    obs, info = env.reset(seed=seed)
    for _ in range(episodes):
        done = False
        while not done:
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        obs, info = env.reset()
    return {
        "win_rate": wins / episodes,
        "mean_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
    }


def compare(
    env: RadarJammingEnv,
    cognitive: Policy,
    baseline: Policy,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    cognitive_metrics = evaluate_policy(env, cognitive, episodes, seed)
    baseline_metrics = evaluate_policy(env, baseline, episodes, seed)
    return {
        "cognitive": cognitive_metrics,
        "baseline": baseline_metrics,
        "delta": {
            "win_rate": cognitive_metrics["win_rate"] - baseline_metrics["win_rate"],
            "mean_reward": cognitive_metrics["mean_reward"] - baseline_metrics["mean_reward"],
        },
    }
