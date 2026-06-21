"""Comparación de regímenes coordinado (QMIX) vs independiente (IQL) en el entorno IADS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from cog_ew.marl_formation.agents import AgentRNN
from cog_ew.marl_formation.env import IADSFormationEnv


class FormationPolicy(Protocol):
    def reset(self, env: IADSFormationEnv) -> None: ...

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]: ...


def _suppress_max(env: IADSFormationEnv, target: int) -> int:
    return env.encode_action(target, jam_type=2, power_level=len(env.config.power_levels) - 1)


class ConcentratedSuppressionPolicy:
    """Baseline sin coordinar: todos los agentes apuntan al mismo radar a máxima potencia."""

    def __init__(self, target: int = 0) -> None:
        self.target = target

    def reset(self, env: IADSFormationEnv) -> None:
        return None

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        action = _suppress_max(env, self.target % env.n_radars)
        return {agent: action for agent in range(env.n_agents)}


class SpreadSuppressionPolicy:
    """Baseline coordinado rule-based: reparte agentes entre radares round-robin."""

    def reset(self, env: IADSFormationEnv) -> None:
        return None

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        return {agent: _suppress_max(env, agent % env.n_radars) for agent in range(env.n_agents)}


class AgentPolicy:
    """Ejecuta una política aprendida (QMIX o IQL) vía AgentRNN greedy descentralizado."""

    def __init__(self, agent: AgentRNN, n_agents: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.agent = agent.to(self.device)
        self.n_agents = n_agents
        self._hidden: dict[int, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        action_dim: int,
        hidden: int,
        n_agents: int,
        device: str = "cpu",
    ) -> AgentPolicy:
        agent = AgentRNN(obs_dim, action_dim, hidden)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
        return cls(agent=agent, n_agents=n_agents, device=device)

    def reset(self, env: IADSFormationEnv) -> None:
        self._hidden = {
            agent: self.agent.init_hidden(1).to(self.device) for agent in range(self.n_agents)
        }

    @torch.no_grad()
    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        if not self._hidden:
            self.reset(env)
        actions: dict[int, int] = {}
        new_hidden: dict[int, torch.Tensor] = {}
        for agent in range(self.n_agents):
            obs_t = torch.as_tensor(obs[agent], dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            q_values, hidden = self.agent(obs_t, self._hidden[agent])
            actions[agent] = int(torch.argmax(q_values, dim=1).item())
            new_hidden[agent] = hidden
        self._hidden = new_hidden
        return actions


def evaluate_policy(
    env: IADSFormationEnv,
    policy: FormationPolicy,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    wins = 0
    total_reward = 0.0
    total_steps = 0
    suppressed_sum = 0.0
    for episode in range(episodes):
        obs, state, info = env.reset(seed=seed if episode == 0 else None)
        policy.reset(env)
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            actions = policy.act(env, obs, state, info)
            obs, state, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += rewards[0]
            steps += 1
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        total_reward += episode_reward
        total_steps += steps
        suppressed_sum += float(info["suppressed_fraction"])
    return {
        "win_rate": wins / episodes,
        "mean_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
        "suppressed_fraction": suppressed_sum / episodes,
    }


def compare_policies(
    env: IADSFormationEnv,
    *,
    coordinated: FormationPolicy,
    independent: FormationPolicy,
    episodes: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    coord = evaluate_policy(env, coordinated, episodes=episodes, seed=seed)
    indep = evaluate_policy(env, independent, episodes=episodes, seed=seed)
    indep_supp = indep["suppressed_fraction"]
    rel_supp = (
        (coord["suppressed_fraction"] - indep_supp) / indep_supp if indep_supp > 0 else float("inf")
    )
    return {
        "coordinated": coord,
        "independent": indep,
        "delta": {
            "win_rate": coord["win_rate"] - indep["win_rate"],
            "mean_reward": coord["mean_reward"] - indep["mean_reward"],
            "suppressed_fraction": coord["suppressed_fraction"] - indep_supp,
        },
        "relative_improvement": {"suppressed_fraction": rel_supp},
    }
