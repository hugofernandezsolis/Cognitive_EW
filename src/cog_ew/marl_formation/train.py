"""Entrenamiento CTDE (QMIX) de la formación EW sobre el entorno IADS."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from cog_ew.marl_formation.agents import AgentRNN, IQLLearner, QMIXConfig, QMIXLearner
from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv
from cog_ew.temporal_cnn_elint.metrics import profile_latency


class EpisodeReplayBuffer:
    def __init__(
        self, capacity: int, horizon: int, n_agents: int, obs_dim: int, state_dim: int
    ) -> None:
        self.capacity = capacity
        self._obs = np.zeros((capacity, horizon, n_agents, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, horizon, n_agents), dtype=np.int64)
        self._rewards = np.zeros((capacity, horizon), dtype=np.float32)
        self._states = np.zeros((capacity, horizon, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, horizon), dtype=np.float32)
        self._filled = np.zeros((capacity, horizon), dtype=np.float32)
        self._size = 0
        self._pos = 0

    def add(
        self,
        obs: NDArray[np.float32],
        actions: NDArray[np.int64],
        rewards: NDArray[np.float32],
        states: NDArray[np.float32],
        dones: NDArray[np.float32],
        filled: NDArray[np.float32],
    ) -> None:
        i = self._pos
        self._obs[i] = obs
        self._actions[i] = actions
        self._rewards[i] = rewards
        self._states[i] = states
        self._dones[i] = dones
        self._filled[i] = filled
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_episodes: int, rng: np.random.Generator
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        idx = rng.integers(0, self._size, size=batch_episodes)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._states[idx],
            self._dones[idx],
            self._filled[idx],
        )

    def __len__(self) -> int:
        return self._size


@dataclass(frozen=True)
class _EpisodeData:
    obs: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    states: NDArray[np.float32]
    dones: NDArray[np.float32]
    filled: NDArray[np.float32]


@dataclass
class TrainConfig:
    env: IADSEnvConfig
    agent: QMIXConfig
    total_episodes: int = 4000
    eval_episodes: int = 50
    eval_every: int = 200
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/marl_formation"
    tracking: bool = False
    regime: str = "qmix"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        env = IADSEnvConfig.from_yaml(raw.pop("env_config"))
        agent = QMIXConfig(**raw.pop("agent"))
        return cls(env=env, agent=agent, **raw)


def _build_learner(
    regime: str,
    env: IADSFormationEnv,
    config: QMIXConfig,
    device: str,
    rng: np.random.Generator,
) -> QMIXLearner | IQLLearner:
    if regime == "qmix":
        return QMIXLearner(
            env.obs_dim, env.action_dim, env.n_agents, env.state_dim, config, device, rng
        )
    if regime == "iql":
        return IQLLearner(env.obs_dim, env.action_dim, env.n_agents, config, device, rng)
    raise ValueError(f"unknown regime: {regime!r}")


class _AgentForward(torch.nn.Module):
    def __init__(self, agent: AgentRNN) -> None:
        super().__init__()
        self.agent = agent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.agent.init_hidden(obs.size(0))
        q_values: torch.Tensor
        q_values, _ = self.agent.forward(obs, hidden)
        return q_values


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epsilon(episode: int, config: QMIXConfig) -> float:
    frac = min(1.0, episode / config.epsilon_decay_steps)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _run_metadata(config: TrainConfig) -> dict[str, Any]:
    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    return {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }


def _rollout(
    env: IADSFormationEnv, learner: QMIXLearner | IQLLearner, epsilon: float, seed: int | None
) -> tuple[_EpisodeData, dict[str, Any]]:
    horizon = env.config.horizon_t
    obs_buf = np.zeros((horizon, env.n_agents, env.obs_dim), dtype=np.float32)
    act_buf = np.zeros((horizon, env.n_agents), dtype=np.int64)
    rew_buf = np.zeros(horizon, dtype=np.float32)
    state_buf = np.zeros((horizon, env.state_dim), dtype=np.float32)
    done_buf = np.zeros(horizon, dtype=np.float32)
    filled_buf = np.zeros(horizon, dtype=np.float32)

    obs, state, info = env.reset(seed=seed)
    hidden = learner.init_hidden()
    done = False
    t = 0
    while not done and t < horizon:
        actions, hidden = learner.select_actions(obs, hidden, epsilon)
        obs_buf[t] = np.stack([obs[a] for a in range(env.n_agents)])
        act_buf[t] = np.asarray([actions[a] for a in range(env.n_agents)], dtype=np.int64)
        state_buf[t] = state
        obs, state, rewards, terminated, truncated, info = env.step(actions)
        rew_buf[t] = rewards[0]
        done = terminated or truncated
        done_buf[t] = 1.0 if done else 0.0
        filled_buf[t] = 1.0
        t += 1

    episode = _EpisodeData(
        obs=obs_buf,
        actions=act_buf,
        rewards=rew_buf,
        states=state_buf,
        dones=done_buf,
        filled=filled_buf,
    )
    return episode, info


def _evaluate(
    env: IADSFormationEnv, learner: QMIXLearner | IQLLearner, n_episodes: int, seed: int
) -> tuple[float, float]:
    wins = 0
    suppressed_sum = 0.0
    for episode in range(n_episodes):
        _, info = _rollout(env, learner, epsilon=0.0, seed=seed if episode == 0 else None)
        if info["outcome"] == "win":
            wins += 1
        suppressed_sum += float(info["suppressed_fraction"])
    return wins / n_episodes, suppressed_sum / n_episodes


def _is_better_checkpoint(
    win_rate: float, suppressed_fraction: float, best_score: tuple[float, float]
) -> bool:
    return (win_rate, suppressed_fraction) > best_score


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    env = IADSFormationEnv(config.env)
    eval_env = IADSFormationEnv(config.env)
    learner = _build_learner(config.regime, env, config.agent, config.device, rng)
    buffer = EpisodeReplayBuffer(
        config.agent.buffer_episodes,
        config.env.horizon_t,
        env.n_agents,
        env.obs_dim,
        env.state_dim,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))
    best_path = out_dir / "best.pt"

    win_rate_history: list[float] = []
    suppressed_history: list[float] = []
    best_score = (-1.0, -1.0)

    for episode_idx in range(config.total_episodes):
        episode, _ = _rollout(
            env,
            learner,
            epsilon=_epsilon(episode_idx, config.agent),
            seed=config.seed if episode_idx == 0 else None,
        )
        buffer.add(
            episode.obs,
            episode.actions,
            episode.rewards,
            episode.states,
            episode.dones,
            episode.filled,
        )
        if len(buffer) >= config.agent.learning_starts_episodes:
            learner.update(buffer.sample(config.agent.batch_episodes, rng))
        if (episode_idx + 1) % config.eval_every == 0:
            win_rate, suppressed = _evaluate(eval_env, learner, config.eval_episodes, config.seed)
            win_rate_history.append(win_rate)
            suppressed_history.append(suppressed)
            if _is_better_checkpoint(win_rate, suppressed, best_score):
                best_score = (win_rate, suppressed)
                torch.save(learner.agent.state_dict(), best_path)

    if not win_rate_history:
        win_rate, suppressed = _evaluate(eval_env, learner, config.eval_episodes, config.seed)
        win_rate_history.append(win_rate)
        suppressed_history.append(suppressed)
        best_score = (win_rate, suppressed)
        torch.save(learner.agent.state_dict(), best_path)

    sample = torch.zeros(1, env.obs_dim, dtype=torch.float32)
    mean_ms, p99_ms = profile_latency(
        _AgentForward(learner.agent),
        sample,
        n_warmup=5,
        n_iter=50,
        device=config.device,
    )
    final = {
        "win_rate": win_rate_history[-1],
        "suppressed_fraction": suppressed_history[-1],
        "best_win_rate": best_score[0],
        "best_suppressed_fraction": best_score[1],
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"win_rate_history": win_rate_history, "final": final}
