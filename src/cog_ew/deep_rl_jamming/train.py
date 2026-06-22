"""Script de entrenamiento del agente Deep RL de jamming adaptativo."""

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
from gymnasium.spaces import Discrete

from cog_ew.deep_rl_jamming.agent import D3QNAgent, D3QNConfig, ReplayBuffer
from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.temporal_cnn_elint.metrics import profile_latency


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


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _epsilon(step: int, config: D3QNConfig) -> float:
    frac = min(1.0, step / config.epsilon_decay_steps)
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


def _evaluate(env: RadarJammingEnv, agent: D3QNAgent, n_episodes: int, seed: int) -> float:
    wins = 0
    obs, info = env.reset(seed=seed)
    for _ in range(n_episodes):
        done = False
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        obs, info = env.reset()
    return wins / n_episodes


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    env = RadarJammingEnv(config.env)
    eval_env = RadarJammingEnv(config.env)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    obs_dim = int(np.prod(obs_shape))
    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    agent = D3QNAgent(obs_dim, n_actions, config.agent, config.device, rng)
    buffer = ReplayBuffer(config.agent.buffer_size, obs_shape)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))
    best_path = out_dir / "best.pt"

    win_rate_history: list[float] = []
    best_win_rate = -1.0

    obs, _ = env.reset(seed=config.seed)
    for step in range(config.total_steps):
        action = agent.select_action(obs, _epsilon(step, config.agent))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if len(buffer) >= config.agent.learning_starts and step % config.agent.train_freq == 0:
            agent.update(buffer.sample(config.agent.batch_size, rng))

        if (step + 1) % config.eval_every == 0:
            win_rate = _evaluate(eval_env, agent, config.eval_episodes, config.seed)
            win_rate_history.append(win_rate)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(agent.online_net.state_dict(), best_path)

    if not win_rate_history:
        win_rate = _evaluate(eval_env, agent, config.eval_episodes, config.seed)
        win_rate_history.append(win_rate)
        torch.save(agent.online_net.state_dict(), best_path)

    sample = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean_ms, p99_ms = profile_latency(
        agent.online_net, sample, n_warmup=5, n_iter=50, device=config.device
    )
    final = {
        "win_rate": win_rate_history[-1],
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"win_rate_history": win_rate_history, "final": final}
