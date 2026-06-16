"""Agente Deep RL para generación de técnicas de jamming adaptativas (<5ms latencia)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs.flatten(start_dim=1))
        value = self.value(h)
        advantage = self.advantage(h)
        q: torch.Tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


@dataclass(frozen=True)
class D3QNConfig:
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    target_sync: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    learning_starts: int = 1000
    train_freq: int = 1


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._pos = 0

    def add(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        done: bool,
    ) -> None:
        i = self._pos
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size


class D3QNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: D3QNConfig,
        device: str,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.rng = rng
        self.online_net = QNetwork(obs_dim, n_actions, config.hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, config.hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self._updates = 0

    @torch.no_grad()
    def select_action(self, obs: NDArray[np.float32], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(torch.argmax(self.online_net(tensor), dim=1).item())

    def update(
        self,
        batch: tuple[
            NDArray[np.float32],
            NDArray[np.int64],
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.float32],
        ],
    ) -> float:
        obs, actions, rewards, next_obs, dones = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q = self.online_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = torch.argmax(self.online_net(next_t), dim=1)
            next_q = self.target_net(next_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())
