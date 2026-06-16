"""Agente Deep RL para generación de técnicas de jamming adaptativas (<5ms latencia)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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
