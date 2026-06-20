"""Agentes QMIX y red de mezcla para coordinación EW en formación (CTDE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F


class AgentRNN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden)
        q: torch.Tensor = self.fc2(h)
        return q, h

    def init_hidden(self, batch: int) -> torch.Tensor:
        return self.fc1.weight.new_zeros(batch, self.hidden)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int, hypernet_hidden: int) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, n_agents * embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(batch, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot: torch.Tensor = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch, 1)


@dataclass(frozen=True)
class QMIXConfig:
    hidden: int = 64
    mixer_embed_dim: int = 32
    hypernet_hidden: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    batch_episodes: int = 8
    buffer_episodes: int = 2000
    target_sync: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    learning_starts_episodes: int = 32
    double_q: bool = True
    grad_clip: float = 10.0


class QMIXLearner:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        state_dim: int,
        config: QMIXConfig,
        device: str,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)
        self.rng = rng
        self.agent = AgentRNN(obs_dim, action_dim, config.hidden).to(self.device)
        self.target_agent = AgentRNN(obs_dim, action_dim, config.hidden).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, config.mixer_embed_dim, config.hypernet_hidden).to(
            self.device
        )
        self.target_mixer = QMixer(
            n_agents, state_dim, config.mixer_embed_dim, config.hypernet_hidden
        ).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        self._updates = 0

    def init_hidden(self) -> dict[int, torch.Tensor]:
        return {a: self.agent.init_hidden(1) for a in range(self.n_agents)}

    @torch.no_grad()
    def select_actions(
        self,
        obs_dict: dict[int, NDArray[np.float32]],
        hidden: dict[int, torch.Tensor],
        epsilon: float,
    ) -> tuple[dict[int, int], dict[int, torch.Tensor]]:
        actions: dict[int, int] = {}
        new_hidden: dict[int, torch.Tensor] = {}
        for a in range(self.n_agents):
            obs_t = torch.as_tensor(obs_dict[a], dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            q, h = self.agent(obs_t, hidden[a])
            new_hidden[a] = h
            if self.rng.random() < epsilon:
                actions[a] = int(self.rng.integers(self.action_dim))
            else:
                actions[a] = int(torch.argmax(q, dim=1).item())
        return actions, new_hidden
