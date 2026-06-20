"""Agentes QMIX y red de mezcla para coordinación EW en formación (CTDE)."""

from __future__ import annotations

import torch
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
