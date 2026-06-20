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
