"""Crítico Wasserstein (WGAN-GP) condicionado para señales PDW (Modelo 4)."""

from __future__ import annotations

import torch
from torch import nn


class PDWCritic(nn.Module):
    def __init__(self, e_dim: int, channels: int, *, in_channels: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(channels)
        self.out = nn.Linear(channels + e_dim, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.net(x)).squeeze(-1)
        h = self.norm(h)
        out: torch.Tensor = self.out(torch.cat([h, e], dim=1))
        return out
