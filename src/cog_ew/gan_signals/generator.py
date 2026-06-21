"""Generador cWGAN-GP y embedding de tipo para señales PDW sintéticas (Modelo 4)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TypeEmbedding(nn.Module):
    def __init__(self, n_emitters: int, e_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_emitters, e_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.embedding(ids)
        return out

    def interpolate(self, id_a: int, id_b: int, alpha: float) -> torch.Tensor:
        weight = self.embedding.weight
        return (1.0 - alpha) * weight[id_a] + alpha * weight[id_b]


class PDWGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        e_dim: int,
        channels: int,
        *,
        n_continuous: int = 5,
        n_categorical: int = 5,
        seq_len: int = 64,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.gumbel_tau = gumbel_tau
        self.channels = channels
        self.init_len = seq_len // 8
        self.project = nn.Linear(z_dim + e_dim, channels * self.init_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(channels // 2, channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(channels // 2, n_continuous + n_categorical, 3, padding=1)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, e], dim=1)
        x = self.project(x).view(-1, self.channels, self.init_len)
        raw = self.head(self.net(x))
        cont = torch.sigmoid(raw[:, : self.n_continuous])
        cat = F.gumbel_softmax(raw[:, self.n_continuous :], tau=self.gumbel_tau, hard=True, dim=1)
        return torch.cat([cont, cat], dim=1)

    def sample(self, e: torch.Tensor) -> torch.Tensor:
        z = torch.randn(e.size(0), self.z_dim, device=e.device)
        return self.forward(z, e)
