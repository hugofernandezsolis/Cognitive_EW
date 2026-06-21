"""Generador cWGAN-GP y embedding de tipo para señales PDW sintéticas (Modelo 4)."""

from __future__ import annotations

import torch
from torch import nn


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
