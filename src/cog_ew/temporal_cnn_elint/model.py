"""Temporal CNN para clasificación en tiempo real de señales de amenaza ELINT (<1ms latencia)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import nn

from cog_ew.data.pdw_library import MODES, mode_to_threat


@dataclass
class TemporalCNNConfig:
    in_channels: int = 10
    seq_len: int = 64
    hidden: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    n_types: int = 8
    n_modes: int = 4
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> TemporalCNNConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if "dilations" in raw:
            raw["dilations"] = tuple(raw["dilations"])
        return cls(**raw)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        out: torch.Tensor = self.act(x + y)
        return out


class TemporalCNN(nn.Module):
    threat_from_mode: torch.Tensor

    def __init__(self, config: TemporalCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.hidden, 3, padding=1),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _TCNBlock(config.hidden, d, config.dropout) for d in config.dilations
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_type = nn.Linear(config.hidden, config.n_types)
        self.head_mode = nn.Linear(config.hidden, config.n_modes)
        threat = torch.tensor([mode_to_threat(m) for m in MODES], dtype=torch.int64)
        self.register_buffer("threat_from_mode", threat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        feat = self.pool(h).squeeze(-1)
        return self.head_type(feat), self.head_mode(feat)
