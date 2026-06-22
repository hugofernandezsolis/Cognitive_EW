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


@dataclass
class TemporalCNNV2Config:
    in_channels: int = 18
    seq_len: int = 64
    hidden: int = 64
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    n_types: int = 8
    n_modes: int = 4
    n_threats: int = 4
    dropout: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> TemporalCNNV2Config:
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


class _DepthwiseTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.act(x + self.net(x))
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

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_logits, mode_logits = self.forward(x)
        type_pred = type_logits.argmax(dim=-1)
        mode_pred = mode_logits.argmax(dim=-1)
        threat_pred: torch.Tensor = self.threat_from_mode[mode_pred]
        return type_pred, mode_pred, threat_pred


class TemporalCNNV2(nn.Module):
    def __init__(self, config: TemporalCNNV2Config) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.hidden, kernel_size=1),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _DepthwiseTCNBlock(config.hidden, config.kernel_size, d, config.dropout)
            for d in config.dilations
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head_type = nn.Linear(config.hidden, config.n_types)
        self.head_mode = nn.Linear(config.hidden, config.n_modes)
        self.head_threat = nn.Linear(config.hidden, config.n_threats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        feat = self.shared(self.pool(h).squeeze(-1))
        return self.head_type(feat), self.head_mode(feat), self.head_threat(feat)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_logits, mode_logits, threat_logits = self.forward(x)
        type_pred = type_logits.argmax(dim=-1)
        mode_pred = mode_logits.argmax(dim=-1)
        threat_pred = threat_logits.argmax(dim=-1)
        return type_pred, mode_pred, threat_pred
