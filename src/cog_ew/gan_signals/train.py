"""Entrenamiento WGAN-GP del núcleo cWGAN-GP de señales PDW (Modelo 4)."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.discriminator import PDWCritic


@dataclass
class WGANGPConfig:
    pdw: PDWConfig
    z_dim: int = 64
    e_dim: int = 16
    channels: int = 64
    n_critic: int = 5
    lambda_gp: float = 10.0
    lr: float = 1e-4
    gumbel_tau: float = 1.0
    batch_size: int = 64
    total_steps: int = 20000
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/gan_signals/wgan_gp"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> WGANGPConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        pdw = PDWConfig.from_yaml(raw.pop("pdw_config"))
        return cls(pdw=pdw, **raw)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_metadata(config: WGANGPConfig) -> dict[str, Any]:
    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    return {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def gradient_penalty(
    critic: PDWCritic, real: torch.Tensor, fake: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(interpolated, e)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    flat = grads.reshape(batch, -1)
    penalty: torch.Tensor = ((flat.norm(2, dim=1) - 1.0) ** 2).mean()
    return penalty
