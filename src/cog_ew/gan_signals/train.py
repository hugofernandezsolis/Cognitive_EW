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
from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


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


class WGANGP:
    def __init__(self, n_emitters: int, config: WGANGPConfig, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.embedding = TypeEmbedding(n_emitters, config.e_dim).to(self.device)
        self.generator = PDWGenerator(
            config.z_dim, config.e_dim, config.channels, gumbel_tau=config.gumbel_tau
        ).to(self.device)
        self.critic = PDWCritic(config.e_dim, config.channels).to(self.device)
        gen_params = list(self.generator.parameters()) + list(self.embedding.parameters())
        self.opt_g = torch.optim.Adam(gen_params, lr=config.lr, betas=(0.0, 0.9))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=config.lr, betas=(0.0, 0.9))

    def _z(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.config.z_dim, device=self.device)

    def critic_update(self, real_x: torch.Tensor, ids: torch.Tensor) -> float:
        real_x = real_x.to(self.device)
        e = self.embedding(ids.to(self.device)).detach()
        with torch.no_grad():
            fake = self.generator(self._z(real_x.size(0)), e)
        real_score = self.critic(real_x, e).mean()
        fake_score = self.critic(fake, e).mean()
        gp = gradient_penalty(self.critic, real_x, fake, e)
        loss = fake_score - real_score + self.config.lambda_gp * gp
        self.opt_c.zero_grad()
        loss.backward()
        self.opt_c.step()
        return float(loss.item())

    def generator_update(self, ids: torch.Tensor) -> float:
        e = self.embedding(ids.to(self.device))
        fake = self.generator(self._z(ids.size(0)), e)
        loss = -self.critic(fake, e).mean()
        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()
        return float(loss.item())
