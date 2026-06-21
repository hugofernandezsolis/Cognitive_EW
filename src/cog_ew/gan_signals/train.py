"""Entrenamiento WGAN-GP del núcleo cWGAN-GP de señales PDW (Modelo 4)."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.gan_signals.discriminator import PDWCritic
from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding
from cog_ew.temporal_cnn_elint.metrics import profile_latency


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


class _GeneratorForward(nn.Module):
    """Wraps PDWGenerator with a fixed embedding for latency profiling."""

    def __init__(self, generator: PDWGenerator, e: torch.Tensor) -> None:
        super().__init__()
        self.generator = generator
        self.e = e

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z, self.e)  # type: ignore[no-any-return]


def _cycle(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def train(config: WGANGPConfig) -> dict[str, Any]:
    """Run the WGAN-GP training loop and write artifacts to config.out_dir."""
    _set_seeds(config.seed)
    library = EmitterLibrary.from_yaml(config.pdw.library_path)
    n_emitters = len(library.emitter_names())
    dataset = PDWSyntheticDataset(config.pdw)
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    learner = WGANGP(n_emitters, config, config.device)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))

    data_iter = _cycle(loader)
    critic_loss = 0.0
    gen_loss = 0.0
    for _ in range(config.total_steps):
        for _ in range(config.n_critic):
            real_x, ids, _, _ = next(data_iter)
            critic_loss = learner.critic_update(real_x, ids)
        _, ids, _, _ = next(data_iter)
        gen_loss = learner.generator_update(ids)

    real_x, ids, _, _ = next(data_iter)
    real_x = real_x.to(learner.device)
    e = learner.embedding(ids.to(learner.device)).detach()
    with torch.no_grad():
        fake = learner.generator(learner._z(real_x.size(0)), e)
        wasserstein = float(
            (learner.critic(real_x, e).mean() - learner.critic(fake, e).mean()).item()
        )
        diversity = float(fake.std(dim=0).mean().item())
    gp = float(gradient_penalty(learner.critic, real_x, fake, e).item())

    torch.save(
        {
            "generator": learner.generator.state_dict(),
            "embedding": learner.embedding.state_dict(),
            "critic": learner.critic.state_dict(),
        },
        out_dir / "best.pt",
    )

    e0 = learner.embedding(torch.zeros(1, dtype=torch.long, device=learner.device)).detach()
    sample = torch.zeros(1, config.z_dim, device=learner.device)
    mean_ms, p99_ms = profile_latency(
        _GeneratorForward(learner.generator, e0),
        sample,
        n_warmup=5,
        n_iter=50,
        device=config.device,
    )

    final: dict[str, Any] = {
        "wasserstein_estimate": wasserstein,
        "gradient_penalty": gp,
        "diversity_std": diversity,
        "critic_loss": critic_loss,
        "gen_loss": gen_loss,
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"final": final}
