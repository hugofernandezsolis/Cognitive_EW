"""Entrenamiento WGAN-GP del núcleo cWGAN-GP de señales PDW (Modelo 4)."""

from __future__ import annotations

import torch

from cog_ew.gan_signals.discriminator import PDWCritic


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
