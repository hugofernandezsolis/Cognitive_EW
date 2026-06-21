import torch

from cog_ew.gan_signals.discriminator import PDWCritic
from cog_ew.gan_signals.train import gradient_penalty


def test_gradient_penalty_is_finite_nonnegative_scalar():
    torch.manual_seed(0)
    critic = PDWCritic(e_dim=4, channels=8)
    real = torch.randn(6, 10, 64)
    fake = torch.randn(6, 10, 64)
    e = torch.randn(6, 4)
    gp = gradient_penalty(critic, real, fake, e)
    assert gp.shape == ()
    assert torch.isfinite(gp)
    assert gp.item() >= 0.0


def test_gradient_penalty_is_differentiable_wrt_critic():
    torch.manual_seed(0)
    critic = PDWCritic(e_dim=4, channels=8)
    real = torch.randn(6, 10, 64)
    fake = torch.randn(6, 10, 64)
    e = torch.randn(6, 4)
    gp = gradient_penalty(critic, real, fake, e)
    gp.backward()
    assert any(p.grad is not None for p in critic.parameters())
