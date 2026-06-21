import torch

from cog_ew.gan_signals.discriminator import PDWCritic


def _critic() -> PDWCritic:
    return PDWCritic(e_dim=4, channels=8)


def test_critic_output_shape():
    torch.manual_seed(0)
    out = _critic()(torch.randn(5, 10, 64), torch.randn(5, 4))
    assert out.shape == (5, 1)


def test_critic_admits_negative_scores():
    torch.manual_seed(0)
    critic = _critic()
    scores = critic(torch.randn(64, 10, 64), torch.randn(64, 4))
    assert scores.min().item() < 0.0


def test_critic_has_no_batchnorm():
    critic = _critic()
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in critic.modules())
