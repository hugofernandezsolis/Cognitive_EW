import torch

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.discriminator import PDWCritic
from cog_ew.gan_signals.train import WGANGPConfig, _run_metadata, _set_seeds, gradient_penalty


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


def test_wgangp_config_from_yaml_loads_pdw():
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    assert isinstance(config.pdw, PDWConfig)
    assert config.z_dim > 0
    assert config.lambda_gp == 10.0


def test_set_seeds_is_deterministic():
    _set_seeds(0)
    a = torch.rand(4)
    _set_seeds(0)
    b = torch.rand(4)
    assert torch.allclose(a, b)


def test_run_metadata_has_required_keys():
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    meta = _run_metadata(config)
    assert set(meta) == {"seed", "hyperparameters", "config_hash", "dependencies"}
    assert meta["seed"] == config.seed
    assert {"python", "torch", "numpy"} <= set(meta["dependencies"])
