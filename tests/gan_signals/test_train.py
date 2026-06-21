import copy
import json

import numpy as np
import torch

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.discriminator import PDWCritic
from cog_ew.gan_signals.train import (
    WGANGP,
    WGANGPConfig,
    _run_metadata,
    _set_seeds,
    gradient_penalty,
    train,
)


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


def _learner() -> WGANGP:
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    config.z_dim, config.e_dim, config.channels = 8, 4, 8
    return WGANGP(n_emitters=8, config=config, device="cpu")


def test_critic_update_changes_only_critic_params():
    torch.manual_seed(0)
    learner = _learner()
    real = torch.rand(6, 10, 64)
    ids = torch.randint(0, 8, (6,))
    before_c = copy.deepcopy(learner.critic.state_dict())
    before_g = {k: v.clone() for k, v in learner.generator.named_parameters()}
    before_e = {k: v.clone() for k, v in learner.embedding.named_parameters()}
    loss = learner.critic_update(real, ids)
    assert np.isfinite(loss)
    assert any(not torch.allclose(before_c[k], v) for k, v in learner.critic.state_dict().items())
    assert all(torch.allclose(before_g[k], v) for k, v in learner.generator.named_parameters())
    assert all(torch.allclose(before_e[k], v) for k, v in learner.embedding.named_parameters())


def test_generator_update_changes_generator_and_embedding():
    torch.manual_seed(0)
    learner = _learner()
    ids = torch.randint(0, 8, (6,))
    before_g = copy.deepcopy(learner.generator.state_dict())
    before_e = copy.deepcopy(learner.embedding.state_dict())
    loss = learner.generator_update(ids)
    assert np.isfinite(loss)
    assert any(
        not torch.allclose(before_g[k], v) for k, v in learner.generator.state_dict().items()
    )
    assert any(
        not torch.allclose(before_e[k], v) for k, v in learner.embedding.state_dict().items()
    )


def _tiny_config(tmp_path) -> WGANGPConfig:
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    config.pdw.n_trains = 2
    config.z_dim, config.e_dim, config.channels = 8, 4, 8
    config.batch_size, config.total_steps, config.n_critic = 8, 2, 2
    config.out_dir = str(tmp_path / "run")
    return config


def test_train_writes_artifacts_and_checkpoint(tmp_path):
    config = _tiny_config(tmp_path)
    result = train(config)
    out = tmp_path / "run"
    assert (out / "run_meta.json").is_file()
    metrics = json.loads((out / "metrics.json").read_text())
    assert {
        "wasserstein_estimate",
        "gradient_penalty",
        "diversity_std",
        "latency_mean_ms",
        "latency_p99_ms",
    } <= set(metrics)
    assert "final" in result
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"generator", "embedding", "critic"}
