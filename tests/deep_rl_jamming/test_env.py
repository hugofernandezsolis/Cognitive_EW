import numpy as np

from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.ew_library.library import JammingTechnique

CONFIG = "configs/deep_rl_jamming/env.yaml"


def _env():
    return RadarJammingEnv(RadarEnvConfig.from_yaml(CONFIG))


def test_config_from_yaml_loads_parameters():
    config = RadarEnvConfig.from_yaml(CONFIG)
    assert config.history_k == 8
    assert config.horizon_t == 64
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert config.effectiveness["none"]["missile_guidance"] == 0.0


def test_reset_returns_obs_with_correct_shape():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 5)
    assert obs.dtype == np.float32
    assert info["outcome"] == "ongoing"
    assert info["real_mode"] == "search"


def test_action_and_observation_spaces():
    env = _env()
    assert env.action_space.n == 10 * 4
    assert env.observation_space.shape == (8, 5)


def test_reset_is_deterministic_by_seed():
    a, _ = _env().reset(seed=7)
    b, _ = _env().reset(seed=7)
    assert np.array_equal(a, b)


def test_encode_action_roundtrip():
    env = _env()
    action = env.encode_action(JammingTechnique.NONE, 0)
    assert action == list(JammingTechnique).index(JammingTechnique.NONE) * 4
