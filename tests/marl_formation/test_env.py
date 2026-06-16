import numpy as np

from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv

CONFIG = "configs/marl_formation/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = IADSEnvConfig.from_yaml(CONFIG)
    assert config.n_agents == 4
    assert config.n_radars == 4
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert "noise" in config.suppression_techniques


def _env():
    return IADSFormationEnv(IADSEnvConfig.from_yaml(CONFIG))


def test_reset_returns_obs_state_info():
    env = _env()
    obs, state, info = env.reset(seed=0)
    assert set(obs) == set(range(4))
    assert obs[0].shape == (4 * 5 + 4,)
    assert obs[0].dtype == np.float32
    assert state.shape == (4 * (4 + 2) + 4,)
    assert info["outcome"] == "ongoing"


def test_reset_is_deterministic_by_seed():
    a, sa, _ = _env().reset(seed=3)
    b, sb, _ = _env().reset(seed=3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(sa, sb)


def test_encode_action_roundtrip():
    env = _env()
    action = env.encode_action(target=2, jam_type=1, power_level=3)
    assert env._decode_action(action) == (2, 1, 3)
    assert env.action_dim == 4 * 3 * 4
