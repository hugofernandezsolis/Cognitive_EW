from cog_ew.deep_rl_jamming.env import RadarEnvConfig

CONFIG = "configs/deep_rl_jamming/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = RadarEnvConfig.from_yaml(CONFIG)
    assert config.history_k == 8
    assert config.horizon_t == 64
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert config.effectiveness["none"]["missile_guidance"] == 0.0
