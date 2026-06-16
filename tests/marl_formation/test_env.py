from cog_ew.marl_formation.env import IADSEnvConfig

CONFIG = "configs/marl_formation/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = IADSEnvConfig.from_yaml(CONFIG)
    assert config.n_agents == 4
    assert config.n_radars == 4
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert "noise" in config.suppression_techniques
