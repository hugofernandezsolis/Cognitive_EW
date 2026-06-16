from cog_ew.deep_rl_jamming.agent import D3QNConfig
from cog_ew.deep_rl_jamming.env import RadarEnvConfig
from cog_ew.deep_rl_jamming.train import TrainConfig

CONFIG = "configs/deep_rl_jamming/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)
    assert isinstance(config.env, RadarEnvConfig)
    assert isinstance(config.agent, D3QNConfig)
    assert config.env.history_k == 8
    assert config.agent.hidden == 128
    assert config.total_steps > 0
