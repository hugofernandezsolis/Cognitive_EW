from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig
from cog_ew.temporal_cnn_elint.train import TrainConfig

CONFIG = "configs/temporal_cnn_elint/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)

    assert isinstance(config.data, PDWConfig)
    assert isinstance(config.model, TemporalCNNConfig)
    assert config.data.window == 64
    assert config.model.dilations == (1, 2, 4, 8)
    assert config.splits == (0.7, 0.15, 0.15)
    assert config.loss_weights == (1.0, 1.0)
    assert config.tracking is False
