from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.robustness import RobustnessConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig


def test_robustness_config_from_yaml():
    config = RobustnessConfig.from_yaml("configs/gan_signals/robustness.yaml")
    assert isinstance(config.model, TemporalCNNConfig)
    assert isinstance(config.pdw, PDWConfig)
    assert config.held_out == ("LPI-FMCW", "LPI-polyphase")
    assert config.augment_held_out_only is True
    assert config.synthetic_path.endswith(".h5")
