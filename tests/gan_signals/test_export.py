from cog_ew.gan_signals.export import ExportConfig


def test_export_config_from_yaml():
    config = ExportConfig.from_yaml("configs/gan_signals/export.yaml")
    assert config.n_emitters == 8
    assert isinstance(config.alphas, tuple)
    assert config.samples_per_type > 0
    assert config.out_path.endswith(".h5")
