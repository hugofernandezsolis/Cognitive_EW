from cog_ew.data.loaders import MODULATIONS_2018, RadioMLConfig


def test_modulations_count():
    assert len(MODULATIONS_2018) == 24
    assert len(set(MODULATIONS_2018)) == 24


def test_config_defaults():
    config = RadioMLConfig()
    assert config.kaggle_dataset == "pinxau1000/radioml2018"
    assert config.h5_path is None
    assert config.normalize is True
    assert config.seed == 0


def test_config_from_yaml(tmp_path):
    yaml_text = (
        "h5_path: /data/foo.h5\n"
        "snr_range: [0, 18]\n"
        "modulations: [BPSK, QPSK]\n"
        "normalize: false\n"
        "seed: 7\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    config = RadioMLConfig.from_yaml(path)

    assert config.h5_path == "/data/foo.h5"
    assert config.snr_range == (0, 18)
    assert config.modulations == ("BPSK", "QPSK")
    assert config.normalize is False
    assert config.seed == 7
