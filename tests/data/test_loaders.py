import numpy as np
import pytest

from cog_ew.data.loaders import MODULATIONS_2018, RadioMLConfig, _mask_to_runs, resolve_h5_path


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


def test_resolve_h5_path_explicit_existing(tmp_path):
    h5 = tmp_path / "data.h5"
    h5.write_bytes(b"")
    config = RadioMLConfig(h5_path=str(h5))

    assert resolve_h5_path(config) == h5


def test_resolve_h5_path_explicit_missing():
    config = RadioMLConfig(h5_path="/nope/missing.h5")

    with pytest.raises(FileNotFoundError):
        resolve_h5_path(config)


def test_resolve_h5_path_none_set():
    config = RadioMLConfig(h5_path=None, kaggle_dataset=None)

    with pytest.raises(FileNotFoundError):
        resolve_h5_path(config)


def test_resolve_h5_path_kaggle(tmp_path, monkeypatch):
    download_dir = tmp_path / "kaggle_cache"
    download_dir.mkdir()
    (download_dir / "GOLD_XYZ_OSC.0001_1024x2M.h5").write_bytes(b"")

    import cog_ew.data.loaders as loaders

    def fake_download(dataset: str) -> str:
        assert dataset == "pinxau1000/radioml2018"
        return str(download_dir)

    monkeypatch.setattr(loaders.kagglehub, "dataset_download", fake_download)
    config = RadioMLConfig(h5_path=None)

    assert resolve_h5_path(config).name == "GOLD_XYZ_OSC.0001_1024x2M.h5"


def test_mask_to_runs_basic():
    mask = np.array([False, True, True, False, True])

    assert _mask_to_runs(mask) == [(1, 3), (4, 5)]


def test_mask_to_runs_empty():
    mask = np.zeros(5, dtype=bool)

    assert _mask_to_runs(mask) == []


def test_mask_to_runs_all_true():
    mask = np.ones(3, dtype=bool)

    assert _mask_to_runs(mask) == [(0, 3)]
