import numpy as np
import pytest
import torch

from cog_ew.data.loaders import (
    MODULATIONS_2018,
    RadioML2018Dataset,
    RadioMLConfig,
    _mask_to_runs,
    resolve_h5_path,
    split_dataset,
)


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


def test_config_rejects_unknown_modulation():
    with pytest.raises(ValueError):
        RadioMLConfig(modulations=("BPSK", "NOPE"))


def test_resolve_h5_path_kaggle_multiple_h5(tmp_path, monkeypatch):
    download_dir = tmp_path / "kaggle_cache"
    download_dir.mkdir()
    (download_dir / "a.h5").write_bytes(b"")
    (download_dir / "b.h5").write_bytes(b"")

    import cog_ew.data.loaders as loaders

    monkeypatch.setattr(loaders.kagglehub, "dataset_download", lambda _: str(download_dir))
    config = RadioMLConfig(h5_path=None)

    with pytest.raises(FileNotFoundError):
        resolve_h5_path(config)


def test_mask_to_runs_basic():
    mask = np.array([False, True, True, False, True])

    assert _mask_to_runs(mask) == [(1, 3), (4, 5)]


def test_mask_to_runs_empty():
    mask = np.zeros(5, dtype=bool)

    assert _mask_to_runs(mask) == []


def test_mask_to_runs_all_true():
    mask = np.ones(3, dtype=bool)

    assert _mask_to_runs(mask) == [(0, 3)]


def test_dataset_loads_full(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 3 * 3 * 4  # mods * snrs * frames

    iq, label, snr = dataset[0]
    assert isinstance(iq, torch.Tensor)
    assert iq.shape == (2, 8)
    assert iq.dtype == torch.float32
    assert iq.device.type == "cpu"
    assert label == 0
    assert snr == -4


def test_dataset_filters_snr(synthetic_h5):
    config = RadioMLConfig(
        h5_path=str(synthetic_h5), kaggle_dataset=None, snr_range=(0, 4), normalize=False
    )
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 3 * 2 * 4  # solo SNR 0 y 4
    snrs = {int(dataset[i][2]) for i in range(len(dataset))}
    assert snrs == {0, 4}


def test_dataset_filters_modulations(synthetic_h5):
    keep = (MODULATIONS_2018[0], MODULATIONS_2018[2])
    config = RadioMLConfig(
        h5_path=str(synthetic_h5), kaggle_dataset=None, modulations=keep, normalize=False
    )
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 2 * 3 * 4  # 2 mods * 3 snrs * 4 frames
    labels = {int(dataset[i][1]) for i in range(len(dataset))}
    assert labels == {0, 2}


def test_dataset_normalizes(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=True)
    dataset = RadioML2018Dataset(config)

    iq, _, _ = dataset[5]
    power = float((iq**2).sum(dim=0).mean())
    assert abs(power - 1.0) < 1e-4


def test_split_dataset_sizes(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)  # 36 ejemplos

    train, val, test = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)

    assert len(train) + len(val) + len(test) == len(dataset)
    assert len(train) == 18
    assert len(val) == 9
    assert len(test) == 9


def test_dataset_label_snr_x_alignment(synthetic_h5):
    config = RadioMLConfig(
        h5_path=str(synthetic_h5), kaggle_dataset=None, snr_range=(0, 4), normalize=False
    )
    dataset = RadioML2018Dataset(config)

    # Primer ejemplo tras filtrar SNR>=0 es la fila original 4: X==5.0, label=0, snr=0
    iq, label, snr = dataset[0]
    assert float(iq[0, 0]) == pytest.approx(5.0)
    assert label == 0
    assert snr == 0


def test_split_dataset_deterministic(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)

    a = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)[0].indices
    b = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)[0].indices
    c = split_dataset(dataset, (0.5, 0.25, 0.25), seed=1)[0].indices

    assert list(a) == list(b)
    assert list(a) != list(c)
