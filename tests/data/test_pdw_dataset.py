import torch

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset

CONFIG_PATH = "configs/temporal_cnn_elint/emitters.yaml"


def _config(**kw):
    base = dict(
        library_path=CONFIG_PATH,
        emitters=("LPI-FMCW",),
        modes=("search",),
        window=64,
        n_pulses=128,
        n_trains=3,
        seed=0,
    )
    base.update(kw)
    return PDWConfig(**base)


def test_dataset_item_shape_and_labels():
    ds = PDWSyntheticDataset(_config())
    assert len(ds) == 3 * (128 // 64)  # n_trains * ventanas por tren

    pdw, type_idx, mode_idx, threat_idx = ds[0]
    assert isinstance(pdw, torch.Tensor)
    assert pdw.shape == (10, 64)
    assert pdw.dtype == torch.float32
    assert pdw.device.type == "cpu"
    assert type_idx == 6  # índice de LPI-FMCW en la librería
    assert mode_idx == 0  # search
    assert threat_idx == 0  # search -> low


def test_dataset_filters_modes():
    ds = PDWSyntheticDataset(_config(modes=("search", "track")))
    modes = {int(ds[i][2]) for i in range(len(ds))}
    assert modes == {0, 2}  # search=0, track=2


def test_dataset_split_deterministic():
    ds = PDWSyntheticDataset(_config(n_trains=8))
    a = split_dataset(ds, (0.5, 0.25, 0.25), seed=0)[0].indices
    b = split_dataset(ds, (0.5, 0.25, 0.25), seed=0)[0].indices
    assert list(a) == list(b)
