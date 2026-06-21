import h5py
import numpy as np

from cog_ew.data.synthetic_loader import SyntheticPDWDataset


def _make_synth(path, source_a, is_known):
    n = len(source_a)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(n, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.asarray(source_a, dtype=np.int64))
        fh.create_dataset("is_known", data=np.asarray(is_known, dtype=bool))
    return str(path)


def test_synthetic_dataset_returns_four_tuple(tmp_path):
    path = _make_synth(tmp_path / "s.h5", [6, 7, 6], [True, True, True])
    ds = SyntheticPDWDataset(path)
    x, type_id, mode, threat = ds[0]
    assert x.shape == (10, 64)
    assert type_id == 6
    assert mode == -1 and threat == -1
    assert len(ds) == 3


def test_synthetic_dataset_filters_emitters_and_known(tmp_path):
    path = _make_synth(
        tmp_path / "s.h5",
        source_a=[6, 7, 6, 0],
        is_known=[True, True, False, True],
    )
    ds = SyntheticPDWDataset(path, emitters=(6, 7), known_only=True)
    types = sorted(ds[i][1] for i in range(len(ds)))
    assert types == [6, 7]
