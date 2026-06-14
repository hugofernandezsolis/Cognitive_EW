import h5py
import numpy as np
import pytest

from cog_ew.data.loaders import MODULATIONS_2018

N_MODS = 3
SNRS = (-4, 0, 4)
FRAMES = 4
N_SAMPLES = 8
N_CLASSES = len(MODULATIONS_2018)


@pytest.fixture
def synthetic_h5(tmp_path):
    rows = N_MODS * len(SNRS) * FRAMES
    x = np.zeros((rows, N_SAMPLES, 2), dtype=np.float32)
    y = np.zeros((rows, N_CLASSES), dtype=np.float32)
    z = np.zeros((rows, 1), dtype=np.int64)

    row = 0
    for mod_idx in range(N_MODS):
        for snr in SNRS:
            for _ in range(FRAMES):
                x[row] = float(row) + 1.0
                y[row, mod_idx] = 1.0
                z[row, 0] = snr
                row += 1

    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=x)
        fh.create_dataset("Y", data=y)
        fh.create_dataset("Z", data=z)
    return path
