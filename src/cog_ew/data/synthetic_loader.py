"""Loader del HDF5 sintético de la GAN para aumentar el clasificador ELINT."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticPDWDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        emitters: tuple[int, ...] | None = None,
        known_only: bool = True,
    ) -> None:
        with h5py.File(hdf5_path, "r") as fh:
            x = np.asarray(fh["X"], dtype=np.float32)
            source_a = np.asarray(fh["source_a"], dtype=np.int64)
            is_known = np.asarray(fh["is_known"], dtype=bool)
        mask = np.ones(x.shape[0], dtype=bool)
        if known_only:
            mask &= is_known
        if emitters is not None:
            mask &= np.isin(source_a, np.asarray(emitters, dtype=np.int64))
        self._x = torch.from_numpy(np.ascontiguousarray(x[mask]))
        self._type = torch.from_numpy(np.ascontiguousarray(source_a[mask]))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        return self._x[index], int(self._type[index]), -1, -1
