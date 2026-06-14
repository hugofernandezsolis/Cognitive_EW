"""Dataset loaders compartidos entre todos los modelos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import kagglehub
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset, random_split

from cog_ew.data.preprocessing import normalize_power, to_channels_first

MODULATIONS_2018: tuple[str, ...] = (
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
)


@dataclass
class RadioMLConfig:
    h5_path: str | None = None
    kaggle_dataset: str | None = "pinxau1000/radioml2018"
    snr_range: tuple[int, int] | None = None
    modulations: tuple[str, ...] | None = None
    normalize: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.modulations is not None:
            unknown = set(self.modulations) - set(MODULATIONS_2018)
            if unknown:
                raise ValueError(f"Modulaciones desconocidas en config: {sorted(unknown)}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RadioMLConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if "snr_range" in raw and raw["snr_range"] is not None:
            raw["snr_range"] = tuple(raw["snr_range"])
        if "modulations" in raw and raw["modulations"] is not None:
            raw["modulations"] = tuple(raw["modulations"])
        return cls(**raw)


def resolve_h5_path(config: RadioMLConfig) -> Path:
    if config.h5_path is not None:
        path = Path(config.h5_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"h5_path no existe: {path}")
    if config.kaggle_dataset is not None:
        download_dir = Path(kagglehub.dataset_download(config.kaggle_dataset))
        candidates = sorted(download_dir.rglob("*.h5"))
        if not candidates:
            raise FileNotFoundError(
                f"No se encontró ningún .h5 en la descarga de {config.kaggle_dataset}"
            )
        if len(candidates) > 1:
            names = [str(c) for c in candidates]
            raise FileNotFoundError(
                f"Se encontraron varios .h5 en la descarga de {config.kaggle_dataset}; "
                f"especifica h5_path explícitamente: {names}"
            )
        return candidates[0]
    raise FileNotFoundError("Define h5_path o kaggle_dataset en RadioMLConfig")


def _mask_to_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev + 1))
        start = prev = current
    runs.append((start, prev + 1))
    return runs


class RadioML2018Dataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, config: RadioMLConfig) -> None:
        path = resolve_h5_path(config)
        with h5py.File(path, "r") as fh:
            labels = np.asarray(fh["Y"]).argmax(axis=1)
            snr = np.asarray(fh["Z"])[:, 0].astype(np.int64)
            mask = np.ones(labels.shape[0], dtype=bool)
            if config.snr_range is not None:
                low, high = config.snr_range
                mask &= (snr >= low) & (snr <= high)
            if config.modulations is not None:
                keep_idx = {MODULATIONS_2018.index(name) for name in config.modulations}
                mask &= np.isin(labels, list(keep_idx))
            runs = _mask_to_runs(mask)
            chunks = [np.asarray(fh["X"][start:stop]) for start, stop in runs]

        x = (
            np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0, 2), dtype=np.float32)
        ).astype(np.float32)
        if config.normalize and x.shape[0] > 0:
            x = normalize_power(x)
        x = to_channels_first(x)

        self._x = torch.from_numpy(np.ascontiguousarray(x))
        self._labels = torch.from_numpy(labels[mask].astype(np.int64))
        self._snr = torch.from_numpy(snr[mask])

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self._x[index], int(self._labels[index]), int(self._snr[index])


def split_dataset(
    dataset: Dataset[tuple[torch.Tensor, int, int]],
    fractions: Sequence[float],
    seed: int,
) -> list[Subset[tuple[torch.Tensor, int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(fractions), generator=generator)
