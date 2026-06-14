"""Dataset loaders compartidos entre todos los modelos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import yaml
from numpy.typing import NDArray

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
