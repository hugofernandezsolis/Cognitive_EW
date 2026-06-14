"""Generador sintético de trenes de pulsos PDW etiquetados (ELINT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PulseTrain:
    toa: NDArray[np.float64]
    rf: NDArray[np.float64]
    pw: NDArray[np.float64]
    pa: NDArray[np.float64]
    aoa: NDArray[np.float64]
    intra_pulse_mod: NDArray[np.int64]


def _generate_pri(
    pattern: str, pri_range: tuple[float, float], n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    lo, hi = pri_range
    if pattern == "fixed":
        return np.full(n, (lo + hi) / 2.0)
    if pattern == "jitter":
        return rng.uniform(lo, hi, n)
    if pattern == "stagger":
        levels = np.linspace(lo, hi, 3)
        return np.resize(levels, n).astype(np.float64)
    raise ValueError(f"Patrón de PRI desconocido: {pattern}")
