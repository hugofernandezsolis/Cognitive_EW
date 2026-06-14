"""Taxonomía y parámetros de emisores para el pipeline PDW sintético (ELINT)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

CONTINUOUS_FEATURES: tuple[str, ...] = ("rf", "pw", "pa", "aoa", "pri")
INTRA_PULSE_MODS: tuple[str, ...] = ("none", "lfm", "barker", "fmcw", "polyphase")
MODES: tuple[str, ...] = ("search", "tws", "track", "missile_guidance")
THREAT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")

# Rangos físicos por feature continua (orden de CONTINUOUS_FEATURES):
# rf [GHz], pw [us], pa [lineal], aoa [deg], pri [us].
CONTINUOUS_RANGES: NDArray[np.float64] = np.array(
    [
        [0.5, 18.0],
        [0.05, 200.0],
        [0.0, 1.0],
        [-180.0, 180.0],
        [1.0, 10000.0],
    ],
    dtype=np.float64,
)

_MODE_TO_THREAT = {"search": 0, "tws": 1, "track": 2, "missile_guidance": 3}

LPI_MODS: frozenset[str] = frozenset({"fmcw", "polyphase"})


def mode_to_threat(mode: str) -> int:
    return _MODE_TO_THREAT[mode]


@dataclass(frozen=True)
class ModeSpec:
    rf_band: tuple[float, float]
    pri_pattern: str
    pri_range: tuple[float, float]
    pw_range: tuple[float, float]
    scan_period: float
    freq_hopping: bool
    lpi: bool
    intra_pulse_mods: tuple[str, ...]


@dataclass(frozen=True)
class EmitterSpec:
    name: str
    modes: dict[str, ModeSpec]


@dataclass(frozen=True)
class EmitterLibrary:
    emitters: tuple[EmitterSpec, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmitterLibrary:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        emitters = []
        for entry in raw["emitters"]:
            modes = {
                mode_name: ModeSpec(
                    rf_band=(float(spec["rf_band"][0]), float(spec["rf_band"][1])),
                    pri_pattern=spec["pri_pattern"],
                    pri_range=(float(spec["pri_range"][0]), float(spec["pri_range"][1])),
                    pw_range=(float(spec["pw_range"][0]), float(spec["pw_range"][1])),
                    scan_period=float(spec["scan_period"]),
                    freq_hopping=bool(spec["freq_hopping"]),
                    lpi=bool(spec["lpi"]),
                    intra_pulse_mods=tuple(spec["intra_pulse_mods"]),
                )
                for mode_name, spec in entry["modes"].items()
            }
            emitters.append(EmitterSpec(name=entry["name"], modes=modes))
        return cls(emitters=tuple(emitters))

    def emitter_names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.emitters)

    def lpi_indices(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, emitter in enumerate(self.emitters)
            if any(mode_spec.lpi for mode_spec in emitter.modes.values())
        )
