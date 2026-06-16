"""Librería de respuestas EW pre-programadas por tipo de amenaza (baseline convencional)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from cog_ew.data.pdw_library import MODES


class JammingTechnique(Enum):
    NOISE = "noise"
    DRFM_REPEATER = "drfm_repeater"
    DECEPTION = "deception"
    CROSS_EYE = "cross_eye"
    VGPO = "vgpo"
    RGPO = "rgpo"
    CHAFF = "chaff"
    DECOY = "decoy"
    EVASIVE = "evasive"
    NONE = "none"


def _parse_techniques(names: list[str]) -> tuple[JammingTechnique, ...]:
    try:
        return tuple(JammingTechnique(name) for name in names)
    except ValueError as exc:
        raise ValueError(f"técnica desconocida en la librería EW: {exc}") from exc


@dataclass(frozen=True)
class EWResponseLibrary:
    rules: dict[tuple[str, str], tuple[JammingTechnique, ...]]
    defaults: dict[str, tuple[JammingTechnique, ...]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EWResponseLibrary:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        rules: dict[tuple[str, str], tuple[JammingTechnique, ...]] = {}
        for emitter, modes in raw["rules"].items():
            for mode, techniques in modes.items():
                rules[(emitter, mode)] = _parse_techniques(techniques)
        defaults = {
            mode: _parse_techniques(techniques) for mode, techniques in raw["defaults"].items()
        }
        return cls(rules=rules, defaults=defaults)

    def select(self, emitter: str, mode: str) -> tuple[JammingTechnique, ...]:
        if mode not in MODES:
            raise ValueError(f"modo desconocido: {mode!r}")
        if (emitter, mode) in self.rules:
            return self.rules[(emitter, mode)]
        return self.defaults[mode]
