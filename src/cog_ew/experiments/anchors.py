"""Runners de las anclas Q1: ejecutan el pipeline de cada modelo y devuelven su AnchorResult."""

from __future__ import annotations

import math
from dataclasses import dataclass

_TARGETS: dict[str, float] = {"jamming": 0.92, "elint": 0.96, "marl": 0.45, "gan": 0.22}


@dataclass(frozen=True)
class AnchorResult:
    name: str
    target: float
    achieved: float
    baseline: float | None
    passed: bool
    run_dir: str


def _passed(achieved: float, target: float) -> bool:
    return math.isfinite(achieved) and achieved >= target
