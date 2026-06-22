"""Runners de las anclas Q1: ejecutan el pipeline de cada modelo y devuelven su AnchorResult."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from cog_ew.temporal_cnn_elint.train import TrainConfig as ElintTrainConfig
from cog_ew.temporal_cnn_elint.train import train as train_elint

if TYPE_CHECKING:
    from cog_ew.experiments.report import ExperimentProfile

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


def run_elint_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "elint"
    config = ElintTrainConfig.from_yaml(profile.elint_config)
    kw: dict[str, object] = dict(device=profile.device, seed=profile.seed, out_dir=str(run_dir))
    if profile.elint_epochs is not None:
        kw["epochs"] = profile.elint_epochs
    config = replace(config, **kw)  # type: ignore[arg-type]
    result = train_elint(config)
    achieved = float(result["test"]["lpi_accuracy"])
    return AnchorResult(
        name="elint",
        target=_TARGETS["elint"],
        achieved=achieved,
        baseline=None,
        passed=_passed(achieved, _TARGETS["elint"]),
        run_dir=str(run_dir),
    )
