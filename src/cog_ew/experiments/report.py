"""Perfiles de experimento y agregación del reporte de anclas Q1."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cog_ew.experiments.anchors import (
    AnchorResult,
    run_elint_anchor,
    run_gan_anchor,
    run_jamming_anchor,
    run_marl_anchor,
)


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    device: str
    seed: int
    jamming_config: str
    jamming_responses_config: str
    jamming_total_steps: int | None
    jamming_eval_episodes: int | None
    jamming_compare_episodes: int
    elint_config: str
    elint_epochs: int | None
    marl_qmix_config: str
    marl_iql_config: str
    marl_total_episodes: int | None
    marl_eval_episodes: int | None
    marl_compare_episodes: int
    gan_config: str
    export_config: str
    robustness_config: str
    gan_total_steps: int | None
    export_samples_per_type: int | None
    robustness_epochs: int | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentProfile:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)


ANCHOR_RUNNERS: dict[str, Callable[[ExperimentProfile, Path], AnchorResult]] = {
    "jamming": run_jamming_anchor,
    "elint": run_elint_anchor,
    "marl": run_marl_anchor,
    "gan": run_gan_anchor,
}


def _config_hash(profile: ExperimentProfile) -> str:
    blob = json.dumps(asdict(profile), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def run_anchors(
    names: tuple[str, ...], profile: ExperimentProfile, out_dir: str | Path
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors: dict[str, Any] = {}
    for name in names:
        result = ANCHOR_RUNNERS[name](profile, out_dir)
        anchors[name] = {
            "target": result.target,
            "achieved": result.achieved,
            "baseline": result.baseline,
            "passed": result.passed,
            "run_dir": result.run_dir,
        }
    report: dict[str, Any] = {
        "profile_name": profile.name,
        "seed": profile.seed,
        "config_hash": _config_hash(profile),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "anchors": anchors,
    }
    (out_dir / "anchors_report.json").write_text(json.dumps(report, indent=2))
    return report
