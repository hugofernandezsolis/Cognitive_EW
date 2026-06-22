"""Perfiles de experimento y agregación del reporte de anclas Q1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
