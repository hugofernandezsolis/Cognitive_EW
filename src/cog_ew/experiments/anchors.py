"""Runners de las anclas Q1: ejecutan el pipeline de cada modelo y devuelven su AnchorResult."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from gymnasium.spaces import Discrete

from cog_ew.deep_rl_jamming.agent import D3QNAgent
from cog_ew.deep_rl_jamming.compare import AgentPolicy, BaselinePolicy, compare
from cog_ew.deep_rl_jamming.env import RadarJammingEnv
from cog_ew.deep_rl_jamming.train import TrainConfig as JammingTrainConfig
from cog_ew.deep_rl_jamming.train import train as train_jamming
from cog_ew.ew_library.library import EWResponseLibrary
from cog_ew.gan_signals.export import ExportConfig, export_synthetic
from cog_ew.gan_signals.robustness import RobustnessConfig, run_robustness_experiment
from cog_ew.gan_signals.train import WGANGPConfig
from cog_ew.gan_signals.train import train as train_gan
from cog_ew.marl_formation.compare import AgentPolicy as MarlAgentPolicy
from cog_ew.marl_formation.compare import compare_policies
from cog_ew.marl_formation.env import IADSFormationEnv
from cog_ew.marl_formation.train import TrainConfig as MarlTrainConfig
from cog_ew.marl_formation.train import train as train_marl
from cog_ew.temporal_cnn_elint.train import TrainConfig as ElintTrainConfig
from cog_ew.temporal_cnn_elint.train import train as train_elint
from cog_ew.temporal_cnn_elint.metrics import strict_elint_passed, strict_elint_score

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


def _overrides(**kwargs: object) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def run_elint_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "elint"
    config = ElintTrainConfig.from_yaml(profile.elint_config)
    config = replace(
        config,
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir),
        **_overrides(epochs=profile.elint_epochs),
    )
    result = train_elint(config)
    metrics = result["test"]
    achieved = strict_elint_score(metrics)
    return AnchorResult(
        name="elint",
        target=_TARGETS["elint"],
        achieved=achieved,
        baseline=None,
        passed=strict_elint_passed(metrics, target=_TARGETS["elint"]),
        run_dir=str(run_dir),
    )


def run_jamming_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "jamming"
    config = JammingTrainConfig.from_yaml(profile.jamming_config)
    config = replace(
        config,
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir),
        **_overrides(
            total_steps=profile.jamming_total_steps,
            eval_episodes=profile.jamming_eval_episodes,
        ),
    )
    train_jamming(config)

    env = RadarJammingEnv(config.env)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    obs_dim = int(np.prod(obs_shape))
    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    rng = np.random.default_rng(profile.seed)
    agent = D3QNAgent(obs_dim, n_actions, config.agent, profile.device, rng)
    state_dict = torch.load(run_dir / "best.pt", map_location=profile.device, weights_only=True)
    agent.online_net.load_state_dict(state_dict)

    library = EWResponseLibrary.from_yaml(profile.jamming_responses_config)
    cognitive = AgentPolicy(agent)
    baseline = BaselinePolicy(library, env)
    result = compare(env, cognitive, baseline, profile.jamming_compare_episodes, profile.seed)

    achieved = float(result["cognitive"]["win_rate"])
    baseline_wr = float(result["baseline"]["win_rate"])
    return AnchorResult(
        name="jamming",
        target=_TARGETS["jamming"],
        achieved=achieved,
        baseline=baseline_wr,
        passed=_passed(achieved, _TARGETS["jamming"]),
        run_dir=str(run_dir),
    )


def run_marl_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "marl"
    size = _overrides(
        total_episodes=profile.marl_total_episodes,
        eval_episodes=profile.marl_eval_episodes,
    )
    qmix_dir = run_dir / "qmix"
    iql_dir = run_dir / "iql"
    qmix_cfg = replace(
        MarlTrainConfig.from_yaml(profile.marl_qmix_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(qmix_dir),
        **size,
    )
    iql_cfg = replace(
        MarlTrainConfig.from_yaml(profile.marl_iql_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(iql_dir),
        **size,
    )
    train_marl(qmix_cfg)
    train_marl(iql_cfg)

    env = IADSFormationEnv(qmix_cfg.env)
    coordinated = MarlAgentPolicy.from_checkpoint(
        qmix_dir / "best.pt",
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=qmix_cfg.agent.hidden,
        n_agents=env.n_agents,
        device=profile.device,
    )
    independent = MarlAgentPolicy.from_checkpoint(
        iql_dir / "best.pt",
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=iql_cfg.agent.hidden,
        n_agents=env.n_agents,
        device=profile.device,
    )
    result = compare_policies(
        env,
        coordinated=coordinated,
        independent=independent,
        episodes=profile.marl_compare_episodes,
        seed=profile.seed,
    )
    achieved = float(result["relative_improvement"]["suppressed_fraction"])
    baseline = float(result["independent"]["suppressed_fraction"])
    return AnchorResult(
        name="marl",
        target=_TARGETS["marl"],
        achieved=achieved,
        baseline=baseline,
        passed=_passed(achieved, _TARGETS["marl"]),
        run_dir=str(run_dir),
    )


def run_gan_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "gan"
    gan_dir = run_dir / "wgan_gp"
    synth_path = run_dir / "synthetic.h5"

    gan_cfg = replace(
        WGANGPConfig.from_yaml(profile.gan_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(gan_dir),
        **_overrides(total_steps=profile.gan_total_steps),
    )
    train_gan(gan_cfg)

    export_cfg = replace(
        ExportConfig.from_yaml(profile.export_config),
        checkpoint=str(gan_dir / "best.pt"),
        out_path=str(synth_path),
        device=profile.device,
        seed=profile.seed,
        **_overrides(samples_per_type=profile.export_samples_per_type),
    )
    export_synthetic(export_cfg)

    rob_cfg = replace(
        RobustnessConfig.from_yaml(profile.robustness_config),
        synthetic_path=str(synth_path),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir / "robustness"),
        **_overrides(epochs=profile.robustness_epochs),
    )
    result = run_robustness_experiment(rob_cfg)

    achieved = float(result["relative_improvement"])
    baseline = float(result["baseline"])
    return AnchorResult(
        name="gan",
        target=_TARGETS["gan"],
        achieved=achieved,
        baseline=baseline,
        passed=_passed(achieved, _TARGETS["gan"]),
        run_dir=str(run_dir),
    )
