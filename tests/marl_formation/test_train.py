from dataclasses import replace

import numpy as np

from cog_ew.marl_formation.train import (
    EpisodeReplayBuffer,
    TrainConfig,
    _is_better_checkpoint,
    train,
)

CONFIG = "configs/marl_formation/qmix.yaml"


def test_episode_buffer_add_and_sample_shapes():
    buf = EpisodeReplayBuffer(capacity=10, horizon=6, n_agents=4, obs_dim=24, state_dim=28)
    for _ in range(3):
        buf.add(
            obs=np.zeros((6, 4, 24), dtype=np.float32),
            actions=np.zeros((6, 4), dtype=np.int64),
            rewards=np.zeros(6, dtype=np.float32),
            states=np.zeros((6, 28), dtype=np.float32),
            dones=np.zeros(6, dtype=np.float32),
            filled=np.ones(6, dtype=np.float32),
        )
    assert len(buf) == 3
    obs, actions, rewards, states, dones, filled = buf.sample(2, np.random.default_rng(0))
    assert obs.shape == (2, 6, 4, 24)
    assert actions.shape == (2, 6, 4)
    assert rewards.shape == (2, 6)
    assert states.shape == (2, 6, 28)
    assert dones.shape == (2, 6)
    assert filled.shape == (2, 6)


def test_episode_buffer_respects_capacity():
    buf = EpisodeReplayBuffer(capacity=2, horizon=4, n_agents=4, obs_dim=24, state_dim=28)
    for _ in range(5):
        buf.add(
            obs=np.zeros((4, 4, 24), dtype=np.float32),
            actions=np.zeros((4, 4), dtype=np.int64),
            rewards=np.zeros(4, dtype=np.float32),
            states=np.zeros((4, 28), dtype=np.float32),
            dones=np.zeros(4, dtype=np.float32),
            filled=np.ones(4, dtype=np.float32),
        )
    assert len(buf) == 2


def test_train_config_from_yaml_parses_sections():
    config = TrainConfig.from_yaml(CONFIG)
    assert config.env.n_agents == 4
    assert config.agent.gamma == 0.99
    assert config.total_episodes > 0


def test_checkpoint_selection_tiebreaks_on_suppressed_fraction():
    assert _is_better_checkpoint(1.0, 0.7, (1.0, 0.5))
    assert not _is_better_checkpoint(1.0, 0.4, (1.0, 0.5))
    assert _is_better_checkpoint(0.9, 1.0, (0.8, 1.0))
    assert not _is_better_checkpoint(0.8, 1.0, (0.9, 0.0))


def _smoke_config(config: TrainConfig, out_dir) -> TrainConfig:
    agent = replace(
        config.agent,
        hidden=16,
        mixer_embed_dim=8,
        hypernet_hidden=16,
        learning_starts_episodes=2,
        batch_episodes=2,
        buffer_episodes=8,
        target_sync=2,
        epsilon_decay_steps=4,
    )
    env = replace(config.env, horizon_t=8)
    return replace(
        config,
        env=env,
        agent=agent,
        total_episodes=6,
        eval_episodes=3,
        eval_every=3,
        out_dir=str(out_dir),
    )


def test_train_smoke_produces_metrics(tmp_path):
    config = _smoke_config(TrainConfig.from_yaml(CONFIG), tmp_path)
    result = train(config)
    assert 0.0 <= result["final"]["win_rate"] <= 1.0
    assert np.isfinite(result["final"]["latency_mean_ms"])
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_train_is_deterministic_by_seed(tmp_path):
    a = train(_smoke_config(TrainConfig.from_yaml(CONFIG), tmp_path / "a"))
    b = train(_smoke_config(TrainConfig.from_yaml(CONFIG), tmp_path / "b"))
    assert a["win_rate_history"] == b["win_rate_history"]
