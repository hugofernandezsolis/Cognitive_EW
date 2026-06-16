import json

from cog_ew.deep_rl_jamming.agent import D3QNConfig
from cog_ew.deep_rl_jamming.env import RadarEnvConfig
from cog_ew.deep_rl_jamming.train import TrainConfig

CONFIG = "configs/deep_rl_jamming/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)
    assert isinstance(config.env, RadarEnvConfig)
    assert isinstance(config.agent, D3QNConfig)
    assert config.env.history_k == 8
    assert config.agent.hidden == 128
    assert config.total_steps > 0


def _tiny_config(out_dir):
    env = RadarEnvConfig.from_yaml("configs/deep_rl_jamming/env.yaml")
    agent = D3QNConfig(
        hidden=16,
        buffer_size=500,
        batch_size=16,
        learning_starts=32,
        target_sync=100,
        epsilon_decay_steps=100,
    )
    return TrainConfig(
        env=env,
        agent=agent,
        total_steps=200,
        eval_episodes=3,
        eval_every=100,
        device="cpu",
        seed=0,
        out_dir=str(out_dir),
        tracking=False,
    )


def test_train_smoke_writes_artifacts_and_metrics(tmp_path):
    from cog_ew.deep_rl_jamming.train import train

    result = train(_tiny_config(tmp_path))

    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "run_meta.json").exists()
    assert len(result["win_rate_history"]) >= 1
    assert result["final"]["latency_mean_ms"] > 0.0

    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["seed"] == 0
    assert "torch" in meta["dependencies"]
    assert "gymnasium" in meta["dependencies"]


def test_train_is_deterministic(tmp_path):
    from cog_ew.deep_rl_jamming.train import train

    a = train(_tiny_config(tmp_path / "a"))
    b = train(_tiny_config(tmp_path / "b"))
    assert a["win_rate_history"] == b["win_rate_history"]
