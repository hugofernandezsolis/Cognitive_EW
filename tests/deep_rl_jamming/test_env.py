import numpy as np

from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.ew_library.library import JammingTechnique

CONFIG = "configs/deep_rl_jamming/env.yaml"


def _env():
    return RadarJammingEnv(RadarEnvConfig.from_yaml(CONFIG))


def test_config_from_yaml_loads_parameters():
    config = RadarEnvConfig.from_yaml(CONFIG)
    assert config.history_k == 8
    assert config.horizon_t == 64
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert config.effectiveness["none"]["missile_guidance"] == 0.0


def test_reset_returns_obs_with_correct_shape():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 5)
    assert obs.dtype == np.float32
    assert info["outcome"] == "ongoing"
    assert info["real_mode"] == "search"


def test_action_and_observation_spaces():
    env = _env()
    assert env.action_space.n == 10 * 4
    assert env.observation_space.shape == (8, 5)


def test_reset_is_deterministic_by_seed():
    a, _ = _env().reset(seed=7)
    b, _ = _env().reset(seed=7)
    assert np.array_equal(a, b)


def test_encode_action_roundtrip():
    env = _env()
    action = env.encode_action(JammingTechnique.NONE, 0)
    assert action == list(JammingTechnique).index(JammingTechnique.NONE) * 4


OPTIMAL = {
    "search": [JammingTechnique.NOISE, JammingTechnique.DECEPTION],
    "tws": [JammingTechnique.DECEPTION, JammingTechnique.NOISE],
    "track": [JammingTechnique.CROSS_EYE, JammingTechnique.VGPO],
    "missile_guidance": [JammingTechnique.CHAFF, JammingTechnique.RGPO],
}


def test_step_returns_gym_tuple():
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8, 5)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["outcome"] in {"ongoing", "win", "lose"}


def test_passive_jammer_loses():
    env = _env()
    env.reset(seed=0)
    passive = env.encode_action(JammingTechnique.NONE, 0)
    outcome = "ongoing"
    for _ in range(env.config.horizon_t):
        _, _, terminated, truncated, info = env.step(passive)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome == "lose"


def test_optimal_jammer_wins():
    env = _env()
    _, info = env.reset(seed=0)
    last_technique = None
    terminated = truncated = False
    while not (terminated or truncated):
        ranked = OPTIMAL[info["real_mode"]]
        technique = ranked[1] if info["eccm_active"] and last_technique == ranked[0] else ranked[0]
        last_technique = technique
        action = env.encode_action(technique, env._n_power - 1)
        _, _, terminated, truncated, info = env.step(action)
    assert info["outcome"] == "win"


def test_rollout_is_deterministic_by_seed():
    def rollout() -> list[float]:
        env = _env()
        env.reset(seed=3)
        rewards = []
        for _ in range(10):
            _, r, term, trunc, _ = env.step(5)
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    assert rollout() == rollout()


def test_info_exposes_emitter_name():
    from cog_ew.data.pdw_library import EmitterLibrary

    env = _env()
    _, info = env.reset(seed=0)
    names = EmitterLibrary.from_yaml("configs/temporal_cnn_elint/emitters.yaml").emitter_names()
    assert info["emitter"] in names


def test_env_exposes_n_power_levels():
    env = _env()
    assert env.n_power_levels == 4
