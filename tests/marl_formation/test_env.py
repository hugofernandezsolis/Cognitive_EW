from dataclasses import replace

import numpy as np

from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv

CONFIG = "configs/marl_formation/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = IADSEnvConfig.from_yaml(CONFIG)
    assert config.n_agents == 4
    assert config.n_radars == 4
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert "noise" in config.suppression_techniques


def _env():
    return IADSFormationEnv(IADSEnvConfig.from_yaml(CONFIG))


def test_reset_returns_obs_state_info():
    env = _env()
    obs, state, info = env.reset(seed=0)
    assert set(obs) == set(range(4))
    assert obs[0].shape == (4 * 5 + 4,)
    assert obs[0].dtype == np.float32
    assert state.shape == (4 * (4 + 2) + 4,)
    assert info["outcome"] == "ongoing"


def test_reset_is_deterministic_by_seed():
    a, sa, _ = _env().reset(seed=3)
    b, sb, _ = _env().reset(seed=3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(sa, sb)


def test_encode_action_roundtrip():
    env = _env()
    action = env.encode_action(target=2, jam_type=1, power_level=3)
    assert env._decode_action(action) == (2, 1, 3)
    assert env.action_dim == 4 * 3 * 4


def _small_env():
    config = replace(IADSEnvConfig.from_yaml(CONFIG), n_agents=2, n_radars=2)
    return IADSFormationEnv(config)


def _all_actions(env, target, jam_type, power_level):
    action = env.encode_action(target, jam_type, power_level)
    return {a: action for a in range(env.n_agents)}


def test_step_returns_ctde_tuple_with_shared_reward():
    env = _small_env()
    env.reset(seed=0)
    obs, state, rewards, terminated, truncated, info = env.step(_all_actions(env, 0, 2, 3))
    assert set(obs) == {0, 1}
    assert obs[0].shape == (2 * 5 + 2,)
    assert len(set(rewards.values())) == 1
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 0.0 <= info["suppressed_fraction"] <= 1.0


def test_splitting_targets_covers_more_than_concentrating():
    split_env = _small_env()
    split_env.reset(seed=0)
    split_actions = {
        0: split_env.encode_action(0, 2, 3),
        1: split_env.encode_action(1, 2, 3),
    }
    _, _, _, _, _, split_info = split_env.step(split_actions)

    conc_env = _small_env()
    conc_env.reset(seed=0)
    _, _, _, _, _, conc_info = conc_env.step(_all_actions(conc_env, 0, 2, 3))

    assert split_info["suppressed_fraction"] >= conc_info["suppressed_fraction"]


def test_uncovered_radars_lead_to_loss():
    env = _small_env()
    env.reset(seed=0)
    passive = _all_actions(env, 0, 0, 0)
    outcome = "ongoing"
    for _ in range(env.config.horizon_t):
        _, _, _, terminated, truncated, info = env.step(passive)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome == "lose"


def test_rollout_is_deterministic_by_seed():
    def rollout():
        env = _small_env()
        env.reset(seed=1)
        rewards = []
        for _ in range(10):
            _, _, r, term, trunc, _ = env.step(_all_actions(env, 0, 2, 2))
            rewards.append(r[0])
            if term or trunc:
                break
        return rewards

    assert rollout() == rollout()
