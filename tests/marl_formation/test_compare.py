import numpy as np
import torch

from cog_ew.marl_formation.agents import AgentRNN
from cog_ew.marl_formation.compare import (
    AgentPolicy,
    ConcentratedSuppressionPolicy,
    SpreadSuppressionPolicy,
    compare_policies,
    evaluate_policy,
)
from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv

CONFIG = "configs/marl_formation/env.yaml"


def _env() -> IADSFormationEnv:
    return IADSFormationEnv(IADSEnvConfig.from_yaml(CONFIG))


def test_concentrated_policy_points_every_agent_to_same_radar():
    env = _env()
    policy = ConcentratedSuppressionPolicy()
    obs, state, info = env.reset(seed=0)
    actions = policy.act(env, obs, state, info)
    expected = env.encode_action(target=0, jam_type=2, power_level=len(env.config.power_levels) - 1)
    assert actions == {agent: expected for agent in range(env.n_agents)}


def test_spread_policy_distributes_targets_round_robin():
    env = _env()
    policy = SpreadSuppressionPolicy()
    obs, state, info = env.reset(seed=0)
    actions = policy.act(env, obs, state, info)
    decoded = [env._decode_action(actions[agent]) for agent in range(env.n_agents)]
    assert decoded == [(0, 2, 3), (1, 2, 3), (2, 2, 3), (3, 2, 3)]


def test_agent_policy_returns_valid_actions():
    env = _env()
    agent = AgentRNN(env.obs_dim, env.action_dim, hidden=16)
    policy = AgentPolicy(agent=agent, n_agents=env.n_agents, device="cpu")
    obs, state, info = env.reset(seed=0)
    policy.reset(env)
    actions = policy.act(env, obs, state, info)
    assert set(actions) == set(range(env.n_agents))
    assert all(0 <= action < env.action_dim for action in actions.values())


def test_agent_policy_loads_weights_from_checkpoint(tmp_path):
    env = _env()
    agent = AgentRNN(env.obs_dim, env.action_dim, hidden=16)
    path = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), path)
    policy = AgentPolicy.from_checkpoint(
        path,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=16,
        n_agents=env.n_agents,
        device="cpu",
    )
    obs, state, info = env.reset(seed=0)
    policy.reset(env)
    assert set(policy.act(env, obs, state, info)) == set(range(env.n_agents))


def test_evaluate_policy_is_deterministic_by_seed():
    a = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    b = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    assert a == b
    assert 0.0 <= a["win_rate"] <= 1.0
    assert np.isfinite(a["mean_reward"])


def test_spread_covers_at_least_concentrated():
    spread = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    concentrated = evaluate_policy(_env(), ConcentratedSuppressionPolicy(), episodes=5, seed=0)
    assert spread["suppressed_fraction"] >= concentrated["suppressed_fraction"]
    assert spread["win_rate"] >= concentrated["win_rate"]


def test_compare_policies_reports_absolute_and_relative():
    result = compare_policies(
        _env(),
        coordinated=SpreadSuppressionPolicy(),
        independent=ConcentratedSuppressionPolicy(),
        episodes=5,
        seed=0,
    )
    assert set(result) == {"coordinated", "independent", "delta", "relative_improvement"}
    assert result["delta"]["suppressed_fraction"] == (
        result["coordinated"]["suppressed_fraction"] - result["independent"]["suppressed_fraction"]
    )
    indep = result["independent"]["suppressed_fraction"]
    coord = result["coordinated"]["suppressed_fraction"]
    expected_rel = (coord - indep) / indep if indep > 0 else float("inf")
    assert result["relative_improvement"]["suppressed_fraction"] == expected_rel
