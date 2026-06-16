import numpy as np

from cog_ew.deep_rl_jamming.agent import D3QNAgent, D3QNConfig
from cog_ew.deep_rl_jamming.compare import (
    AgentPolicy,
    BaselinePolicy,
    compare,
    evaluate_policy,
)
from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.ew_library.library import EWResponseLibrary, JammingTechnique

ENV_CONFIG = "configs/deep_rl_jamming/env.yaml"
LIB_CONFIG = "configs/ew_library/responses.yaml"


def _env():
    return RadarJammingEnv(RadarEnvConfig.from_yaml(ENV_CONFIG))


def _library():
    return EWResponseLibrary.from_yaml(LIB_CONFIG)


def test_baseline_policy_selects_top_technique_at_max_power():
    env = _env()
    library = _library()
    policy = BaselinePolicy(library, env)
    info = {"emitter": "S-400", "real_mode": "missile_guidance"}
    top = library.select("S-400", "missile_guidance")[0]
    expected = env.encode_action(top, env.n_power_levels - 1)
    assert policy.act(np.zeros((8, 5), dtype=np.float32), info) == expected


def test_baseline_policy_lpi_uses_poor_technique():
    env = _env()
    library = _library()
    policy = BaselinePolicy(library, env)
    info = {"emitter": "LPI-FMCW", "real_mode": "track"}
    top = library.select("LPI-FMCW", "track")[0]
    assert top == JammingTechnique.EVASIVE
    expected = env.encode_action(JammingTechnique.EVASIVE, env.n_power_levels - 1)
    assert policy.act(np.zeros((8, 5), dtype=np.float32), info) == expected


_OPTIMAL = {
    "search": [JammingTechnique.NOISE, JammingTechnique.DECEPTION],
    "tws": [JammingTechnique.DECEPTION, JammingTechnique.NOISE],
    "track": [JammingTechnique.CROSS_EYE, JammingTechnique.VGPO],
    "missile_guidance": [JammingTechnique.CHAFF, JammingTechnique.RGPO],
}


class _OraclePolicy:
    def __init__(self, env):
        self.env = env
        self.last = None

    def act(self, obs, info):
        ranked = _OPTIMAL[info["real_mode"]]
        technique = ranked[1] if info["eccm_active"] and self.last == ranked[0] else ranked[0]
        self.last = technique
        return self.env.encode_action(technique, self.env.n_power_levels - 1)


def _baseline(env):
    return BaselinePolicy(_library(), env)


def test_evaluate_policy_returns_bounded_metrics():
    env = _env()
    metrics = evaluate_policy(env, _baseline(env), episodes=5, seed=0)
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert np.isfinite(metrics["mean_reward"])


def test_evaluate_policy_is_deterministic_by_seed():
    env_a = _env()
    env_b = _env()
    a = evaluate_policy(env_a, _baseline(env_a), episodes=5, seed=0)
    b = evaluate_policy(env_b, _baseline(env_b), episodes=5, seed=0)
    assert a == b


def test_oracle_beats_or_matches_fixed_baseline():
    base_env = _env()
    oracle_env = _env()
    base_metrics = evaluate_policy(base_env, _baseline(base_env), episodes=8, seed=0)
    oracle_metrics = evaluate_policy(oracle_env, _OraclePolicy(oracle_env), episodes=8, seed=0)
    assert oracle_metrics["win_rate"] >= base_metrics["win_rate"]


def test_compare_delta_is_difference():
    env = _env()
    cognitive = AgentPolicy(
        D3QNAgent(40, 40, D3QNConfig(hidden=16), "cpu", np.random.default_rng(0))
    )
    result = compare(env, cognitive, _baseline(env), episodes=5, seed=0)
    assert set(result) == {"cognitive", "baseline", "delta"}
    assert result["delta"]["win_rate"] == (
        result["cognitive"]["win_rate"] - result["baseline"]["win_rate"]
    )
