import numpy as np

from cog_ew.deep_rl_jamming.compare import BaselinePolicy
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
