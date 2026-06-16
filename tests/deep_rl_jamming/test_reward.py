from cog_ew.deep_rl_jamming.reward import compute_reward, jamming_effectiveness
from cog_ew.ew_library.library import JammingTechnique

MATRIX = {
    "noise": {"search": 0.8, "tws": 0.5, "track": 0.1, "missile_guidance": 0.1},
    "vgpo": {"search": 0.0, "tws": 0.3, "track": 0.8, "missile_guidance": 0.5},
    "none": {"search": 0.0, "tws": 0.0, "track": 0.0, "missile_guidance": 0.0},
}
KW = dict(
    matrix=MATRIX,
    base_js_db=(0.0, 10.0, 20.0, 30.0),
    js_scale=20.0,
    burnthrough=15.0,
    eff_threshold=0.5,
)


def test_effective_technique_against_mode_is_suppressed():
    j_s, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="search", band_match=True, **KW
    )
    assert suppressed is True
    assert j_s == 30.0 + 0.8 * 20.0


def test_wrong_technique_for_mode_not_suppressed():
    _, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="track", band_match=True, **KW
    )
    assert suppressed is False


def test_none_technique_never_suppresses():
    _, suppressed = jamming_effectiveness(
        JammingTechnique.NONE, power_level=3, mode="search", band_match=True, **KW
    )
    assert suppressed is False


def test_band_mismatch_kills_effectiveness():
    j_s, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="search", band_match=False, **KW
    )
    assert suppressed is False
    assert j_s < 0.0


RKW = dict(
    burnthrough=15.0,
    w_eff=1.0,
    lambda_power=0.5,
    n_power_levels=4,
    r_win=10.0,
    r_lose=10.0,
)


def test_suppressed_beats_not_suppressed():
    suppressed_r = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    failed_r = compute_reward(40.0, False, power_level=0, terminal=None, **RKW)
    assert suppressed_r > failed_r


def test_higher_power_is_penalised():
    low = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    high = compute_reward(40.0, True, power_level=3, terminal=None, **RKW)
    assert high < low


def test_terminal_win_adds_bonus():
    base = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    win = compute_reward(40.0, True, power_level=0, terminal="win", **RKW)
    assert win == base + 10.0


def test_terminal_lose_subtracts_penalty():
    base = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    lose = compute_reward(40.0, True, power_level=0, terminal="lose", **RKW)
    assert lose == base - 10.0
