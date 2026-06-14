from cog_ew.data.pdw_library import (
    CONTINUOUS_FEATURES,
    CONTINUOUS_RANGES,
    INTRA_PULSE_MODS,
    MODES,
    THREAT_LEVELS,
    mode_to_threat,
)


def test_constant_shapes():
    assert CONTINUOUS_FEATURES == ("rf", "pw", "pa", "aoa", "pri")
    assert INTRA_PULSE_MODS == ("none", "lfm", "barker", "fmcw", "polyphase")
    assert MODES == ("search", "tws", "track", "missile_guidance")
    assert THREAT_LEVELS == ("low", "medium", "high", "critical")
    assert CONTINUOUS_RANGES.shape == (5, 2)


def test_mode_to_threat_mapping():
    assert mode_to_threat("search") == 0
    assert mode_to_threat("tws") == 1
    assert mode_to_threat("track") == 2
    assert mode_to_threat("missile_guidance") == 3
