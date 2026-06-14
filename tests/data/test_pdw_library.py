from pathlib import Path

from cog_ew.data.pdw_library import (
    CONTINUOUS_FEATURES,
    CONTINUOUS_RANGES,
    INTRA_PULSE_MODS,
    MODES,
    THREAT_LEVELS,
    EmitterLibrary,
    ModeSpec,
    mode_to_threat,
)

CONFIG = Path("configs/temporal_cnn_elint/emitters.yaml")


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


def test_library_from_yaml_loads_all_emitters():
    lib = EmitterLibrary.from_yaml(CONFIG)
    assert lib.emitter_names() == (
        "SA-2",
        "SA-6",
        "S-300",
        "S-400",
        "HQ-9",
        "AESA",
        "LPI-FMCW",
        "LPI-polyphase",
    )


def test_modespec_tuples_and_types():
    lib = EmitterLibrary.from_yaml(CONFIG)
    sa2 = lib.emitters[0]
    search = sa2.modes["search"]
    assert isinstance(search, ModeSpec)
    assert search.rf_band == (2.9, 3.1)
    assert search.intra_pulse_mods == ("none",)
    assert search.freq_hopping is False


def test_lpi_emitters_declare_lpi_modulation():
    lib = EmitterLibrary.from_yaml(CONFIG)
    for emitter in lib.emitters:
        for mode_spec in emitter.modes.values():
            if mode_spec.lpi:
                assert any(m in ("fmcw", "polyphase") for m in mode_spec.intra_pulse_mods)
