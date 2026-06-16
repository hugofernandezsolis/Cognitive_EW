import pytest

from cog_ew.ew_library.library import EWResponseLibrary, JammingTechnique

LIB = "configs/ew_library/responses.yaml"


def test_jamming_technique_has_expected_vocabulary():
    values = {t.value for t in JammingTechnique}
    assert values == {
        "noise",
        "drfm_repeater",
        "deception",
        "cross_eye",
        "vgpo",
        "rgpo",
        "chaff",
        "decoy",
        "evasive",
        "none",
    }


def test_from_yaml_loads_rules_and_defaults():
    lib = EWResponseLibrary.from_yaml(LIB)
    assert ("S-400", "missile_guidance") in lib.rules
    assert set(lib.defaults) == {"search", "tws", "track", "missile_guidance"}
    assert all(isinstance(t, JammingTechnique) for combo in lib.rules.values() for t in combo)


def test_from_yaml_rejects_unknown_technique(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "rules:\n"
        "  X:\n"
        "    search: [teleport]\n"
        "defaults:\n"
        "  search: [noise]\n"
        "  tws: [noise]\n"
        "  track: [noise]\n"
        "  missile_guidance: [noise]\n"
    )
    with pytest.raises(ValueError):
        EWResponseLibrary.from_yaml(bad)
