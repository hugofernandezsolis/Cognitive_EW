from cog_ew.ew_library.library import JammingTechnique


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
