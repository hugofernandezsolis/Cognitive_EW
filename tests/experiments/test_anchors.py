import math

from cog_ew.experiments.anchors import _TARGETS, AnchorResult, _passed


def test_targets_are_the_q1_anchors():
    assert _TARGETS == {"jamming": 0.92, "elint": 0.96, "marl": 0.45, "gan": 0.22}


def test_passed_requires_finite_and_ge_target():
    assert _passed(0.93, 0.92) is True
    assert _passed(0.92, 0.92) is True
    assert _passed(0.91, 0.92) is False
    assert _passed(math.inf, 0.45) is False
    assert _passed(math.nan, 0.22) is False


def test_anchor_result_is_frozen():
    r = AnchorResult("elint", 0.96, 0.5, None, False, "/tmp/run")
    assert r.name == "elint"
    assert r.baseline is None
