import dataclasses
import json
import math
from pathlib import Path

import pytest

from cog_ew.experiments.anchors import (
    _TARGETS,
    AnchorResult,
    _passed,
    run_elint_anchor,
    run_gan_anchor,
    run_jamming_anchor,
)
from cog_ew.experiments.report import ExperimentProfile

QUICK = "configs/experiments/quick.yaml"


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.name = "gan"  # type: ignore[misc]


def test_run_elint_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_elint_anchor(profile, tmp_path)
    metrics = json.loads((Path(result.run_dir) / "metrics.json").read_text())
    expected = (
        0.0
        if metrics["latency_p99_ms"] >= 1.0
        else min(
            metrics["macro_acc_type"],
            metrics["macro_acc_mode"],
            metrics["macro_acc_threat"],
            metrics["lpi_accuracy"],
        )
    )

    assert result.name == "elint"
    assert result.target == 0.96
    assert result.baseline is None
    assert 0.0 <= result.achieved <= 1.0
    assert result.achieved == expected
    assert result.passed == (expected >= result.target)
    assert Path(result.run_dir).exists()
    assert (Path(result.run_dir) / "metrics.json").exists()


def test_run_jamming_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_jamming_anchor(profile, tmp_path)
    assert result.name == "jamming"
    assert result.target == 0.92
    assert 0.0 <= result.achieved <= 1.0
    assert result.baseline is not None and 0.0 <= result.baseline <= 1.0
    assert (Path(result.run_dir) / "best.pt").exists()


def test_run_marl_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    from cog_ew.experiments.anchors import run_marl_anchor

    result = run_marl_anchor(profile, tmp_path)
    assert result.name == "marl"
    assert result.target == 0.45
    assert result.baseline is not None
    assert math.isfinite(result.achieved) or math.isinf(result.achieved)
    if math.isinf(result.achieved):
        assert result.passed is False
    assert (Path(result.run_dir) / "qmix" / "best.pt").exists()
    assert (Path(result.run_dir) / "iql" / "best.pt").exists()


def test_run_gan_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_gan_anchor(profile, tmp_path)
    assert result.name == "gan"
    assert result.target == 0.22
    assert result.baseline is not None
    assert math.isfinite(result.achieved)
    assert result.passed == (result.achieved >= result.target)
    assert (Path(result.run_dir) / "synthetic.h5").exists()
