import json

from cog_ew.experiments.report import ANCHOR_RUNNERS, ExperimentProfile, run_anchors

QUICK = "configs/experiments/quick.yaml"


def test_quick_profile_loads_from_yaml():
    profile = ExperimentProfile.from_yaml("configs/experiments/quick.yaml")
    assert profile.name == "quick"
    assert profile.device == "cpu"
    assert profile.jamming_total_steps is not None and profile.jamming_total_steps > 0
    assert profile.elint_epochs is not None
    assert profile.marl_compare_episodes > 0
    assert profile.jamming_config == "configs/deep_rl_jamming/train.yaml"


def test_full_profile_uses_null_for_yaml_durations():
    profile = ExperimentProfile.from_yaml("configs/experiments/full.yaml")
    assert profile.name == "full"
    assert profile.device == "cuda"
    assert profile.jamming_total_steps is None
    assert profile.elint_epochs is None
    assert profile.gan_total_steps is None
    assert profile.jamming_compare_episodes > 0


def test_anchor_runners_cover_all_four():
    assert set(ANCHOR_RUNNERS) == {"jamming", "elint", "marl", "gan"}


def test_run_anchors_single_writes_report(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    report = run_anchors(("elint",), profile, tmp_path)
    assert report["profile_name"] == "quick"
    assert report["seed"] == 0
    assert "config_hash" in report
    assert "torch" in report["dependencies"]
    elint = report["anchors"]["elint"]
    metrics = json.loads((tmp_path / "elint" / "metrics.json").read_text())
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
    assert elint["target"] == 0.96
    assert elint["achieved"] == expected
    assert elint["passed"] == (expected >= elint["target"])
    on_disk = json.loads((tmp_path / "anchors_report.json").read_text())
    assert on_disk == report


def test_run_anchors_elint_reproducible_by_seed(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    a = run_anchors(("elint",), profile, tmp_path / "a")
    b = run_anchors(("elint",), profile, tmp_path / "b")
    assert a["anchors"]["elint"]["achieved"] == b["anchors"]["elint"]["achieved"]


def test_run_anchors_all_aggregates_four(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    report = run_anchors(("jamming", "elint", "marl", "gan"), profile, tmp_path)
    assert set(report["anchors"]) == {"jamming", "elint", "marl", "gan"}
