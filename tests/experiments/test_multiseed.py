import json
import math

import pytest

from cog_ew.experiments.multiseed import (
    aggregate_reports,
    run_anchors_multiseed,
    summary_stats,
)
from cog_ew.experiments.report import ExperimentProfile

QUICK = "configs/experiments/quick.yaml"


def test_summary_stats_basic_mean_std_n():
    s = summary_stats([1.0, 2.0, 3.0])
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.0)  # sample std, ddof=1
    assert s["n"] == 3
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]


def test_summary_stats_single_value_has_zero_spread():
    s = summary_stats([5.0])
    assert s["mean"] == pytest.approx(5.0)
    assert s["std"] == pytest.approx(0.0)
    assert s["n"] == 1
    assert s["ci95_low"] == pytest.approx(5.0)
    assert s["ci95_high"] == pytest.approx(5.0)


def test_summary_stats_identical_values_zero_ci_width():
    s = summary_stats([2.0, 2.0, 2.0])
    assert s["std"] == pytest.approx(0.0)
    assert s["ci95_low"] == pytest.approx(2.0)
    assert s["ci95_high"] == pytest.approx(2.0)


def test_summary_stats_empty_raises():
    with pytest.raises(ValueError):
        summary_stats([])


def test_summary_stats_ci_wider_than_std_error_is_finite():
    s = summary_stats([0.9, 1.0, 0.8, 1.0, 0.95])
    assert math.isfinite(s["ci95_low"]) and math.isfinite(s["ci95_high"])
    assert s["ci95_high"] - s["ci95_low"] > 0.0


def _report(seed, jam_ach, jam_base, jam_pass, elint_ach, elint_pass):
    return {
        "profile_name": "quick",
        "seed": seed,
        "dependencies": {"python": "3.12.0", "torch": "2.11.0", "numpy": "2.0.2"},
        "anchors": {
            "jamming": {
                "target": 0.92,
                "achieved": jam_ach,
                "baseline": jam_base,
                "passed": jam_pass,
                "metrics": {"latency_p99_ms": 0.3},
            },
            "elint": {
                "target": 0.96,
                "achieved": elint_ach,
                "baseline": None,
                "passed": elint_pass,
                "metrics": {"latency_p99_ms": 1.3},
            },
        },
    }


def test_aggregate_reports_mean_over_seeds():
    reports = [
        _report(0, 0.90, 0.70, True, 0.98, False),
        _report(1, 1.00, 0.70, True, 0.99, False),
    ]
    agg = aggregate_reports(reports)
    assert agg["n_seeds"] == 2
    assert agg["seeds"] == [0, 1]
    jam = agg["anchors"]["jamming"]
    assert jam["target"] == 0.92
    assert jam["achieved"]["mean"] == pytest.approx(0.95)
    assert jam["baseline"]["mean"] == pytest.approx(0.70)
    assert jam["pass_rate"] == pytest.approx(1.0)


def test_aggregate_reports_pass_rate_partial():
    reports = [
        _report(0, 0.90, 0.70, True, 0.98, False),
        _report(1, 0.80, 0.70, False, 0.99, True),
    ]
    agg = aggregate_reports(reports)
    assert agg["anchors"]["jamming"]["pass_rate"] == pytest.approx(0.5)
    assert agg["anchors"]["elint"]["pass_rate"] == pytest.approx(0.5)


def test_aggregate_reports_null_baseline_is_none():
    reports = [
        _report(0, 0.90, 0.70, True, 0.98, False),
        _report(1, 1.00, 0.70, True, 0.99, False),
    ]
    agg = aggregate_reports(reports)
    assert agg["anchors"]["elint"]["baseline"] is None


def test_aggregate_reports_aggregates_latency_metric():
    reports = [
        _report(0, 0.90, 0.70, True, 0.98, False),
        _report(1, 1.00, 0.70, True, 0.99, False),
    ]
    agg = aggregate_reports(reports)
    lat = agg["anchors"]["elint"]["metrics"]["latency_p99_ms"]
    assert lat["mean"] == pytest.approx(1.3)
    assert lat["n"] == 2


def test_aggregate_reports_requires_at_least_one():
    with pytest.raises(ValueError):
        aggregate_reports([])


def test_run_anchors_multiseed_writes_per_seed_and_aggregate(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    report = run_anchors_multiseed(("elint",), profile, seeds=(0, 1), out_dir=tmp_path)

    assert report["n_seeds"] == 2
    assert report["seeds"] == [0, 1]
    elint = report["anchors"]["elint"]
    assert "mean" in elint["achieved"]
    assert elint["achieved"]["n"] == 2
    assert "latency_p99_ms" in elint["metrics"]

    # per-seed subdirectories with their own single-seed reports
    assert (tmp_path / "seed_0" / "anchors_report.json").exists()
    assert (tmp_path / "seed_1" / "anchors_report.json").exists()
    # aggregate written at the top level
    on_disk = json.loads((tmp_path / "anchors_multiseed_report.json").read_text())
    assert on_disk == report


def test_run_anchors_multiseed_seeds_override_profile_seed(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    run_anchors_multiseed(("elint",), profile, seeds=(3,), out_dir=tmp_path)
    seed_report = json.loads((tmp_path / "seed_3" / "anchors_report.json").read_text())
    assert seed_report["seed"] == 3
