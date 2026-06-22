"""Agregación multi-semilla de las anclas Q1: media, desviación estándar e IC al 95 %."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from cog_ew.experiments.report import ExperimentProfile


def summary_stats(values: list[float], confidence: float = 0.95) -> dict[str, float]:
    if not values:
        raise ValueError("summary_stats requiere al menos un valor")
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    margin = t_crit * sem
    return {
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "n": n,
    }


def _aggregate_anchor(name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    target = entries[0]["target"]
    achieved = summary_stats([float(e["achieved"]) for e in entries])

    baselines = [e["baseline"] for e in entries]
    baseline: dict[str, float] | None
    if all(b is not None for b in baselines):
        baseline = summary_stats([float(b) for b in baselines])
    else:
        baseline = None

    pass_rate = float(np.mean([1.0 if e["passed"] else 0.0 for e in entries]))

    metrics: dict[str, Any] = {}
    metric_keys = entries[0].get("metrics", {}).keys()
    for key in metric_keys:
        if all(key in e.get("metrics", {}) for e in entries):
            metrics[key] = summary_stats([float(e["metrics"][key]) for e in entries])

    return {
        "target": target,
        "achieved": achieved,
        "baseline": baseline,
        "pass_rate": pass_rate,
        "metrics": metrics,
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("aggregate_reports requiere al menos un reporte")
    seeds = [int(r["seed"]) for r in reports]
    anchor_names = list(reports[0]["anchors"].keys())
    anchors: dict[str, Any] = {}
    for name in anchor_names:
        entries = [r["anchors"][name] for r in reports if name in r["anchors"]]
        anchors[name] = _aggregate_anchor(name, entries)
    return {
        "profile_name": reports[0].get("profile_name"),
        "n_seeds": len(reports),
        "seeds": seeds,
        "dependencies": reports[0].get("dependencies"),
        "anchors": anchors,
    }


def run_anchors_multiseed(
    names: tuple[str, ...],
    profile: ExperimentProfile,
    seeds: Sequence[int],
    out_dir: str | Path,
) -> dict[str, Any]:
    from cog_ew.experiments.report import run_anchors

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        seed_profile = replace(profile, seed=int(seed))
        report = run_anchors(names, seed_profile, out_dir / f"seed_{seed}")
        reports.append(report)
    aggregate = aggregate_reports(reports)
    (out_dir / "anchors_multiseed_report.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate
