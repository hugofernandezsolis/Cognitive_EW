"""Generate LaTeX table bodies for the paper from the accepted multi-seed JSON artifacts.

Run from the repository root:

    uv run python scripts/paper/extract_results_tables.py

Reads ``results/anchors_full_ms_accepted/`` (the accepted N-seed campaign) and writes ``*.tex``
fragments into ``paper/tables/``. Headline anchor/latency statistics come from the aggregate
``anchors_multiseed_report.json``; the ELINT accuracy sub-metrics and the GAN robustness
sub-metrics (which the aggregate gates away) are re-aggregated across seeds here with the same
estimator used by the harness (``summary_stats``). Every number in the manuscript therefore maps
to a result file and reports mean over seeds with sample standard deviation.
"""

from __future__ import annotations

import json
from pathlib import Path

from cog_ew.experiments.multiseed import summary_stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "anchors_full_ms_accepted"
OUT = ROOT / "paper" / "tables"

REPORT = json.loads((RESULTS / "anchors_multiseed_report.json").read_text())
SEEDS = REPORT["seeds"]


def _seed_json(seed: int, rel: str) -> dict:
    return json.loads((RESULTS / f"seed_{seed}" / rel).read_text())


def _seed_metric(seed: int, rel: str, key: str) -> float:
    return float(_seed_json(seed, rel)[key])


def _across(rel: str, key: str) -> dict[str, float]:
    return summary_stats([_seed_metric(s, rel, key) for s in SEEDS])


def _w(name: str, rows: list[list[str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = " \\\\\n".join(" & ".join(r) for r in rows) + "\n"
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"wrote {OUT / name}")


def _ms(stat: dict[str, float], p: int = 3) -> str:
    return f"${stat['mean']:.{p}f} \\pm {stat['std']:.{p}f}$"


def _f(x: float, p: int = 2) -> str:
    return f"{x:.{p}f}"


def anchor_summary() -> None:
    a = REPORT["anchors"]
    # per-seed accuracy floor (min of type/mode/threat/lpi), independent of the latency gate
    acc_floor = summary_stats(
        [
            min(
                _seed_metric(s, "elint/metrics.json", "macro_acc_type"),
                _seed_metric(s, "elint/metrics.json", "macro_acc_mode"),
                _seed_metric(s, "elint/metrics.json", "macro_acc_threat"),
                _seed_metric(s, "elint/metrics.json", "lpi_accuracy"),
            )
            for s in SEEDS
        ]
    )
    rows = [
        [
            "M1 Adaptive jamming",
            "win rate vs.\\ library",
            _f(a["jamming"]["target"]),
            _ms(a["jamming"]["achieved"]),
            _ms(a["jamming"]["baseline"]),
            "5/5",
            r"\cmark",
        ],
        [
            "M2 ELINT (accuracy)",
            "min.\\ macro accuracy",
            _f(a["elint"]["target"]),
            _ms(acc_floor),
            "--",
            "5/5",
            r"\cmark",
        ],
        [
            "M2 ELINT (latency)",
            "$p99$ latency (ms)",
            r"$<1.0$",
            _ms(a["elint"]["metrics"]["latency_p99_ms"]),
            "--",
            "0/5",
            r"\xmark",
        ],
        [
            "M3 Formation MARL",
            "rel.\\ suppression impr.",
            _f(a["marl"]["target"]),
            _ms(a["marl"]["achieved"]),
            _ms(a["marl"]["baseline"]),
            "5/5",
            r"\cmark",
        ],
        [
            "M4 GAN substitution",
            "synthetic-substitution gain",
            _f(a["gan"]["target"]),
            _ms(a["gan"]["achieved"]),
            _ms(a["gan"]["baseline"]),
            "5/5",
            r"\cmark",
        ],
    ]
    _w("anchor_summary.tex", rows)


def elint_strict() -> None:
    e = REPORT["anchors"]["elint"]["metrics"]
    rows = [
        [
            "Emitter type (macro)",
            _ms(_across("elint/metrics.json", "macro_acc_type"), 4),
            r"$\ge 0.96$",
            r"\cmark",
        ],
        [
            "Operating mode (macro)",
            _ms(_across("elint/metrics.json", "macro_acc_mode"), 4),
            r"$\ge 0.96$",
            r"\cmark",
        ],
        [
            "Threat state (macro)",
            _ms(_across("elint/metrics.json", "macro_acc_threat"), 4),
            r"$\ge 0.96$",
            r"\cmark",
        ],
        [
            "LPI accuracy",
            _ms(_across("elint/metrics.json", "lpi_accuracy"), 4),
            r"$\ge 0.96$",
            r"\cmark",
        ],
        ["Latency mean (ms)", _ms(e["latency_mean_ms"]), r"$<1.0$", r"\xmark"],
        ["Latency $p99$ (ms)", _ms(e["latency_p99_ms"]), r"$<1.0$", r"\xmark"],
    ]
    _w("elint_strict.tex", rows)


def gan_validity() -> None:
    g0 = _seed_json(SEEDS[0], "gan/metrics.json")
    gm = "gan/metrics.json"
    rob = "gan/robustness/metrics.json"
    glob = [_seed_json(s, rob)["global"] for s in SEEDS]
    rows = [
        ["Synthetic windows exported", f"{int(g0['n_windows']):,}", r"$>200{,}000$"],
        ["Synthetic variants (8 base + interp.)", f"{int(g0['n_types'])}", r"$\ge 50$"],
        ["Continuous-in-range fraction", _ms(_across(gm, "continuous_in_range_frac")), "1.0"],
        ["One-hot categorical fraction", _ms(_across(gm, "categorical_onehot_frac")), "1.0"],
        ["Catalog coverage", _ms(_across(gm, "coverage")), "1.0"],
        ["Mean Wasserstein-1 / feature", _ms(_across(gm, "wasserstein1_mean"), 4), "lower better"],
        ["Held-out acc.\\ (baseline)", _ms(_across(rob, "baseline")), "--"],
        ["Held-out acc.\\ (augmented)", _ms(_across(rob, "augmented")), "--"],
        ["Global acc.\\ (baseline)", _ms(summary_stats([g["baseline"] for g in glob])), "--"],
        ["Global acc.\\ (augmented)", _ms(summary_stats([g["augmented"] for g in glob])), "--"],
    ]
    _w("gan_validity.tex", rows)


def reproducibility() -> None:
    dep = REPORT["dependencies"]
    run_meta = json.loads((RESULTS / f"seed_{SEEDS[0]}" / "elint/run_meta.json").read_text())
    chash = str(run_meta.get("config_hash") or run_meta.get("data_config_hash") or "")[:16]
    rows = [
        ["Profile", REPORT["profile_name"]],
        ["Seeds", f"{REPORT['n_seeds']} (" + ", ".join(str(s) for s in SEEDS) + ")"],
        ["ELINT data config hash", r"\texttt{" + chash + r"\ldots}"],
        ["Python", dep["python"]],
        ["PyTorch", dep["torch"].replace("_", r"\_")],
        ["NumPy", dep["numpy"]],
    ]
    _w("reproducibility.tex", rows)


def latency_table() -> None:
    a = REPORT["anchors"]
    rows = [
        [
            "M1 Adaptive jamming",
            _ms(a["jamming"]["metrics"]["latency_mean_ms"]),
            _ms(a["jamming"]["metrics"]["latency_p99_ms"]),
            r"$<5$",
            r"\cmark",
        ],
        [
            "M2 ELINT classifier",
            _ms(a["elint"]["metrics"]["latency_mean_ms"]),
            _ms(a["elint"]["metrics"]["latency_p99_ms"]),
            r"$<1$",
            r"\xmark",
        ],
        [
            "M3 Formation agent",
            _ms(a["marl"]["metrics"]["latency_mean_ms"]),
            _ms(a["marl"]["metrics"]["latency_p99_ms"]),
            "--",
            "--",
        ],
    ]
    _w("latency.tex", rows)


def main() -> None:
    anchor_summary()
    elint_strict()
    gan_validity()
    reproducibility()
    latency_table()


if __name__ == "__main__":
    main()
