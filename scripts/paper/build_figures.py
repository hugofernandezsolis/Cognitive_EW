"""Generate the paper figures (PDF, grayscale-safe) from the accepted multi-seed artifacts.

Run from the repository root (matplotlib is not a project dependency, so pull it in on the fly):

    uv run --with matplotlib python scripts/paper/build_figures.py

Reads ``results/anchors_full_ms_accepted/`` and writes into ``paper/figs/``:

- ``anchor_results.pdf``     -- achieved (mean over seeds, std error bars) vs.\ target, M1/M3/M4.
- ``gan_robustness.pdf``     -- held-out and global classifier accuracy, baseline vs.\ augmented.
- ``m2_confusion_type.pdf``  -- emitter-type confusion summed over all seeds, row-normalised.
- ``baseline_comparison.pdf`` -- M1/M3 cognitive policy vs.\ conventional/independent baseline.

Bars show the mean over seeds; error bars are the sample standard deviation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "anchors_full_ms_accepted"
FIGS = ROOT / "paper" / "figs"

REPORT = json.loads((RESULTS / "anchors_multiseed_report.json").read_text())
SEEDS = REPORT["seeds"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)

EMITTERS = ["SA-2", "SA-6", "S-300", "S-400", "HQ-9", "AESA", "LPI-FMCW", "LPI-poly"]


def _seed_json(seed: int, rel: str) -> dict:
    return json.loads((RESULTS / f"seed_{seed}" / rel).read_text())


def _seed_vals(rel: str, key: str) -> list[float]:
    return [float(_seed_json(s, rel)[key]) for s in SEEDS]


def _save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"wrote {FIGS / name}")


def fig_anchor_results() -> None:
    a = REPORT["anchors"]
    keys = ["jamming", "marl", "gan"]
    names = ["M1\njamming", "M3\nMARL", "M4\nGAN"]
    mean = [a[k]["achieved"]["mean"] for k in keys]
    std = [a[k]["achieved"]["std"] for k in keys]
    target = [a[k]["target"] for k in keys]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, target, w, label="target", color="0.75", edgecolor="black", hatch="//")
    ax.bar(
        x + w / 2,
        mean,
        w,
        yerr=std,
        label="achieved (mean$\\pm$std)",
        color="0.35",
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("anchor metric")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Achieved vs. target over 5 seeds", fontsize=8)
    _save(fig, "anchor_results.pdf")


def fig_gan_robustness() -> None:
    held_base = _seed_vals("gan/robustness/metrics.json", "baseline")
    held_aug = _seed_vals("gan/robustness/metrics.json", "augmented")
    glob = [_seed_json(s, "gan/robustness/metrics.json")["global"] for s in SEEDS]
    glob_base = [g["baseline"] for g in glob]
    glob_aug = [g["augmented"] for g in glob]

    groups = ["Held-out\nemitters", "Global\ntest set"]
    base_mean = [np.mean(held_base), np.mean(glob_base)]
    base_std = [np.std(held_base, ddof=1), np.std(glob_base, ddof=1)]
    aug_mean = [np.mean(held_aug), np.mean(glob_aug)]
    aug_std = [np.std(held_aug, ddof=1), np.std(glob_aug, ddof=1)]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(
        x - w / 2,
        base_mean,
        w,
        yerr=base_std,
        label="baseline (no synth.)",
        color="0.75",
        edgecolor="black",
        hatch="//",
        capsize=3,
        error_kw={"elinewidth": 1.0},
    )
    ax.bar(
        x + w / 2,
        aug_mean,
        w,
        yerr=aug_std,
        label="augmented",
        color="0.35",
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("type accuracy")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("M4: synthetic augmentation robustness", fontsize=8)
    _save(fig, "gan_robustness.pdf")


def fig_confusion_type() -> None:
    cm = np.zeros((8, 8), dtype=float)
    for s in SEEDS:
        cm += np.array(_seed_json(s, "elint/metrics.json")["confusion_type"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMITTERS)))
    ax.set_yticks(range(len(EMITTERS)))
    ax.set_xticklabels(EMITTERS, rotation=90, fontsize=6)
    ax.set_yticklabels(EMITTERS, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(8):
        for j in range(8):
            if norm[i, j] >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("M2: emitter-type confusion (5 seeds, row-norm.)", fontsize=8)
    _save(fig, "m2_confusion_type.pdf")


def fig_baseline_comparison() -> None:
    a = REPORT["anchors"]
    jam_cog = _seed_vals("jamming/metrics.json", "win_rate")
    marl_cog = _seed_vals("marl/qmix/metrics.json", "suppressed_fraction")

    labels = ["M1 win rate", "M3 suppressed frac."]
    base_mean = [a["jamming"]["baseline"]["mean"], a["marl"]["baseline"]["mean"]]
    base_std = [a["jamming"]["baseline"]["std"], a["marl"]["baseline"]["std"]]
    cog_mean = [float(np.mean(jam_cog)), float(np.mean(marl_cog))]
    cog_std = [float(np.std(jam_cog, ddof=1)), float(np.std(marl_cog, ddof=1))]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(
        x - w / 2,
        base_mean,
        w,
        yerr=base_std,
        label="conventional / independent",
        color="0.75",
        edgecolor="black",
        hatch="//",
        capsize=3,
        error_kw={"elinewidth": 1.0},
    )
    ax.bar(
        x + w / 2,
        cog_mean,
        w,
        yerr=cog_std,
        label="cognitive / coordinated",
        color="0.35",
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_title("Cognitive policy vs. baseline (5 seeds)", fontsize=8)
    _save(fig, "baseline_comparison.pdf")


def main() -> None:
    fig_anchor_results()
    fig_gan_robustness()
    fig_confusion_type()
    fig_baseline_comparison()


if __name__ == "__main__":
    main()
