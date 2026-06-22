"""Entrada CLI para correr las anclas Q1 (Fase 6) en Colab o local.

Uso:
    python notebooks/run_anchors.py --profile quick --anchors all --out-dir runs/anchors
    python notebooks/run_anchors.py --profile full --anchors elint,marl
    python notebooks/run_anchors.py --profile full --seeds 0,1,2,3,4 --out-dir runs/anchors_full_ms

Con --seeds se corre la campaña multi-semilla: cada semilla en su subdirectorio y un reporte
agregado (media, std, IC al 95 %, pass-rate) en anchors_multiseed_report.json.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from cog_ew.experiments.multiseed import run_anchors_multiseed
from cog_ew.experiments.report import ANCHOR_RUNNERS, ExperimentProfile, run_anchors

_PROFILES = {
    "quick": "configs/experiments/quick.yaml",
    "full": "configs/experiments/full.yaml",
}
_ORDER = ("jamming", "elint", "marl", "gan")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre las anclas Q1 del proyecto Cognitive EW.")
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--anchors", default="all", help="'all' o lista separada por comas")
    parser.add_argument("--out-dir", default="runs/anchors")
    parser.add_argument(
        "--seeds",
        default=None,
        help="lista de semillas separada por comas (p. ej. 0,1,2,3,4) para campaña multi-semilla",
    )
    return parser.parse_args(argv)


def resolve_anchors(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return _ORDER
    names = tuple(name.strip() for name in spec.split(",") if name.strip())
    unknown = [name for name in names if name not in ANCHOR_RUNNERS]
    if unknown:
        raise ValueError(f"anclas desconocidas: {unknown}")
    return names


def resolve_seeds(spec: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in spec.split(",") if s.strip())


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    profile = ExperimentProfile.from_yaml(_PROFILES[args.profile])
    names = resolve_anchors(args.anchors)
    if args.seeds is not None:
        seeds = resolve_seeds(args.seeds)
        report = run_anchors_multiseed(names, profile, seeds, args.out_dir)
    else:
        report = run_anchors(names, profile, args.out_dir)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
