"""Entrada CLI para correr las anclas Q1 (Fase 6) en Colab o local.

Uso:
    python notebooks/run_anchors.py --profile quick --anchors all --out-dir runs/anchors
    python notebooks/run_anchors.py --profile full --anchors elint,marl
"""

from __future__ import annotations

import argparse
import json
from typing import Any

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
    return parser.parse_args(argv)


def resolve_anchors(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return _ORDER
    names = tuple(name.strip() for name in spec.split(",") if name.strip())
    unknown = [name for name in names if name not in ANCHOR_RUNNERS]
    if unknown:
        raise ValueError(f"anclas desconocidas: {unknown}")
    return names


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    profile = ExperimentProfile.from_yaml(_PROFILES[args.profile])
    names = resolve_anchors(args.anchors)
    report = run_anchors(names, profile, args.out_dir)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
