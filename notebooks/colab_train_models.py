"""Colab runner for the trainable Cognitive EW models.

Usage in Colab, after cloning the repo:

    !pip install -q uv
    %cd /content/cog_ew
    !uv sync --frozen
    !uv run python notebooks/colab_train_models.py --model all --drive

Use --quick for a short smoke run before spending GPU time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _mount_drive() -> Path | None:
    if not _is_colab():
        print("Google Colab no detectado; no se monta Drive.")
        return None

    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")
    root = Path("/content/drive/MyDrive/cog_ew_runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _copy_config(config_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / config_path.name)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def train_temporal_cnn(
    *,
    device: str,
    run_root: Path,
    quick: bool,
) -> dict[str, Any]:
    from cog_ew.temporal_cnn_elint.train import TrainConfig, train

    config_path = Path("configs/temporal_cnn_elint/train.yaml")
    config = TrainConfig.from_yaml(config_path)
    out_dir = run_root / "temporal_cnn_elint" / _timestamp()
    config.device = device
    config.out_dir = str(out_dir)

    if quick:
        config.epochs = 2
        config.batch_size = 32
        config.data = replace(config.data, n_trains=2, n_pulses=128)

    _copy_config(config_path, out_dir)
    print(f"[Temporal CNN] device={config.device} out_dir={config.out_dir}")
    result = train(config)
    _write_summary(out_dir / "result.json", result)
    return {"model": "temporal_cnn_elint", "out_dir": str(out_dir), "result": result}


def train_deep_rl_jamming(
    *,
    device: str,
    run_root: Path,
    quick: bool,
) -> dict[str, Any]:
    from cog_ew.deep_rl_jamming.train import TrainConfig, train

    config_path = Path("configs/deep_rl_jamming/train.yaml")
    config = TrainConfig.from_yaml(config_path)
    out_dir = run_root / "deep_rl_jamming" / _timestamp()
    config.device = device
    config.out_dir = str(out_dir)

    if quick:
        config.total_steps = 800
        config.eval_every = 400
        config.eval_episodes = 5
        config.agent = replace(
            config.agent,
            buffer_size=2_000,
            batch_size=32,
            learning_starts=64,
            epsilon_decay_steps=400,
            target_sync=50,
        )

    _copy_config(config_path, out_dir)
    print(f"[Deep RL Jamming] device={config.device} out_dir={config.out_dir}")
    result = train(config)
    _write_summary(out_dir / "result.json", result)
    return {"model": "deep_rl_jamming", "out_dir": str(out_dir), "result": result}


def train_marl_formation(
    *,
    device: str,
    run_root: Path,
    quick: bool,
) -> dict[str, Any]:
    from cog_ew.marl_formation.train import TrainConfig, train

    config_path = Path("configs/marl_formation/qmix.yaml")
    config = TrainConfig.from_yaml(config_path)
    out_dir = run_root / "marl_formation" / _timestamp()
    config.device = device
    config.out_dir = str(out_dir)

    if quick:
        config.env = replace(config.env, horizon_t=8)
        config.total_episodes = 6
        config.eval_every = 3
        config.eval_episodes = 3
        config.agent = replace(
            config.agent,
            hidden=16,
            mixer_embed_dim=8,
            hypernet_hidden=16,
            buffer_episodes=8,
            batch_episodes=2,
            learning_starts_episodes=2,
            epsilon_decay_steps=4,
            target_sync=2,
        )

    _copy_config(config_path, out_dir)
    print(f"[MARL Formation QMIX] device={config.device} out_dir={config.out_dir}")
    result = train(config)
    _write_summary(out_dir / "result.json", result)
    return {"model": "marl_formation", "out_dir": str(out_dir), "result": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cognitive EW models on Google Colab.")
    parser.add_argument(
        "--model",
        choices=("temporal_cnn", "deep_rl_jamming", "marl_formation", "all"),
        default="all",
        help="Model to train.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cuda', or 'cpu'. Defaults to auto.",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Mount Google Drive and write runs under MyDrive/cog_ew_runs.",
    )
    parser.add_argument(
        "--run-root",
        default="runs/colab",
        help="Output root when --drive is not used.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke training pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    drive_root = _mount_drive() if args.drive else None
    run_root = drive_root if drive_root is not None else Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    if args.model in {"temporal_cnn", "all"}:
        results.append(train_temporal_cnn(device=device, run_root=run_root, quick=args.quick))
    if args.model in {"deep_rl_jamming", "all"}:
        results.append(train_deep_rl_jamming(device=device, run_root=run_root, quick=args.quick))
    if args.model in {"marl_formation", "all"}:
        results.append(train_marl_formation(device=device, run_root=run_root, quick=args.quick))

    summary_path = run_root / f"summary_{_timestamp()}.json"
    _write_summary(summary_path, {"device": device, "quick": args.quick, "runs": results})
    print(f"Resumen escrito en: {summary_path}")


if __name__ == "__main__":
    main()
