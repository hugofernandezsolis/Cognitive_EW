"""Export masivo de señales PDW sintéticas a HDF5 (Modelo 4, sub-pieza B)."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.gan_signals.sampler import build_type_catalog, load_generator, sample_type
from cog_ew.gan_signals.validity import (
    distributional_realism,
    diversity,
    structural_validity,
)


@dataclass
class ExportConfig:
    checkpoint: str
    z_dim: int = 64
    e_dim: int = 16
    channels: int = 64
    n_emitters: int = 8
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75)
    extrapolate: bool = False
    samples_per_type: int = 2500
    out_path: str = "data/synthetic/wgan_gp.h5"
    library_path: str = "configs/temporal_cnn_elint/emitters.yaml"
    n_real_compare: int = 4000
    seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExportConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("alphas") is not None:
            raw["alphas"] = tuple(raw["alphas"])
        return cls(**raw)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _real_windows(library_path: str, n: int) -> torch.Tensor:
    dataset = PDWSyntheticDataset(PDWConfig(library_path=library_path))
    count = min(n, len(dataset))
    return torch.stack([dataset[i][0] for i in range(count)])


def export_synthetic(config: ExportConfig) -> dict[str, Any]:
    """Export mass synthetic PDW windows to HDF5 with validity metrics and run metadata.

    Fills the HDF5 file per-type via slices (never accumulates all N windows in RAM).
    Validity metrics are computed on a bounded buffer capped near n_real_compare.

    Args:
        config: ExportConfig with all hyperparameters.

    Returns:
        Metrics dict (same content as metrics.json) including n_windows and n_types.
    """
    _set_seeds(config.seed)
    generator, embedding = load_generator(
        config.checkpoint,
        z_dim=config.z_dim,
        e_dim=config.e_dim,
        channels=config.channels,
        n_emitters=config.n_emitters,
        device=config.device,
    )
    catalog = build_type_catalog(
        config.n_emitters, alphas=config.alphas, extrapolate=config.extrapolate
    )
    n_types = len(catalog)
    spt = config.samples_per_type
    n = n_types * spt

    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute checkpoint hash once, reuse in both HDF5 attrs and run_meta.
    checkpoint_hash = _file_sha256(config.checkpoint)

    # Per-type budget for the validity buffer so total stays near n_real_compare.
    per_type_val = max(1, -(-config.n_real_compare // n_types))
    val_windows: list[torch.Tensor] = []
    val_type_ids: list[np.ndarray] = []

    with h5py.File(out_path, "w") as fh:
        x_ds = fh.create_dataset("X", shape=(n, 10, 64), dtype="float32")
        type_ds = fh.create_dataset("type_id", shape=(n,), dtype="int64")
        a_ds = fh.create_dataset("source_a", shape=(n,), dtype="int64")
        b_ds = fh.create_dataset("source_b", shape=(n,), dtype="int64")
        alpha_ds = fh.create_dataset("alpha", shape=(n,), dtype="float32")
        known_ds = fh.create_dataset("is_known", shape=(n,), dtype=bool)

        for idx, stype in enumerate(catalog):
            lo = idx * spt
            hi = lo + spt
            windows = sample_type(generator, embedding, stype, spt, config.device)
            x_ds[lo:hi] = windows.cpu().numpy()
            type_ds[lo:hi] = stype.type_id
            a_ds[lo:hi] = stype.source_a
            b_ds[lo:hi] = stype.source_b
            alpha_ds[lo:hi] = stype.alpha
            known_ds[lo:hi] = stype.is_known

            take = min(per_type_val, spt)
            val_windows.append(windows[:take].cpu())
            val_type_ids.append(np.full(take, stype.type_id, dtype=np.int64))

        fh.attrs["n_types"] = n_types
        fh.attrs["samples_per_type"] = spt
        fh.attrs["checkpoint_hash"] = checkpoint_hash
        fh.attrs["seed"] = config.seed

    val_gen = torch.cat(val_windows)
    val_ids = np.concatenate(val_type_ids)
    real = _real_windows(config.library_path, config.n_real_compare)

    metrics: dict[str, Any] = {
        **structural_validity(val_gen),
        **distributional_realism(val_gen, real),
        **diversity(val_gen, val_ids),
        "n_windows": n,
        "n_types": n_types,
    }
    (out_path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    run_meta: dict[str, Any] = {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "checkpoint_hash": checkpoint_hash,
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    (out_path.parent / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    return metrics
