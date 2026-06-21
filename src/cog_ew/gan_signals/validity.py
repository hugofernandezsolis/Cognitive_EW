"""Métricas de validez y diversidad de señales PDW sintéticas (Modelo 4, sub-pieza B)."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def structural_validity(windows: torch.Tensor) -> dict[str, float]:
    """Check structural validity of PDW windows.

    Args:
        windows: Tensor of shape (N, 10, 64) with continuous channels 0-4 in [0,1]
                and categorical one-hot channels 5-9.

    Returns:
        Dict with keys:
        - continuous_in_range_frac: Fraction of continuous samples in [0, 1].
        - categorical_onehot_frac: Fraction of categorical channels that are valid one-hot.
    """
    cont = windows[:, :5]
    cat = windows[:, 5:]
    in_range = ((cont >= 0.0) & (cont <= 1.0)).float().mean().item()
    sums = cat.sum(dim=1)
    binary = ((cat == 0.0) | (cat == 1.0)).all(dim=1)
    onehot = (torch.isclose(sums, torch.ones_like(sums)) & binary).float().mean().item()
    return {"continuous_in_range_frac": in_range, "categorical_onehot_frac": onehot}


def diversity(windows: torch.Tensor, type_ids: NDArray[np.int64]) -> dict[str, float]:
    """Measure diversity in synthetic PDW windows.

    Args:
        windows: Tensor of shape (N, 10, 64).
        type_ids: Array of shape (N,) with type IDs for each window.

    Returns:
        Dict with keys:
        - mean_intersample_std: Average std across samples (mode-collapse proxy).
        - n_distinct_categorical_patterns: Number of unique categorical patterns.
        - n_types: Number of distinct type IDs.
        - coverage: Fraction of type ID range covered (n_types / (max_type_id + 1)).
    """
    mean_std = windows.std(dim=0).mean().item()
    codes = windows[:, 5:].argmax(dim=1)
    patterns = {tuple(row.tolist()) for row in codes}
    unique_types = {int(t) for t in type_ids.tolist()}
    n_types = len(unique_types)
    max_id = int(type_ids.max()) if type_ids.size > 0 else -1
    coverage = n_types / (max_id + 1) if max_id >= 0 else 0.0
    return {
        "mean_intersample_std": mean_std,
        "n_distinct_categorical_patterns": float(len(patterns)),
        "n_types": float(n_types),
        "coverage": coverage,
    }
