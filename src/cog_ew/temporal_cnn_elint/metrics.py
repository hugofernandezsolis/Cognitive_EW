"""Métricas y perfilado de latencia para el clasificador ELINT."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def macro_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    recalls: list[float] = []
    for c in range(num_classes):
        mask = targets == c
        support = int(mask.sum().item())
        if support == 0:
            continue
        correct = int((preds[mask] == c).sum().item())
        recalls.append(correct / support)
    return sum(recalls) / len(recalls) if recalls else 0.0


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm: torch.Tensor = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets.tolist(), preds.tolist(), strict=True):
        cm[t, p] += 1
    return cm


def lpi_accuracy(
    type_preds: torch.Tensor,
    type_targets: torch.Tensor,
    lpi_indices: tuple[int, ...],
) -> float:
    lpi = torch.tensor(list(lpi_indices), dtype=type_targets.dtype)
    mask = torch.isin(type_targets, lpi)
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((type_preds[mask] == type_targets[mask]).float().mean().item())


def profile_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    n_warmup: int = 10,
    n_iter: int = 100,
    device: str = "cpu",
) -> tuple[float, float]:
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    sample = sample.to(dev)
    is_cuda = dev.type == "cuda"
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)
        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(n_iter):
            start = time.perf_counter()
            model(sample)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(np.percentile(arr, 99))
