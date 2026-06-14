"""Métricas y perfilado de latencia para el clasificador ELINT."""

from __future__ import annotations

import torch


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
