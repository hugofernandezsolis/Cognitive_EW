"""Muestreo y catálogo de tipos para el export de señales PDW sintéticas (Modelo 4, sub-pieza B)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticType:
    type_id: int
    source_a: int
    source_b: int
    alpha: float
    is_known: bool


def build_type_catalog(
    n_known: int, *, alphas: tuple[float, ...], extrapolate: bool = False
) -> list[SyntheticType]:
    types: list[SyntheticType] = []
    next_id = 0
    for i in range(n_known):
        types.append(SyntheticType(next_id, i, i, 0.0, True))
        next_id += 1
    novel_alphas = list(alphas) + ([-0.25, 1.25] if extrapolate else [])
    for a in range(n_known):
        for b in range(a + 1, n_known):
            for alpha in novel_alphas:
                types.append(SyntheticType(next_id, a, b, float(alpha), False))
                next_id += 1
    return types
