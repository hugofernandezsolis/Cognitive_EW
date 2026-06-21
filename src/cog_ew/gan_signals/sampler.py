"""Muestreo y catálogo de tipos para el export de señales PDW sintéticas (Modelo 4, sub-pieza B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


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


def load_generator(
    checkpoint: str | Path,
    *,
    z_dim: int,
    e_dim: int,
    channels: int,
    n_emitters: int,
    device: str = "cpu",
) -> tuple[PDWGenerator, TypeEmbedding]:
    dev = torch.device(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    generator = PDWGenerator(z_dim, e_dim, channels).to(dev)
    embedding = TypeEmbedding(n_emitters, e_dim).to(dev)
    generator.load_state_dict(state["generator"])
    embedding.load_state_dict(state["embedding"])
    generator.eval()
    embedding.eval()
    return generator, embedding


def resolve_embedding(embedding: TypeEmbedding, stype: SyntheticType) -> torch.Tensor:
    if stype.is_known:
        ids = torch.tensor(
            [stype.source_a], dtype=torch.long, device=embedding.embedding.weight.device
        )
        result: torch.Tensor = embedding(ids).squeeze(0)
        return result
    return embedding.interpolate(stype.source_a, stype.source_b, stype.alpha)


@torch.no_grad()
def sample_type(
    generator: PDWGenerator,
    embedding: TypeEmbedding,
    stype: SyntheticType,
    n: int,
    device: str = "cpu",
) -> torch.Tensor:
    dev = torch.device(device)
    e = resolve_embedding(embedding, stype).to(dev)
    e_batch = e.unsqueeze(0).expand(n, -1)
    return generator.sample(e_batch)
