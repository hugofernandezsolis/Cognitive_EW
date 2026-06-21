import torch

from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding
from cog_ew.gan_signals.sampler import (
    SyntheticType,
    build_type_catalog,
    load_generator,
    resolve_embedding,
    sample_type,
)


def test_catalog_has_known_first_then_interpolated():
    catalog = build_type_catalog(8, alphas=(0.25, 0.5, 0.75), extrapolate=False)
    assert len(catalog) >= 50
    known = catalog[:8]
    assert all(t.is_known and t.alpha == 0.0 and t.source_a == t.source_b for t in known)
    assert [t.source_a for t in known] == list(range(8))
    novel = catalog[8:]
    assert all(not t.is_known and t.source_a < t.source_b for t in novel)
    assert all(t.alpha in (0.25, 0.5, 0.75) for t in novel)
    assert [t.type_id for t in catalog] == list(range(len(catalog)))


def test_catalog_is_deterministic():
    a = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    b = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    assert a == b


def test_catalog_extrapolate_adds_out_of_range_alphas():
    catalog = build_type_catalog(8, alphas=(0.5,), extrapolate=True)
    novel_alphas = {t.alpha for t in catalog if not t.is_known}
    assert -0.25 in novel_alphas and 1.25 in novel_alphas


def _save_ckpt(tmp_path):
    gen = PDWGenerator(z_dim=8, e_dim=4, channels=8)
    emb = TypeEmbedding(n_emitters=8, e_dim=4)
    path = tmp_path / "best.pt"
    torch.save({"generator": gen.state_dict(), "embedding": emb.state_dict(), "critic": {}}, path)
    return path


def test_load_generator_roundtrips(tmp_path):
    path = _save_ckpt(tmp_path)
    gen, emb = load_generator(path, z_dim=8, e_dim=4, channels=8, n_emitters=8, device="cpu")
    assert isinstance(gen, PDWGenerator) and isinstance(emb, TypeEmbedding)
    assert not gen.training and not emb.training


def test_resolve_embedding_known_and_novel():
    emb = TypeEmbedding(n_emitters=8, e_dim=4)
    known = resolve_embedding(emb, SyntheticType(0, 3, 3, 0.0, True))
    novel = resolve_embedding(emb, SyntheticType(1, 0, 1, 0.5, False))
    assert known.shape == (4,) and novel.shape == (4,)
    assert torch.allclose(known, emb.embedding.weight[3])


def test_sample_type_shape(tmp_path):
    path = _save_ckpt(tmp_path)
    gen, emb = load_generator(path, z_dim=8, e_dim=4, channels=8, n_emitters=8)
    out = sample_type(gen, emb, SyntheticType(0, 0, 1, 0.5, False), n=6)
    assert out.shape == (6, 10, 64)
