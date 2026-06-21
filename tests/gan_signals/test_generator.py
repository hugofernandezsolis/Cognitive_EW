import torch

from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


def test_type_embedding_maps_ids_to_vectors():
    torch.manual_seed(0)
    emb = TypeEmbedding(n_emitters=8, e_dim=16)
    ids = torch.tensor([0, 3, 7], dtype=torch.long)
    out = emb(ids)
    assert out.shape == (3, 16)


def test_type_embedding_interpolates_linearly():
    torch.manual_seed(0)
    emb = TypeEmbedding(n_emitters=8, e_dim=16)
    mid = emb.interpolate(0, 1, alpha=0.5)
    expected = 0.5 * emb.embedding.weight[0] + 0.5 * emb.embedding.weight[1]
    assert mid.shape == (16,)
    assert torch.allclose(mid, expected)


def _gen() -> PDWGenerator:
    return PDWGenerator(z_dim=8, e_dim=4, channels=8)


def test_generator_output_shape():
    torch.manual_seed(0)
    gen = _gen()
    z = torch.randn(5, 8)
    e = torch.randn(5, 4)
    out = gen(z, e)
    assert out.shape == (5, 10, 64)


def test_generator_continuous_channels_in_unit_range():
    torch.manual_seed(0)
    gen = _gen()
    out = gen(torch.randn(5, 8), torch.randn(5, 4))
    cont = out[:, :5]
    assert torch.all(cont >= 0.0) and torch.all(cont <= 1.0)


def test_generator_categorical_channels_are_one_hot():
    torch.manual_seed(0)
    gen = _gen()
    out = gen(torch.randn(5, 8), torch.randn(5, 4))
    cat = out[:, 5:]
    sums = cat.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))
    assert torch.all((cat == 0.0) | (cat == 1.0))


def test_generator_is_deterministic_by_seed():
    z = torch.randn(3, 8)
    e = torch.randn(3, 4)
    torch.manual_seed(0)
    a = _gen()(z, e)
    torch.manual_seed(0)
    b = _gen()(z, e)
    assert torch.allclose(a, b)


def test_generator_conditioning_changes_output():
    torch.manual_seed(0)
    gen = _gen()
    z = torch.randn(4, 8)
    out_a = gen(z, torch.zeros(4, 4))
    out_b = gen(z, torch.ones(4, 4))
    assert not torch.allclose(out_a, out_b)


def test_generator_sample_shape():
    torch.manual_seed(0)
    gen = _gen()
    out = gen.sample(torch.randn(6, 4))
    assert out.shape == (6, 10, 64)
