import torch

from cog_ew.gan_signals.generator import TypeEmbedding


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
