import torch

from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def test_forward_output_shapes():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(4, 10, 64)

    type_logits, mode_logits = model(x)

    assert type_logits.shape == (4, 8)
    assert mode_logits.shape == (4, 4)


def test_forward_deterministic_in_eval():
    torch.manual_seed(0)
    model = TemporalCNN(TemporalCNNConfig())
    model.eval()
    x = torch.randn(2, 10, 64)

    a = model(x)[0]
    b = model(x)[0]

    assert torch.allclose(a, b)


def test_param_count_is_lightweight():
    model = TemporalCNN(TemporalCNNConfig())
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 300_000
