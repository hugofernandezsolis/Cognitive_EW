import torch

from cog_ew.data.pdw_library import mode_to_threat
from cog_ew.temporal_cnn_elint.model import (
    TemporalCNN,
    TemporalCNNConfig,
    TemporalCNNV2,
    TemporalCNNV2Config,
)


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


def test_predict_shapes_and_dtypes():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(5, 10, 64)

    type_pred, mode_pred, threat_pred = model.predict(x)

    assert type_pred.shape == (5,)
    assert mode_pred.shape == (5,)
    assert threat_pred.shape == (5,)


def test_predict_threat_is_consistent_with_mode():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(16, 10, 64)

    _, mode_pred, threat_pred = model.predict(x)

    from cog_ew.data.pdw_library import MODES

    expected = torch.tensor([mode_to_threat(MODES[m]) for m in mode_pred.tolist()])
    assert torch.equal(threat_pred.cpu(), expected)


def test_v2_forward_output_shapes():
    model = TemporalCNNV2(TemporalCNNV2Config())
    x = torch.randn(4, 18, 64)

    type_logits, mode_logits, threat_logits = model(x)

    assert type_logits.shape == (4, 8)
    assert mode_logits.shape == (4, 4)
    assert threat_logits.shape == (4, 4)


def test_v2_predict_shapes_and_dtypes():
    model = TemporalCNNV2(TemporalCNNV2Config())
    x = torch.randn(5, 18, 64)

    type_pred, mode_pred, threat_pred = model.predict(x)

    assert type_pred.shape == (5,)
    assert mode_pred.shape == (5,)
    assert threat_pred.shape == (5,)
    assert type_pred.dtype == torch.long
    assert mode_pred.dtype == torch.long
    assert threat_pred.dtype == torch.long


def test_v2_param_count_stays_lightweight():
    model = TemporalCNNV2(TemporalCNNV2Config())
    n_params = sum(p.numel() for p in model.parameters())

    assert 40_000 < n_params < 250_000
