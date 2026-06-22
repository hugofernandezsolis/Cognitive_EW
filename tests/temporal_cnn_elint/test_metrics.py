import torch

from cog_ew.temporal_cnn_elint.metrics import (
    confusion_matrix,
    lpi_accuracy,
    macro_accuracy,
    profile_latency,
    strict_elint_passed,
    strict_elint_score,
)
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def test_macro_accuracy_perfect():
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3])

    assert macro_accuracy(preds, targets, 4) == 1.0


def test_macro_accuracy_balances_classes():
    # clase 0 perfecta (2/2), clase 1 fallada (0/2) → macro = 0.5
    preds = torch.tensor([0, 0, 0, 0])
    targets = torch.tensor([0, 0, 1, 1])

    assert macro_accuracy(preds, targets, 2) == 0.5


def test_macro_accuracy_ignores_unsupported_classes():
    # sólo aparece la clase 0 (perfecta); las clases 1 y 2 no tienen soporte
    preds = torch.tensor([0, 0])
    targets = torch.tensor([0, 0])

    assert macro_accuracy(preds, targets, 3) == 1.0


def test_confusion_matrix_counts():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])

    cm = confusion_matrix(preds, targets, 3)

    assert cm.shape == (3, 3)
    assert cm[2, 1].item() == 1  # un real-2 predicho como 1
    assert cm[2, 2].item() == 1
    assert cm[0, 0].item() == 1


def test_lpi_accuracy_filters_to_lpi_classes():
    # tipos reales: [6, 7, 0, 1]; LPI = {6, 7}
    type_preds = torch.tensor([6, 0, 0, 1])
    type_targets = torch.tensor([6, 7, 0, 1])

    # sólo se evalúan los reales 6 y 7: 6→6 acierta, 7→0 falla → 0.5
    assert lpi_accuracy(type_preds, type_targets, (6, 7)) == 0.5


def test_lpi_accuracy_no_lpi_samples_returns_zero():
    type_preds = torch.tensor([0, 1])
    type_targets = torch.tensor([0, 1])

    assert lpi_accuracy(type_preds, type_targets, (6, 7)) == 0.0


def test_profile_latency_returns_positive_mean_and_p99():
    model = TemporalCNN(TemporalCNNConfig())
    sample = torch.randn(1, 10, 64)

    mean_ms, p99_ms = profile_latency(model, sample, n_warmup=2, n_iter=10, device="cpu")

    assert mean_ms > 0.0
    assert p99_ms > 0.0
    assert p99_ms >= mean_ms


def test_strict_elint_score_uses_worst_required_metric():
    metrics = {
        "macro_acc_type": 1.0,
        "macro_acc_mode": 0.97,
        "macro_acc_threat": 0.98,
        "lpi_accuracy": 0.99,
        "latency_p99_ms": 0.7,
    }

    assert strict_elint_score(metrics) == 0.97


def test_strict_elint_score_zero_when_latency_fails():
    metrics = {
        "macro_acc_type": 1.0,
        "macro_acc_mode": 1.0,
        "macro_acc_threat": 1.0,
        "lpi_accuracy": 1.0,
        "latency_p99_ms": 1.01,
    }

    assert strict_elint_score(metrics) == 0.0


def test_strict_elint_passed_requires_all_metrics_and_latency():
    passing = {
        "macro_acc_type": 0.96,
        "macro_acc_mode": 0.97,
        "macro_acc_threat": 0.98,
        "lpi_accuracy": 0.99,
        "latency_p99_ms": 0.99,
    }
    failing = dict(passing, macro_acc_mode=0.95)

    assert strict_elint_passed(passing, target=0.96, latency_p99_ms=1.0)
    assert not strict_elint_passed(failing, target=0.96, latency_p99_ms=1.0)
