import torch

from cog_ew.temporal_cnn_elint.metrics import confusion_matrix, macro_accuracy


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
