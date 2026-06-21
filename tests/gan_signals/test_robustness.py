import torch

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.data.synthetic_loader import SyntheticPDWDataset
from cog_ew.gan_signals.robustness import RobustnessConfig, _classifier_loss, evaluate_type_accuracy
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def test_robustness_config_from_yaml():
    config = RobustnessConfig.from_yaml("configs/gan_signals/robustness.yaml")
    assert isinstance(config.model, TemporalCNNConfig)
    assert isinstance(config.pdw, PDWConfig)
    assert config.held_out == ("LPI-FMCW", "LPI-polyphase")
    assert config.augment_held_out_only is True
    assert config.synthetic_path.endswith(".h5")


def _tiny_model() -> TemporalCNN:
    return TemporalCNN(TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0))


def test_synthetic_only_batch_does_not_touch_mode_head():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.rand(4, 10, 64)
    y_type = torch.tensor([6, 7, 6, 7])
    y_mode = torch.full((4,), -1)
    type_logits, mode_logits = model(x)
    loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head_mode.weight.grad is None


def test_mixed_batch_loss_is_finite_and_trains_mode_head():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.rand(4, 10, 64)
    y_type = torch.tensor([0, 1, 2, 3])
    y_mode = torch.tensor([0, -1, 2, -1])
    type_logits, mode_logits = model(x)
    loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head_mode.weight.grad is not None


def test_evaluate_type_accuracy_in_unit_range(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(12, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.full(12, 3, dtype=np.int64))
        fh.create_dataset("is_known", data=np.ones(12, dtype=bool))
    ds = SyntheticPDWDataset(path)
    acc = evaluate_type_accuracy(_tiny_model(), ds, n_types=8, device="cpu")
    assert 0.0 <= acc <= 1.0
