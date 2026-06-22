import json

import h5py
import numpy as np
import torch

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.data.synthetic_loader import SyntheticPDWDataset
from cog_ew.gan_signals.robustness import (
    RobustnessConfig,
    _classifier_loss,
    _fit_classifier,
    robustness_improvement_score,
    evaluate_type_accuracy,
    run_robustness_experiment,
)
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def _tiny_robustness_config(tmp_path) -> RobustnessConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    synth = tmp_path / "s.h5"
    n = 24
    np.random.seed(0)
    with h5py.File(synth, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(n, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.repeat([6, 7], n // 2).astype(np.int64))
        fh.create_dataset("is_known", data=np.ones(n, dtype=bool))
    pdw = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml", n_trains=2, n_pulses=256, window=64
    )
    return RobustnessConfig(
        synthetic_path=str(synth),
        held_out=("LPI-FMCW", "LPI-polyphase"),
        model=TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0),
        pdw=pdw,
        epochs=1,
        batch_size=16,
        seed=0,
        device="cpu",
        out_dir=str(tmp_path / "run"),
    )


def test_run_robustness_experiment_reports_delta(tmp_path):
    config = _tiny_robustness_config(tmp_path)
    result = run_robustness_experiment(config)
    assert set(result) == {"baseline", "augmented", "delta", "relative_improvement", "global"}
    assert result["delta"] == result["augmented"] - result["baseline"]
    assert result["relative_improvement"] == robustness_improvement_score(
        result["baseline"], result["augmented"]
    )
    assert set(result["global"]) == {"baseline", "augmented"}
    out = tmp_path / "run"
    assert (out / "run_meta.json").is_file()
    assert json.loads((out / "metrics.json").read_text())["delta"] == result["delta"]
    assert "latency_mean_ms" not in result
    disk = json.loads((out / "metrics.json").read_text())
    assert "latency_mean_ms" in disk


def test_robustness_improvement_score_uses_ratio_with_positive_baseline():
    assert robustness_improvement_score(0.5, 0.75) == 0.5


def test_robustness_improvement_score_uses_absolute_gain_with_zero_baseline():
    assert robustness_improvement_score(0.0, 1.0) == 1.0
    assert robustness_improvement_score(0.0, 0.0) == 0.0


def test_run_robustness_experiment_is_reproducible(tmp_path):
    r1 = run_robustness_experiment(_tiny_robustness_config(tmp_path / "a"))
    r2 = run_robustness_experiment(_tiny_robustness_config(tmp_path / "b"))
    assert r1["baseline"] == r2["baseline"]
    assert r1["augmented"] == r2["augmented"]


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


def test_evaluate_type_accuracy_moves_model_to_device(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(8, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.full(8, 2, dtype=np.int64))
        fh.create_dataset("is_known", data=np.ones(8, dtype=bool))
    ds = SyntheticPDWDataset(path)
    model = _tiny_model()
    evaluate_type_accuracy(model, ds, n_types=8, device="cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_fit_classifier_returns_trained_model(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(40, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.random.randint(0, 8, 40).astype(np.int64))
        fh.create_dataset("is_known", data=np.ones(40, dtype=bool))
    ds = SyntheticPDWDataset(path)

    model = _fit_classifier(
        TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0),
        ds,
        ds,
        epochs=2,
        batch_size=16,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )
    assert isinstance(model, TemporalCNN)
    acc = evaluate_type_accuracy(model, ds, n_types=8, device="cpu")
    assert 0.0 <= acc <= 1.0
