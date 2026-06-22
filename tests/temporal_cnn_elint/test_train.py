import json

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig, TemporalCNNV2Config
from cog_ew.temporal_cnn_elint.train import TrainConfig, train

CONFIG = "configs/temporal_cnn_elint/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)

    assert isinstance(config.data, PDWConfig)
    assert isinstance(config.model, TemporalCNNConfig)
    assert config.architecture == "v1"
    assert config.data.window == 64
    assert config.model.dilations == (1, 2, 4, 8)
    assert config.splits == (0.7, 0.15, 0.15)
    assert config.loss_weights == (1.0, 1.0)
    assert config.tracking is False


def test_train_config_v2_yaml_parses():
    config = TrainConfig.from_yaml("configs/temporal_cnn_elint/train_v2.yaml")

    assert config.architecture == "v2"
    assert isinstance(config.model, TemporalCNNV2Config)
    assert config.data.feature_set == "v2"
    assert config.model.in_channels == 18
    assert len(config.loss_weights) == 3


def _tiny_config(out_dir):
    data = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        emitters=("SA-2", "LPI-FMCW"),
        modes=("search", "track"),
        window=64,
        n_pulses=192,
        n_trains=6,
        seed=0,
    )
    model = TemporalCNNConfig()
    return TrainConfig(
        data=data,
        model=model,
        splits=(0.6, 0.2, 0.2),
        batch_size=16,
        epochs=3,
        lr=1e-3,
        device="cpu",
        seed=0,
        out_dir=str(out_dir),
        tracking=False,
    )


def _tiny_v2_config(out_dir):
    data = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        emitters=("SA-2", "LPI-FMCW"),
        modes=("search", "track"),
        window=64,
        n_pulses=128,
        n_trains=4,
        seed=1,
        feature_set="v2",
    )
    model = TemporalCNNV2Config(hidden=16, dilations=(1,), dropout=0.0)
    return TrainConfig(
        data=data,
        model=model,
        architecture="v2",
        splits=(0.6, 0.2, 0.2),
        batch_size=8,
        epochs=1,
        lr=1e-3,
        device="cpu",
        seed=1,
        out_dir=str(out_dir),
        tracking=False,
        loss_weights=(1.0, 1.0, 1.0),
    )


def test_train_smoke_reduces_loss_and_writes_metrics(tmp_path):
    result = train(_tiny_config(tmp_path))

    history = result["train_loss_history"]
    assert history[-1] < history[0]
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    test_metrics = result["test"]
    assert "macro_acc_type" in test_metrics
    assert "macro_acc_mode" in test_metrics
    assert "lpi_accuracy" in test_metrics
    assert test_metrics["latency_mean_ms"] > 0.0


def test_train_v2_smoke_outputs_threat_metrics(tmp_path):
    result = train(_tiny_v2_config(tmp_path))

    test_metrics = result["test"]
    assert "macro_acc_threat" in test_metrics
    assert "confusion_threat" in test_metrics
    assert "strict_elint_score" in test_metrics
    assert test_metrics["macro_acc_type"] >= 0.0
    assert test_metrics["macro_acc_mode"] >= 0.0
    assert test_metrics["macro_acc_threat"] >= 0.0


def test_train_is_deterministic(tmp_path):
    a = train(_tiny_config(tmp_path / "a"))
    b = train(_tiny_config(tmp_path / "b"))
    assert a["train_loss_history"] == b["train_loss_history"]


def test_train_writes_reproducibility_metadata(tmp_path):
    train(_tiny_config(tmp_path))

    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["seed"] == 0
    assert meta["hyperparameters"]["batch_size"] == 16
    assert meta["hyperparameters"]["data"]["window"] == 64
    assert "torch" in meta["dependencies"]
    assert "numpy" in meta["dependencies"]
    assert "python" in meta["dependencies"]
    assert isinstance(meta["data_config_hash"], str)
    assert len(meta["data_config_hash"]) > 0


def test_train_includes_confusion_matrices(tmp_path):
    result = train(_tiny_config(tmp_path))

    test_metrics = result["test"]
    cm_type = test_metrics["confusion_type"]
    cm_mode = test_metrics["confusion_mode"]
    assert len(cm_type) == 8
    assert len(cm_type[0]) == 8
    assert len(cm_mode) == 4
    assert len(cm_mode[0]) == 4
