from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig
from cog_ew.temporal_cnn_elint.train import TrainConfig, train

CONFIG = "configs/temporal_cnn_elint/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)

    assert isinstance(config.data, PDWConfig)
    assert isinstance(config.model, TemporalCNNConfig)
    assert config.data.window == 64
    assert config.model.dilations == (1, 2, 4, 8)
    assert config.splits == (0.7, 0.15, 0.15)
    assert config.loss_weights == (1.0, 1.0)
    assert config.tracking is False


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


def test_train_is_deterministic(tmp_path):
    a = train(_tiny_config(tmp_path / "a"))
    b = train(_tiny_config(tmp_path / "b"))
    assert a["train_loss_history"] == b["train_loss_history"]
