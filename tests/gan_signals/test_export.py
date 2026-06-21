import json
from dataclasses import replace

import h5py
import numpy as np
import torch

from cog_ew.gan_signals.export import ExportConfig, export_synthetic
from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


def _tiny_config(tmp_path) -> ExportConfig:
    gen = PDWGenerator(z_dim=8, e_dim=4, channels=8)
    emb = TypeEmbedding(n_emitters=4, e_dim=4)
    ckpt = tmp_path / "best.pt"
    torch.save({"generator": gen.state_dict(), "embedding": emb.state_dict(), "critic": {}}, ckpt)
    return ExportConfig(
        checkpoint=str(ckpt),
        z_dim=8,
        e_dim=4,
        channels=8,
        n_emitters=4,
        alphas=(0.5,),
        samples_per_type=5,
        out_path=str(tmp_path / "out" / "synth.h5"),
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        n_real_compare=20,
        seed=0,
        device="cpu",
    )


def test_export_config_from_yaml():
    config = ExportConfig.from_yaml("configs/gan_signals/export.yaml")
    assert config.n_emitters == 8
    assert isinstance(config.alphas, tuple)
    assert config.samples_per_type > 0
    assert config.out_path.endswith(".h5")


def test_export_writes_hdf5_and_metrics(tmp_path):
    config = _tiny_config(tmp_path)
    result = export_synthetic(config)
    out = tmp_path / "out"
    n_types = 4 + 6  # 4 known + C(4,2)=6 pairs * 1 alpha
    n = n_types * 5
    with h5py.File(config.out_path, "r") as fh:
        assert fh["X"].shape == (n, 10, 64)
        assert fh["type_id"].shape == (n,)
        assert fh["is_known"].dtype == np.bool_
        assert fh.attrs["n_types"] == n_types
        assert fh.attrs["seed"] == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert {
        "continuous_in_range_frac",
        "wasserstein1_mean",
        "mean_intersample_std",
        "n_windows",
    } <= set(metrics)
    assert (out / "run_meta.json").is_file()
    assert result["n_windows"] == n


def test_export_is_reproducible_by_seed(tmp_path):
    base = _tiny_config(tmp_path)
    c1 = replace(base, out_path=str(tmp_path / "r1" / "s.h5"))
    c2 = replace(base, out_path=str(tmp_path / "r2" / "s.h5"))
    export_synthetic(c1)
    export_synthetic(c2)
    with h5py.File(c1.out_path, "r") as f1, h5py.File(c2.out_path, "r") as f2:
        assert np.array_equal(f1["X"][:], f2["X"][:])
