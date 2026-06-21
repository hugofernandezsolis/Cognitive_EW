import numpy as np
import torch

from cog_ew.gan_signals.validity import diversity, structural_validity


def _valid_windows(n=8):
    cont = torch.rand(n, 5, 64)
    cat = torch.zeros(n, 5, 64)
    codes = torch.randint(0, 5, (n, 64))
    cat.scatter_(1, codes.unsqueeze(1), 1.0)
    return torch.cat([cont, cat], dim=1)


def test_structural_validity_perfect_on_valid_windows():
    out = structural_validity(_valid_windows())
    assert out["continuous_in_range_frac"] == 1.0
    assert out["categorical_onehot_frac"] == 1.0


def test_structural_validity_flags_out_of_range_continuous():
    w = _valid_windows()
    w[0, 0, 0] = 5.0
    assert structural_validity(w)["continuous_in_range_frac"] < 1.0


def test_diversity_detects_mode_collapse():
    one = _valid_windows(1)
    collapsed = one.repeat(10, 1, 1)
    type_ids = np.zeros(10, dtype=np.int64)
    out = diversity(collapsed, type_ids)
    assert out["mean_intersample_std"] == 0.0


def test_diversity_full_coverage_and_variety():
    w = _valid_windows(6)
    type_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = diversity(w, type_ids)
    assert out["mean_intersample_std"] > 0.0
    assert out["n_types"] == 3
    assert out["coverage"] == 1.0


def test_diversity_single_sample_does_not_crash():
    out = diversity(_valid_windows(1), np.array([0], dtype=np.int64))
    assert out["mean_intersample_std"] == 0.0
