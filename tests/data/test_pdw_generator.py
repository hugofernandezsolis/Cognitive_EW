import numpy as np

from cog_ew.data.pdw_generator import _generate_pri, generate_pulse_train
from cog_ew.data.pdw_library import INTRA_PULSE_MODS, ModeSpec

CLEAN_MODE = ModeSpec(
    rf_band=(9.0, 10.0),
    pri_pattern="fixed",
    pri_range=(100.0, 100.0),
    pw_range=(1.0, 2.0),
    scan_period=1.0,
    freq_hopping=False,
    lpi=False,
    intra_pulse_mods=("lfm", "barker"),
)


def test_generate_pri_fixed_is_constant():
    pri = _generate_pri("fixed", (100.0, 300.0), 10, np.random.default_rng(0))
    assert np.allclose(pri, 200.0)


def test_generate_pri_jitter_within_range():
    pri = _generate_pri("jitter", (100.0, 300.0), 1000, np.random.default_rng(0))
    assert pri.min() >= 100.0
    assert pri.max() <= 300.0


def test_generate_pri_stagger_uses_three_levels():
    pri = _generate_pri("stagger", (100.0, 300.0), 9, np.random.default_rng(0))
    assert set(np.round(np.unique(pri), 6)) == {100.0, 200.0, 300.0}


def test_generate_pri_unknown_pattern_raises():
    import pytest

    with pytest.raises(ValueError):
        _generate_pri("nope", (1.0, 2.0), 4, np.random.default_rng(0))


def test_generate_pulse_train_clean_count_and_shapes():
    train = generate_pulse_train(CLEAN_MODE, 50, np.random.default_rng(1))
    assert train.toa.shape == (50,)
    assert train.rf.shape == (50,)
    assert train.intra_pulse_mod.shape == (50,)


def test_generate_pulse_train_deterministic():
    a = generate_pulse_train(CLEAN_MODE, 30, np.random.default_rng(7))
    b = generate_pulse_train(CLEAN_MODE, 30, np.random.default_rng(7))
    assert np.array_equal(a.toa, b.toa)
    assert np.array_equal(a.intra_pulse_mod, b.intra_pulse_mod)


def test_generate_pulse_train_mods_within_allowed_set():
    train = generate_pulse_train(CLEAN_MODE, 200, np.random.default_rng(2))
    allowed = {INTRA_PULSE_MODS.index(m) for m in CLEAN_MODE.intra_pulse_mods}
    assert set(np.unique(train.intra_pulse_mod)).issubset(allowed)


def test_generate_pulse_train_drop_reduces_count():
    train = generate_pulse_train(CLEAN_MODE, 200, np.random.default_rng(3), drop_prob=1.0)
    assert train.toa.shape[0] == 0


def test_generate_pulse_train_spurious_increases_count():
    train = generate_pulse_train(CLEAN_MODE, 200, np.random.default_rng(4), spurious_prob=0.5)
    assert train.toa.shape[0] > 200
    assert np.all(np.diff(train.toa) >= 0)  # ordenado por TOA
