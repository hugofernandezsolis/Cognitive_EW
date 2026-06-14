import numpy as np

from cog_ew.data.pdw_generator import _generate_pri


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
