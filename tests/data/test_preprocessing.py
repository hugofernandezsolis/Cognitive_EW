import numpy as np

from cog_ew.data.preprocessing import normalize_power


def test_normalize_power_unit_mean_power_single_example():
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(128, 2)).astype(np.float32) * 5.0

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1))
    assert np.isclose(power, 1.0, atol=1e-5)


def test_normalize_power_batched():
    rng = np.random.default_rng(1)
    iq = rng.normal(size=(4, 128, 2)).astype(np.float32) * np.array([1.0, 2.0, 3.0, 4.0])[:, None, None]

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1), axis=-1)
    assert np.allclose(power, 1.0, atol=1e-5)
