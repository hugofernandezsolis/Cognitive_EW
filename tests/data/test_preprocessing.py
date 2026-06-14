import numpy as np

from cog_ew.data.preprocessing import (
    complex_to_iq,
    iq_to_complex,
    normalize_power,
    to_channels_first,
)


def test_normalize_power_unit_mean_power_single_example():
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(128, 2)).astype(np.float32) * 5.0

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1))
    assert np.isclose(power, 1.0, atol=1e-5)


def test_normalize_power_batched():
    rng = np.random.default_rng(1)
    iq = (
        rng.normal(size=(4, 128, 2)).astype(np.float32)
        * np.array([1.0, 2.0, 3.0, 4.0])[:, None, None]
    )

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1), axis=-1)
    assert np.allclose(power, 1.0, atol=1e-5)


def test_to_channels_first_single_example():
    iq = np.zeros((128, 2), dtype=np.float32)

    out = to_channels_first(iq)

    assert out.shape == (2, 128)


def test_to_channels_first_batched():
    iq = np.zeros((4, 128, 2), dtype=np.float32)

    out = to_channels_first(iq)

    assert out.shape == (4, 2, 128)


def test_iq_complex_roundtrip():
    rng = np.random.default_rng(2)
    iq = rng.normal(size=(128, 2)).astype(np.float32)

    out = complex_to_iq(iq_to_complex(iq))

    assert np.allclose(out, iq, atol=1e-6)


def test_normalize_power_zero_input_no_nan():
    iq = np.zeros((16, 2), dtype=np.float32)

    out = normalize_power(iq)

    assert not np.isnan(out).any()
    assert np.all(out == 0.0)
