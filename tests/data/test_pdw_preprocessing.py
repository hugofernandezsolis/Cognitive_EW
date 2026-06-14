import numpy as np

from cog_ew.data.preprocessing import (
    normalize_pdw,
    one_hot_intra_pulse,
    toa_to_pri,
    window_sequence,
)


def test_toa_to_pri_diffs_with_padding():
    toa = np.array([0.0, 10.0, 25.0, 45.0])

    pri = toa_to_pri(toa)

    assert pri.shape == (4,)
    assert np.allclose(pri, [10.0, 10.0, 15.0, 20.0])  # pri[0] == pri[1]


def test_normalize_pdw_maps_ranges_to_unit():
    ranges = np.array([[0.0, 10.0], [0.0, 100.0]], dtype=np.float64)
    cont = np.array([[0.0, 50.0], [10.0, 100.0]], dtype=np.float64)

    out = normalize_pdw(cont, ranges)

    assert np.allclose(out, [[0.0, 0.5], [1.0, 1.0]])


def test_normalize_pdw_clips_out_of_range():
    ranges = np.array([[0.0, 10.0]], dtype=np.float64)
    cont = np.array([[-5.0], [15.0]], dtype=np.float64)

    out = normalize_pdw(cont, ranges)

    assert np.allclose(out, [[0.0], [1.0]])


def test_one_hot_intra_pulse():
    codes = np.array([0, 2, 4])

    out = one_hot_intra_pulse(codes, 5)

    assert out.shape == (3, 5)
    assert np.array_equal(out, np.eye(5)[codes])


def test_window_sequence_channels_first_and_discards_tail():
    seq = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)  # 7 pulsos, 2 features

    out = window_sequence(seq, 3)

    assert out.shape == (2, 2, 3)  # W=2, C=2, n=3 (descarta el 7º pulso)
    assert np.allclose(out[0, :, 0], seq[0])
