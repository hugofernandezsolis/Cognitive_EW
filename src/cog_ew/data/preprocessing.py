"""Preprocesado de señales RF: IQ samples, PDW (Pulse Descriptor Words), normalización."""

import numpy as np
from numpy.typing import NDArray


def normalize_power(iq: NDArray[np.floating]) -> NDArray[np.float32]:
    iq = iq.astype(np.float32, copy=False)
    power: NDArray[np.float32] = np.mean(np.sum(iq**2, axis=-1), axis=-1)
    scale: NDArray[np.float32] = np.sqrt(power)
    # Evita NaN en frames de potencia nula (capturas en silencio, GAN inestable)
    scale = np.where(scale == 0.0, np.float32(1.0), scale)
    return (iq / scale[..., np.newaxis, np.newaxis]).astype(np.float32)


def to_channels_first(iq: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.swapaxes(iq, -1, -2)


def iq_to_complex(iq: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    return iq[..., 0] + 1j * iq[..., 1]


def complex_to_iq(z: NDArray[np.complexfloating]) -> NDArray[np.float32]:
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def toa_to_pri(toa: NDArray[np.floating]) -> NDArray[np.float32]:
    pri = np.empty_like(toa, dtype=np.float32)
    pri[1:] = np.diff(toa)
    pri[0] = pri[1] if pri.shape[0] > 1 else 0.0
    return pri


def normalize_pdw(cont: NDArray[np.floating], ranges: NDArray[np.floating]) -> NDArray[np.float32]:
    lo = ranges[:, 0]
    span = ranges[:, 1] - ranges[:, 0]
    out = (cont - lo) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def one_hot_intra_pulse(codes: NDArray[np.integer], k: int) -> NDArray[np.float32]:
    onehot: NDArray[np.float32] = np.eye(k, dtype=np.float32)[codes]
    return onehot


def window_sequence(seq: NDArray[np.floating], n: int) -> NDArray[np.float32]:
    n_windows = seq.shape[0] // n
    trimmed = seq[: n_windows * n]
    reshaped = trimmed.reshape(n_windows, n, seq.shape[1])
    return np.transpose(reshaped, (0, 2, 1)).astype(np.float32)
