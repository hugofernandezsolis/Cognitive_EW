"""Preprocesado de señales RF: IQ samples, PDW (Pulse Descriptor Words), normalización."""

import numpy as np
from numpy.typing import NDArray


def normalize_power(iq: NDArray[np.floating]) -> NDArray[np.float32]:
    iq = iq.astype(np.float32, copy=False)
    power: NDArray[np.float32] = np.mean(np.sum(iq**2, axis=-1), axis=-1)
    scale: NDArray[np.float32] = np.sqrt(power)
    return (iq / scale[..., np.newaxis, np.newaxis]).astype(np.float32)


def to_channels_first(iq: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.swapaxes(iq, -1, -2)


def iq_to_complex(iq: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    return iq[..., 0] + 1j * iq[..., 1]


def complex_to_iq(z: NDArray[np.complexfloating]) -> NDArray[np.float32]:
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)
