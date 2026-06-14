"""Preprocesado de señales RF: IQ samples, PDW (Pulse Descriptor Words), normalización."""

import numpy as np
from numpy.typing import NDArray


def normalize_power(iq: NDArray[np.floating]) -> NDArray[np.float32]:
    iq = iq.astype(np.float32, copy=False)
    power = np.mean(np.sum(iq**2, axis=-1), axis=-1)
    scale = np.sqrt(power)
    return (iq / scale[..., np.newaxis, np.newaxis]).astype(np.float32)
