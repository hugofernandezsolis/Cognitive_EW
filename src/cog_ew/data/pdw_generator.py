"""Generador sintético de trenes de pulsos PDW etiquetados (ELINT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cog_ew.data.pdw_library import INTRA_PULSE_MODS, ModeSpec


@dataclass
class PulseTrain:
    toa: NDArray[np.float64]
    rf: NDArray[np.float64]
    pw: NDArray[np.float64]
    pa: NDArray[np.float64]
    aoa: NDArray[np.float64]
    intra_pulse_mod: NDArray[np.int64]


def _generate_pri(
    pattern: str, pri_range: tuple[float, float], n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    lo, hi = pri_range
    if pattern == "fixed":
        return np.full(n, (lo + hi) / 2.0)
    if pattern == "jitter":
        return rng.uniform(lo, hi, n)
    if pattern == "stagger":
        levels = np.linspace(lo, hi, 3)
        return np.resize(levels, n).astype(np.float64)
    raise ValueError(f"Patrón de PRI desconocido: {pattern}")


def generate_pulse_train(
    mode_spec: ModeSpec,
    n_pulses: int,
    rng: np.random.Generator,
    *,
    noise_std: float = 0.0,
    drop_prob: float = 0.0,
    spurious_prob: float = 0.0,
) -> PulseTrain:
    pri = _generate_pri(mode_spec.pri_pattern, mode_spec.pri_range, n_pulses, rng)
    toa = np.cumsum(pri)

    band_lo, band_hi = mode_spec.rf_band
    if mode_spec.freq_hopping:
        rf = rng.uniform(band_lo, band_hi, n_pulses)
    else:
        rf = np.full(n_pulses, (band_lo + band_hi) / 2.0)

    pw = rng.uniform(mode_spec.pw_range[0], mode_spec.pw_range[1], n_pulses)

    scan_us = mode_spec.scan_period * 1e6
    pa = 0.5 * (1.0 + np.cos(2.0 * np.pi * (toa % scan_us) / scan_us))

    aoa = np.full(n_pulses, rng.uniform(-180.0, 180.0))

    mod_codes = np.array([INTRA_PULSE_MODS.index(m) for m in mode_spec.intra_pulse_mods])
    intra = rng.choice(mod_codes, n_pulses)

    if noise_std > 0.0:
        rf = rf + rng.normal(0.0, noise_std, n_pulses)
        pw = pw + rng.normal(0.0, noise_std, n_pulses)
        pa = np.clip(pa + rng.normal(0.0, noise_std, n_pulses), 0.0, 1.0)
        aoa = aoa + rng.normal(0.0, noise_std, n_pulses)

    if drop_prob > 0.0:
        keep = rng.random(n_pulses) >= drop_prob
        toa, rf, pw, pa, aoa, intra = (arr[keep] for arr in (toa, rf, pw, pa, aoa, intra))

    if spurious_prob > 0.0 and toa.shape[0] > 0:
        n_sp = int(rng.binomial(n_pulses, spurious_prob))
        if n_sp > 0:
            sp_toa = rng.uniform(toa.min(), toa.max(), n_sp)
            sp_rf = rng.uniform(band_lo, band_hi, n_sp)
            sp_pw = rng.uniform(mode_spec.pw_range[0], mode_spec.pw_range[1], n_sp)
            sp_pa = rng.random(n_sp)
            sp_aoa = rng.uniform(-180.0, 180.0, n_sp)
            sp_intra = rng.choice(mod_codes, n_sp)
            toa = np.concatenate([toa, sp_toa])
            rf = np.concatenate([rf, sp_rf])
            pw = np.concatenate([pw, sp_pw])
            pa = np.concatenate([pa, sp_pa])
            aoa = np.concatenate([aoa, sp_aoa])
            intra = np.concatenate([intra, sp_intra])
            order = np.argsort(toa)
            toa, rf, pw, pa, aoa, intra = (arr[order] for arr in (toa, rf, pw, pa, aoa, intra))

    return PulseTrain(
        toa=toa.astype(np.float64),
        rf=rf.astype(np.float64),
        pw=pw.astype(np.float64),
        pa=pa.astype(np.float64),
        aoa=aoa.astype(np.float64),
        intra_pulse_mod=intra.astype(np.int64),
    )
