"""Modelo de efectividad (J/S) y recompensa del entorno de jamming."""

from __future__ import annotations

from cog_ew.ew_library.library import JammingTechnique

_BAND_MISS_PENALTY = 1000.0


def jamming_effectiveness(
    technique: JammingTechnique,
    power_level: int,
    mode: str,
    band_match: bool,
    *,
    matrix: dict[str, dict[str, float]],
    base_js_db: tuple[float, ...],
    js_scale: float,
    burnthrough: float,
    eff_threshold: float,
) -> tuple[float, bool]:
    eff = matrix[technique.value][mode]
    j_s = base_js_db[power_level] + eff * js_scale
    if not band_match:
        return j_s - _BAND_MISS_PENALTY, False
    suppressed = eff >= eff_threshold and j_s >= burnthrough
    return j_s, suppressed
