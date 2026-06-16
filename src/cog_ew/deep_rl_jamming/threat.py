"""Estado oculto del radar amenaza y sus transiciones (promoción de modo, ECCM)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RadarState:
    mode_idx: int = 0
    lock_energy: float = 0.0
    eccm_active: bool = False
    eccm_technique_idx: int = -1
    effective_streak: int = 0


def advance_threat(
    state: RadarState,
    technique_idx: int,
    suppressed: bool,
    n_modes: int,
    *,
    lock_gain: float,
    lock_decay: float,
    n_eccm: int,
) -> RadarState:
    if suppressed:
        lock_energy = max(0.0, state.lock_energy - lock_decay)
        effective_streak = state.effective_streak + 1
    else:
        lock_energy = min(1.0, state.lock_energy + lock_gain)
        effective_streak = 0

    mode_idx = min(n_modes - 1, int(lock_energy * n_modes))

    eccm_active = state.eccm_active
    eccm_technique_idx = state.eccm_technique_idx
    if state.eccm_active and technique_idx != state.eccm_technique_idx:
        eccm_active = False
        eccm_technique_idx = -1
    elif not state.eccm_active and effective_streak >= n_eccm:
        eccm_active = True
        eccm_technique_idx = technique_idx
        effective_streak = 0

    return replace(
        state,
        mode_idx=mode_idx,
        lock_energy=lock_energy,
        eccm_active=eccm_active,
        eccm_technique_idx=eccm_technique_idx,
        effective_streak=effective_streak,
    )
