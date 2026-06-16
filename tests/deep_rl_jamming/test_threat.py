from cog_ew.deep_rl_jamming.threat import RadarState, advance_threat

PARAMS = dict(lock_gain=0.15, lock_decay=0.15, n_eccm=3)


def test_not_suppressed_increases_lock_and_promotes_mode():
    state = RadarState()
    for _ in range(7):
        state = advance_threat(state, technique_idx=9, suppressed=False, n_modes=4, **PARAMS)
    assert state.lock_energy == 1.0
    assert state.mode_idx == 3


def test_suppressed_decays_lock():
    state = RadarState(mode_idx=2, lock_energy=0.6)
    state = advance_threat(state, technique_idx=0, suppressed=True, n_modes=4, **PARAMS)
    assert state.lock_energy < 0.6


def test_eccm_activates_after_n_consecutive_suppressed():
    state = RadarState()
    for _ in range(3):
        state = advance_threat(state, technique_idx=0, suppressed=True, n_modes=4, **PARAMS)
    assert state.eccm_active is True
    assert state.eccm_technique_idx == 0


def test_switching_technique_clears_eccm():
    state = RadarState(eccm_active=True, eccm_technique_idx=0)
    state = advance_threat(state, technique_idx=4, suppressed=True, n_modes=4, **PARAMS)
    assert state.eccm_active is False
    assert state.eccm_technique_idx == -1


def test_mode_idx_clamped_to_available_modes():
    state = RadarState(lock_energy=1.0)
    state = advance_threat(state, technique_idx=9, suppressed=False, n_modes=2, **PARAMS)
    assert state.mode_idx == 1
