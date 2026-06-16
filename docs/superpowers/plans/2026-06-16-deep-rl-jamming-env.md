# Entorno del ciclo radar (Modelo 1, sub-pieza A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar `RadarJammingEnv`, un entorno Gymnasium del ciclo radar amenaza (POMDP con historial, acción discreta técnica×potencia, estado oculto con promoción de modo y ECCM, recompensa densa), testable sin RL.

**Architecture:** Tres módulos enfocados: `threat.py` (estado oculto del radar + transiciones, puro), `reward.py` (efectividad J/S y recompensa, puro) y `env.py` (`RadarEnvConfig` + `RadarJammingEnv` que orquesta spaces/reset/step). Parámetros en `configs/deep_rl_jamming/env.yaml`. El emisor objetivo se muestrea de la `EmitterLibrary` ya existente.

**Tech Stack:** Python 3.11, NumPy, Gymnasium (dep nueva), PyYAML, pytest, ruff, mypy. Herramientas vía `.venv/bin/<tool>`.

---

## File Structure

- Modify: `pyproject.toml` — añadir `gymnasium`.
- Create: `src/cog_ew/deep_rl_jamming/threat.py` — `RadarState` (dataclass) + `advance_threat` (pura).
- Create: `src/cog_ew/deep_rl_jamming/reward.py` — `jamming_effectiveness` + `compute_reward` (puras).
- Create: `src/cog_ew/deep_rl_jamming/env.py` — `RadarEnvConfig` (dataclass + `from_yaml`) + `RadarJammingEnv` (`gymnasium.Env`).
- Create: `configs/deep_rl_jamming/env.yaml` — parámetros del entorno.
- Test: `tests/deep_rl_jamming/test_threat.py`, `test_reward.py`, `test_env.py`.

Los stubs actuales `src/cog_ew/deep_rl_jamming/{env,agent,train}.py` contienen solo un docstring de módulo. `env.py` se sobrescribe con su contenido real (manteniendo el docstring); `agent.py` y `train.py` se dejan intactos (sub-pieza B). `tests/deep_rl_jamming/__init__.py` ya existe (vacío).

**Constantes de diseño (deterministas, fijadas para los tests):**
- Técnicas: las 10 de `JammingTechnique` (orden de declaración del Enum; `none` es la última, índice 9).
- Modos: `MODES = (search, tws, track, missile_guidance)`.
- `eff_threshold = 0.5`, `burnthrough = 15.0` dB, `js_scale = 20.0`, `power_levels = (0, 10, 20, 30)` dB (4 niveles; el valor es el J/S base).
- `lock_gain = 0.15`, `lock_decay = 0.15`, `n_eccm = 3`.

---

## Task 1: Añadir dependencia `gymnasium`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Añadir la dependencia**

En `pyproject.toml`, dentro de `dependencies = [ ... ]`, añadir la línea (orden alfabético, antes de `h5py`):

```toml
    "gymnasium>=1.0.0",
```

- [ ] **Step 2: Sincronizar el entorno**

Run: `.venv/bin/python -m pip install "gymnasium>=1.0.0"` (o `uv sync` si el proyecto usa uv lock).
Expected: instala gymnasium sin errores.

- [ ] **Step 3: Verificar import**

Run: `.venv/bin/python -c "import gymnasium; print(gymnasium.__version__)"`
Expected: imprime una versión ≥ 1.0.0.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock 2>/dev/null; git add pyproject.toml
git commit -m "build(deep-rl): añadir dependencia gymnasium"
```

---

## Task 2: `threat.py` — `RadarState` + `advance_threat`

**Files:**
- Create: `src/cog_ew/deep_rl_jamming/threat.py`
- Test: `tests/deep_rl_jamming/test_threat.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_threat.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_threat.py -v`
Expected: FAIL con `ModuleNotFoundError: cog_ew.deep_rl_jamming.threat`.

- [ ] **Step 3: Write minimal implementation**

Crear `src/cog_ew/deep_rl_jamming/threat.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_threat.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/threat.py tests/deep_rl_jamming/test_threat.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/threat.py tests/deep_rl_jamming/test_threat.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/threat.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/threat.py tests/deep_rl_jamming/test_threat.py
git commit -m "feat(deep-rl): RadarState + advance_threat (modo/lock/ECCM)"
```

---

## Task 3: `reward.py` — `jamming_effectiveness`

**Files:**
- Create: `src/cog_ew/deep_rl_jamming/reward.py`
- Test: `tests/deep_rl_jamming/test_reward.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_reward.py`:

```python
from cog_ew.deep_rl_jamming.reward import jamming_effectiveness
from cog_ew.ew_library.library import JammingTechnique

MATRIX = {
    "noise": {"search": 0.8, "tws": 0.5, "track": 0.1, "missile_guidance": 0.1},
    "vgpo": {"search": 0.0, "tws": 0.3, "track": 0.8, "missile_guidance": 0.5},
    "none": {"search": 0.0, "tws": 0.0, "track": 0.0, "missile_guidance": 0.0},
}
KW = dict(
    matrix=MATRIX,
    base_js_db=(0.0, 10.0, 20.0, 30.0),
    js_scale=20.0,
    burnthrough=15.0,
    eff_threshold=0.5,
)


def test_effective_technique_against_mode_is_suppressed():
    j_s, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="search", band_match=True, **KW
    )
    assert suppressed is True
    assert j_s == 30.0 + 0.8 * 20.0


def test_wrong_technique_for_mode_not_suppressed():
    _, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="track", band_match=True, **KW
    )
    assert suppressed is False


def test_none_technique_never_suppresses():
    _, suppressed = jamming_effectiveness(
        JammingTechnique.NONE, power_level=3, mode="search", band_match=True, **KW
    )
    assert suppressed is False


def test_band_mismatch_kills_effectiveness():
    j_s, suppressed = jamming_effectiveness(
        JammingTechnique.NOISE, power_level=3, mode="search", band_match=False, **KW
    )
    assert suppressed is False
    assert j_s < 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_reward.py -v`
Expected: FAIL con `ModuleNotFoundError: cog_ew.deep_rl_jamming.reward`.

- [ ] **Step 3: Write minimal implementation**

Crear `src/cog_ew/deep_rl_jamming/reward.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_reward.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/reward.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
git commit -m "feat(deep-rl): jamming_effectiveness (J/S vs burnthrough)"
```

---

## Task 4: `reward.py` — `compute_reward`

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/reward.py`
- Test: `tests/deep_rl_jamming/test_reward.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_reward.py` (añadir `compute_reward` al import existente de `reward`):

```python
from cog_ew.deep_rl_jamming.reward import compute_reward, jamming_effectiveness

RKW = dict(
    burnthrough=15.0,
    w_eff=1.0,
    lambda_power=0.5,
    n_power_levels=4,
    r_win=10.0,
    r_lose=10.0,
)


def test_suppressed_beats_not_suppressed():
    suppressed_r = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    failed_r = compute_reward(40.0, False, power_level=0, terminal=None, **RKW)
    assert suppressed_r > failed_r


def test_higher_power_is_penalised():
    low = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    high = compute_reward(40.0, True, power_level=3, terminal=None, **RKW)
    assert high < low


def test_terminal_win_adds_bonus():
    base = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    win = compute_reward(40.0, True, power_level=0, terminal="win", **RKW)
    assert win == base + 10.0


def test_terminal_lose_subtracts_penalty():
    base = compute_reward(40.0, True, power_level=0, terminal=None, **RKW)
    lose = compute_reward(40.0, True, power_level=0, terminal="lose", **RKW)
    assert lose == base - 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_reward.py -v`
Expected: FAIL con `cannot import name 'compute_reward'`.

- [ ] **Step 3: Write minimal implementation**

Añadir al final de `src/cog_ew/deep_rl_jamming/reward.py`:

```python
def compute_reward(
    j_s: float,
    suppressed: bool,
    power_level: int,
    *,
    burnthrough: float,
    w_eff: float,
    lambda_power: float,
    n_power_levels: int,
    r_win: float,
    r_lose: float,
    terminal: str | None,
) -> float:
    eff_term = w_eff * (j_s - burnthrough) if suppressed else -w_eff
    power_term = lambda_power * (power_level / (n_power_levels - 1)) if n_power_levels > 1 else 0.0
    reward = eff_term - power_term
    if terminal == "win":
        reward += r_win
    elif terminal == "lose":
        reward -= r_lose
    return float(reward)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_reward.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/reward.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/reward.py tests/deep_rl_jamming/test_reward.py
git commit -m "feat(deep-rl): compute_reward (densa: efectividad - potencia + terminal)"
```

---

## Task 5: `env.py` — `RadarEnvConfig` + `from_yaml` + `env.yaml`

**Files:**
- Create: `configs/deep_rl_jamming/env.yaml`
- Modify: `src/cog_ew/deep_rl_jamming/env.py`
- Test: `tests/deep_rl_jamming/test_env.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_env.py`:

```python
from cog_ew.deep_rl_jamming.env import RadarEnvConfig

CONFIG = "configs/deep_rl_jamming/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = RadarEnvConfig.from_yaml(CONFIG)
    assert config.history_k == 8
    assert config.horizon_t == 64
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert config.effectiveness["none"]["missile_guidance"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: FAIL con `cannot import name 'RadarEnvConfig'`.

- [ ] **Step 3: Crear `configs/deep_rl_jamming/env.yaml`**

```yaml
library_path: configs/temporal_cnn_elint/emitters.yaml
emitters: null
history_k: 8
horizon_t: 64
power_levels: [0.0, 10.0, 20.0, 30.0]
burnthrough: 15.0
eff_threshold: 0.5
js_scale: 20.0
lock_gain: 0.15
lock_decay: 0.15
n_eccm: 3
w_eff: 1.0
lambda_power: 0.5
r_win: 10.0
r_lose: 10.0
obs_noise_std: 0.0
seed: 0
effectiveness:
  noise:         {search: 0.8, tws: 0.5, track: 0.1, missile_guidance: 0.1}
  drfm_repeater: {search: 0.1, tws: 0.4, track: 0.7, missile_guidance: 0.4}
  deception:     {search: 0.5, tws: 0.7, track: 0.4, missile_guidance: 0.3}
  cross_eye:     {search: 0.0, tws: 0.2, track: 0.9, missile_guidance: 0.7}
  vgpo:          {search: 0.0, tws: 0.3, track: 0.8, missile_guidance: 0.5}
  rgpo:          {search: 0.0, tws: 0.3, track: 0.8, missile_guidance: 0.7}
  chaff:         {search: 0.1, tws: 0.3, track: 0.5, missile_guidance: 0.8}
  decoy:         {search: 0.4, tws: 0.5, track: 0.4, missile_guidance: 0.6}
  evasive:       {search: 0.2, tws: 0.2, track: 0.3, missile_guidance: 0.5}
  none:          {search: 0.0, tws: 0.0, track: 0.0, missile_guidance: 0.0}
```

- [ ] **Step 4: Write minimal implementation**

Sobrescribir `src/cog_ew/deep_rl_jamming/env.py` (mantener el docstring de módulo) con el `RadarEnvConfig` (el `RadarJammingEnv` se añade en la Task 6):

```python
"""Entorno RL que simula el ciclo radar amenaza (PRI, frecuencia, modos ECCM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RadarEnvConfig:
    library_path: str
    effectiveness: dict[str, dict[str, float]]
    emitters: tuple[str, ...] | None = None
    history_k: int = 8
    horizon_t: int = 64
    power_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    burnthrough: float = 15.0
    eff_threshold: float = 0.5
    js_scale: float = 20.0
    lock_gain: float = 0.15
    lock_decay: float = 0.15
    n_eccm: int = 3
    w_eff: float = 1.0
    lambda_power: float = 0.5
    r_win: float = 10.0
    r_lose: float = 10.0
    obs_noise_std: float = 0.0
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RadarEnvConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("emitters") is not None:
            raw["emitters"] = tuple(raw["emitters"])
        if "power_levels" in raw:
            raw["power_levels"] = tuple(float(p) for p in raw["power_levels"])
        return cls(**raw)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/env.py
```
Expected: sin errores.

- [ ] **Step 7: Commit**

```bash
git add configs/deep_rl_jamming/env.yaml src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
git commit -m "feat(deep-rl): RadarEnvConfig.from_yaml + env.yaml"
```

---

## Task 6: `env.py` — `RadarJammingEnv` (`__init__`, `reset`, spaces, observación)

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/env.py`
- Test: `tests/deep_rl_jamming/test_env.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_env.py` (añadir los imports `import numpy as np` y `from cog_ew.deep_rl_jamming.env import RadarJammingEnv`):

```python
import numpy as np

from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv


def _env():
    return RadarJammingEnv(RadarEnvConfig.from_yaml(CONFIG))


def test_reset_returns_obs_with_correct_shape():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 5)
    assert obs.dtype == np.float32
    assert info["outcome"] == "ongoing"
    assert info["real_mode"] == "search"


def test_action_and_observation_spaces():
    env = _env()
    assert env.action_space.n == 10 * 4
    assert env.observation_space.shape == (8, 5)


def test_reset_is_deterministic_by_seed():
    a, _ = _env().reset(seed=7)
    b, _ = _env().reset(seed=7)
    assert np.array_equal(a, b)


def test_encode_action_roundtrip():
    env = _env()
    from cog_ew.ew_library.library import JammingTechnique

    action = env.encode_action(JammingTechnique.NONE, 0)
    assert action == list(JammingTechnique).index(JammingTechnique.NONE) * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: FAIL con `cannot import name 'RadarJammingEnv'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/cog_ew/deep_rl_jamming/env.py`. Añadir los imports arriba (numpy, gymnasium, NDArray, Any, y los del proyecto):

```python
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from cog_ew.data.pdw_library import CONTINUOUS_RANGES, MODES, EmitterLibrary, EmitterSpec
from cog_ew.deep_rl_jamming.threat import RadarState
from cog_ew.ew_library.library import JammingTechnique

_SCAN_MAX = 15.0
_N_FEATURES = 5
```

Y añadir la clase debajo de `RadarEnvConfig`:

```python
class RadarJammingEnv(gym.Env[NDArray[np.float32], int]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: RadarEnvConfig) -> None:
        super().__init__()
        self.config = config
        library = EmitterLibrary.from_yaml(config.library_path)
        if config.emitters is not None:
            self._candidates = tuple(
                e for e in library.emitters if e.name in config.emitters
            )
        else:
            self._candidates = library.emitters
        self._techniques = list(JammingTechnique)
        self._n_power = len(config.power_levels)
        self.action_space = gym.spaces.Discrete(len(self._techniques) * self._n_power)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(config.history_k, _N_FEATURES), dtype=np.float32
        )
        self._emitter: EmitterSpec = self._candidates[0]
        self._ladder: tuple[str, ...] = ()
        self._state = RadarState()
        self._t = 0
        self._history = np.zeros((config.history_k, _N_FEATURES), dtype=np.float32)

    def encode_action(self, technique: JammingTechnique, power_level: int) -> int:
        return self._techniques.index(technique) * self._n_power + power_level

    def _decode_action(self, action: int) -> tuple[JammingTechnique, int]:
        technique_idx, power_level = divmod(int(action), self._n_power)
        return self._techniques[technique_idx], power_level

    def _emitted_features(self) -> NDArray[np.float32]:
        spec = self._emitter.modes[self._ladder[self._state.mode_idx]]
        rf = 0.5 * (spec.rf_band[0] + spec.rf_band[1])
        pri = 0.5 * (spec.pri_range[0] + spec.pri_range[1])
        pw = 0.5 * (spec.pw_range[0] + spec.pw_range[1])
        rf_n = _normalize(rf, CONTINUOUS_RANGES[0])
        pw_n = _normalize(pw, CONTINUOUS_RANGES[1])
        pri_n = _normalize(pri, CONTINUOUS_RANGES[4])
        scan_n = min(1.0, spec.scan_period / _SCAN_MAX)
        eccm = 1.0 if self._state.eccm_active else 0.0
        feat = np.array([rf_n, pri_n, pw_n, scan_n, eccm], dtype=np.float32)
        if self.config.obs_noise_std > 0.0:
            feat = feat + self.np_random.normal(0.0, self.config.obs_noise_std, _N_FEATURES)
        return np.clip(feat, 0.0, 1.0).astype(np.float32)

    def _push_obs(self) -> NDArray[np.float32]:
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = self._emitted_features()
        return self._history.copy()

    def _info(self, outcome: str, j_s: float) -> dict[str, Any]:
        return {
            "real_mode": self._ladder[self._state.mode_idx],
            "j_s": j_s,
            "eccm_active": self._state.eccm_active,
            "outcome": outcome,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._candidates)))
        self._emitter = self._candidates[idx]
        self._ladder = tuple(m for m in MODES if m in self._emitter.modes)
        self._state = RadarState()
        self._t = 0
        self._history = np.zeros((self.config.history_k, _N_FEATURES), dtype=np.float32)
        obs = self._push_obs()
        return obs, self._info("ongoing", 0.0)
```

Y añadir el helper `_normalize` arriba (después de `_N_FEATURES`):

```python
def _normalize(value: float, value_range: NDArray[np.float64]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))
```

Nota mypy: `gym.Env[NDArray[np.float32], int]` puede requerir que `reset`/`step` casen exactamente con las firmas de Gymnasium (que ya lo hacen aquí). Si mypy se queja del genérico de la clase base, el fix mínimo es un `# type: ignore[...]` con el código exacto en la línea de `class`; no añadir ignores especulativos.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/env.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
git commit -m "feat(deep-rl): RadarJammingEnv reset + spaces + observación POMDP"
```

---

## Task 7: `env.py` — `step` (integración: efectividad + amenaza + terminal)

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/env.py`
- Test: `tests/deep_rl_jamming/test_env.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_env.py` (añadir `from cog_ew.ew_library.library import JammingTechnique` al inicio del fichero si no está ya):

```python
from cog_ew.ew_library.library import JammingTechnique

OPTIMAL = {
    "search": [JammingTechnique.NOISE, JammingTechnique.DECEPTION],
    "tws": [JammingTechnique.DECEPTION, JammingTechnique.NOISE],
    "track": [JammingTechnique.CROSS_EYE, JammingTechnique.VGPO],
    "missile_guidance": [JammingTechnique.CHAFF, JammingTechnique.RGPO],
}


def test_step_returns_gym_tuple():
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8, 5)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["outcome"] in {"ongoing", "win", "lose"}


def test_passive_jammer_loses():
    env = _env()
    env.reset(seed=0)
    passive = env.encode_action(JammingTechnique.NONE, 0)
    outcome = "ongoing"
    for _ in range(env.config.horizon_t):
        _, _, terminated, truncated, info = env.step(passive)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome == "lose"


def test_optimal_jammer_wins():
    env = _env()
    _, info = env.reset(seed=0)
    last_technique = None
    terminated = truncated = False
    while not (terminated or truncated):
        ranked = OPTIMAL[info["real_mode"]]
        technique = ranked[1] if info["eccm_active"] and last_technique == ranked[0] else ranked[0]
        last_technique = technique
        action = env.encode_action(technique, env._n_power - 1)
        _, _, terminated, truncated, info = env.step(action)
    assert info["outcome"] == "win"


def test_rollout_is_deterministic_by_seed():
    def rollout() -> list[float]:
        env = _env()
        env.reset(seed=3)
        rewards = []
        for _ in range(10):
            _, r, term, trunc, _ = env.step(5)
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    assert rollout() == rollout()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: FAIL — `step` aún no implementado (`NotImplementedError` de `gym.Env` o `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Añadir el import de las funciones puras arriba en `env.py`:

```python
from cog_ew.deep_rl_jamming.reward import compute_reward, jamming_effectiveness
from cog_ew.deep_rl_jamming.threat import advance_threat
```

Y añadir el método `step` dentro de `RadarJammingEnv` (después de `reset`):

```python
    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        technique, power_level = self._decode_action(action)
        technique_idx = self._techniques.index(technique)
        mode = self._ladder[self._state.mode_idx]
        band_match = not (
            self._state.eccm_active and technique_idx == self._state.eccm_technique_idx
        )
        j_s, suppressed = jamming_effectiveness(
            technique,
            power_level,
            mode,
            band_match,
            matrix=self.config.effectiveness,
            base_js_db=self.config.power_levels,
            js_scale=self.config.js_scale,
            burnthrough=self.config.burnthrough,
            eff_threshold=self.config.eff_threshold,
        )
        self._state = advance_threat(
            self._state,
            technique_idx,
            suppressed,
            len(self._ladder),
            lock_gain=self.config.lock_gain,
            lock_decay=self.config.lock_decay,
            n_eccm=self.config.n_eccm,
        )
        self._t += 1
        launch = self._state.mode_idx == len(self._ladder) - 1 and self._state.lock_energy >= 1.0
        terminated = launch
        truncated = (not launch) and self._t >= self.config.horizon_t
        if launch:
            outcome = "lose"
        elif truncated:
            outcome = "win"
        else:
            outcome = "ongoing"
        terminal = outcome if outcome != "ongoing" else None
        reward = compute_reward(
            j_s,
            suppressed,
            power_level,
            burnthrough=self.config.burnthrough,
            w_eff=self.config.w_eff,
            lambda_power=self.config.lambda_power,
            n_power_levels=self._n_power,
            r_win=self.config.r_win,
            r_lose=self.config.r_lose,
            terminal=terminal,
        )
        obs = self._push_obs()
        return obs, reward, terminated, truncated, self._info(outcome, j_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: PASS (9 tests en total en el fichero).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/env.py
```
Expected: sin errores. (En el test, `env._n_power` accede a un atributo "privado"; es aceptable en tests. Si ruff marca algo de estilo en los tests, corregir sin cambiar la lógica.)

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/env.py tests/deep_rl_jamming/test_env.py
git commit -m "feat(deep-rl): RadarJammingEnv.step (efectividad + amenaza + terminal)"
```

---

## Task 8: Verificación final (lint, tipos, suite completa)

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: todos los tests pasan (los anteriores + los nuevos de `deep_rl_jamming`).

- [ ] **Step 2: Gym API check (opcional, recomendado)**

Run:
```bash
.venv/bin/python -c "from gymnasium.utils.env_checker import check_env; from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv; check_env(RadarJammingEnv(RadarEnvConfig.from_yaml('configs/deep_rl_jamming/env.yaml')).unwrapped)"
```
Expected: sin excepciones (o solo *warnings* informativos). Si `check_env` lanza un error de conformidad, corregirlo antes de continuar.

- [ ] **Step 3: Lint y formato**

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```
Expected: `All checks passed!` y todos los ficheros formateados.

- [ ] **Step 4: Tipos sobre `src/`**

Run: `.venv/bin/mypy src`
Expected: `Success: no issues found`.

- [ ] **Step 5: Commit (si hubo algún ajuste)**

```bash
git add -A && git commit -m "chore(deep-rl): verificación final del entorno radar" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **Interfaz Gymnasium (spaces, reset/step):** Tasks 6-7; `gym.Env`, `Discrete`, `Box`; `check_env` en Task 8. ✅
- **Observación parcial + historial (POMDP):** Task 6 `_push_obs`/`_emitted_features` apilan K pasos de los 5 parámetros emitidos; no exponen modo/lock. ✅
- **Acción discreta técnica×potencia:** Task 6 `action_space = Discrete(10*4)`, `encode_action`/`_decode_action`. ✅
- **Estado oculto (modo/lock/ECCM):** Task 2 `threat.advance_threat`; promoción/regresión + ECCM + clear al cambiar técnica. ✅
- **Efectividad J/S y recompensa densa:** Tasks 3-4 `jamming_effectiveness` + `compute_reward` (terminal win/lose). ✅
- **Emisor de `EmitterLibrary`:** Task 6 `__init__`/`reset` muestrean de la librería; `_ladder` = modos del emisor en orden de `MODES`. ✅
- **Determinismo por seed:** Tasks 6-7 (`super().reset(seed)`, `self.np_random`); tests de determinismo de reset y de rollout. ✅
- **Dinámica del ancla (win/lose):** Task 7 tests `test_passive_jammer_loses` y `test_optimal_jammer_wins` fijan derrota/victoria. ✅
- **Config en YAML versionado:** Task 5 `RadarEnvConfig.from_yaml` + `env.yaml`. ✅
- **Dependencia `gymnasium`:** Task 1. ✅
- **Fuera de alcance respetado:** sin agente/red/entrenamiento (B), sin comparación (C), sin multi-agente. ✅
- **Consistencia de tipos:** `RadarState` (Task 2) usado por `advance_threat` (Task 2) y `env.step` (Task 7); `jamming_effectiveness`/`compute_reward` firmas idénticas entre Tasks 3-4 y sus llamadas en Task 7 (`matrix`, `base_js_db=power_levels`, `n_power_levels`, `terminal`). ✅
