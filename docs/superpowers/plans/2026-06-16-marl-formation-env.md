# Entorno IADS multi-agente (Modelo 3, sub-pieza A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar `IADSFormationEnv`, un entorno multi-agente CTDE (N jammers vs M radares) con acción discreta compuesta, lock abstracto por radar y recompensa de equipo, testable sin RL.

**Architecture:** Interfaz CTDE propia (`reset`/`step` devuelven obs local por agente + estado global). Reutiliza `threat.advance_threat` y `reward.jamming_effectiveness` del Modelo 1, `JammingTechnique` y `EmitterLibrary`. Acción por agente `Discrete(M×3×P)` = (objetivo, tipo, potencia); resolución mejor-de-los-que-apuntan por radar; recompensa de equipo compartida.

**Tech Stack:** Python 3.11, NumPy, PyYAML, pytest, ruff, mypy. Herramientas vía `.venv/bin/<tool>`.

---

## File Structure

- Create: `src/cog_ew/marl_formation/env.py` — `IADSEnvConfig` (dataclass + `from_yaml`) e `IADSFormationEnv`.
- Create: `configs/marl_formation/env.yaml` — parámetros del entorno.
- Test: `tests/marl_formation/test_env.py`.

El stub `src/cog_ew/marl_formation/env.py` contiene solo un docstring; se sobrescribe manteniéndolo.
`tests/marl_formation/__init__.py` ya existe (vacío).

**Constantes de diseño (deterministas):** técnicas de las 10 de `JammingTechnique`; `jam_type ∈ {0:none, 1:deception, 2:suppression}`; `MODES = (search, tws, track, missile_guidance)`; features por radar = 5 (`rf, pri, pw, scan, eccm`).

**Firmas reutilizadas (no modificar):**
- `advance_threat(state, technique_idx, suppressed, n_modes, *, lock_gain, lock_decay, n_eccm) -> RadarState`; `RadarState(mode_idx, lock_energy, eccm_active, eccm_technique_idx, effective_streak)`.
- `jamming_effectiveness(technique, power_level, mode, band_match, *, matrix, base_js_db, js_scale, burnthrough, eff_threshold) -> tuple[float, bool]`.

---

## Task 1: `IADSEnvConfig` + `env.yaml`

**Files:**
- Create: `configs/marl_formation/env.yaml`
- Modify: `src/cog_ew/marl_formation/env.py`
- Test: `tests/marl_formation/test_env.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/marl_formation/test_env.py`:

```python
from cog_ew.marl_formation.env import IADSEnvConfig

CONFIG = "configs/marl_formation/env.yaml"


def test_config_from_yaml_loads_parameters():
    config = IADSEnvConfig.from_yaml(CONFIG)
    assert config.n_agents == 4
    assert config.n_radars == 4
    assert config.power_levels == (0.0, 10.0, 20.0, 30.0)
    assert config.effectiveness["noise"]["search"] == 0.8
    assert "noise" in config.suppression_techniques
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: FAIL con `cannot import name 'IADSEnvConfig'`.

- [ ] **Step 3: Crear `configs/marl_formation/env.yaml`**

```yaml
library_path: configs/temporal_cnn_elint/emitters.yaml
emitters: [S-300, S-400]
n_agents: 4
n_radars: 4
power_levels: [0.0, 10.0, 20.0, 30.0]
burnthrough: 15.0
eff_threshold: 0.5
js_scale: 20.0
lock_gain: 0.15
lock_decay: 0.15
n_eccm: 3
w_lock: 1.0
lambda_power: 0.5
w_supp: 1.0
r_win: 10.0
r_lose: 10.0
horizon_t: 64
seed: 0
suppression_techniques: [noise, drfm_repeater, vgpo, rgpo, cross_eye, chaff]
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

Sobrescribir `src/cog_ew/marl_formation/env.py` (mantener el docstring) con el `IADSEnvConfig` (el env se añade en Tasks 2-3):

```python
"""Entorno multi-agente que simula el IADS adversario y la formación de aeronaves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IADSEnvConfig:
    library_path: str
    effectiveness: dict[str, dict[str, float]]
    suppression_techniques: tuple[str, ...] = (
        "noise",
        "drfm_repeater",
        "vgpo",
        "rgpo",
        "cross_eye",
        "chaff",
    )
    emitters: tuple[str, ...] | None = None
    n_agents: int = 4
    n_radars: int = 4
    power_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    burnthrough: float = 15.0
    eff_threshold: float = 0.5
    js_scale: float = 20.0
    lock_gain: float = 0.15
    lock_decay: float = 0.15
    n_eccm: int = 3
    w_lock: float = 1.0
    lambda_power: float = 0.5
    w_supp: float = 1.0
    r_win: float = 10.0
    r_lose: float = 10.0
    horizon_t: int = 64
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> IADSEnvConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("emitters") is not None:
            raw["emitters"] = tuple(raw["emitters"])
        if "power_levels" in raw:
            raw["power_levels"] = tuple(float(p) for p in raw["power_levels"])
        if "suppression_techniques" in raw:
            raw["suppression_techniques"] = tuple(raw["suppression_techniques"])
        return cls(**raw)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/ruff check src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/mypy src/cog_ew/marl_formation/env.py
```
Expected: sin errores.

- [ ] **Step 7: Commit**

```bash
git add configs/marl_formation/env.yaml src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
git commit -m "feat(marl): IADSEnvConfig.from_yaml + env.yaml"
```

---

## Task 2: `IADSFormationEnv` — `__init__`, `reset`, observación/estado, codificación

**Files:**
- Modify: `src/cog_ew/marl_formation/env.py`
- Test: `tests/marl_formation/test_env.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/marl_formation/test_env.py` (añadir `import numpy as np` arriba y `IADSFormationEnv` al import):

```python
import numpy as np

from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv


def _env():
    return IADSFormationEnv(IADSEnvConfig.from_yaml(CONFIG))


def test_reset_returns_obs_state_info():
    env = _env()
    obs, state, info = env.reset(seed=0)
    assert set(obs) == set(range(4))
    assert obs[0].shape == (4 * 5 + 4,)
    assert obs[0].dtype == np.float32
    assert state.shape == (4 * (4 + 2) + 4,)
    assert info["outcome"] == "ongoing"


def test_reset_is_deterministic_by_seed():
    a, sa, _ = _env().reset(seed=3)
    b, sb, _ = _env().reset(seed=3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(sa, sb)


def test_encode_action_roundtrip():
    env = _env()
    action = env.encode_action(target=2, jam_type=1, power_level=3)
    assert env._decode_action(action) == (2, 1, 3)
    assert env.action_dim == 4 * 3 * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: FAIL con `cannot import name 'IADSFormationEnv'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/cog_ew/marl_formation/env.py` (añadir los imports arriba):

```python
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cog_ew.data.pdw_library import CONTINUOUS_RANGES, MODES, EmitterLibrary, EmitterSpec
from cog_ew.deep_rl_jamming.reward import jamming_effectiveness
from cog_ew.deep_rl_jamming.threat import RadarState, advance_threat
from cog_ew.ew_library.library import JammingTechnique

_SCAN_MAX = 15.0
_N_RADAR_FEATURES = 5
_N_MODES = len(MODES)
```

Y añadir el helper `_normalize` (antes de la clase de config) y la clase `IADSFormationEnv` debajo de `IADSEnvConfig`:

```python
def _normalize(value: float, value_range: NDArray[np.float64]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


class IADSFormationEnv:
    def __init__(self, config: IADSEnvConfig) -> None:
        self.config = config
        library = EmitterLibrary.from_yaml(config.library_path)
        if config.emitters is not None:
            self._candidates = tuple(e for e in library.emitters if e.name in config.emitters)
        else:
            self._candidates = library.emitters
        self._techniques = list(JammingTechnique)
        self._none_idx = self._techniques.index(JammingTechnique.NONE)
        self._n_power = len(config.power_levels)
        self.n_agents = config.n_agents
        self.n_radars = config.n_radars
        self.action_dim = config.n_radars * 3 * self._n_power
        self.obs_dim = config.n_radars * _N_RADAR_FEATURES + config.n_agents
        self.state_dim = config.n_radars * (_N_MODES + 2) + config.n_agents
        self._rng = np.random.default_rng(config.seed)
        self._emitters: list[EmitterSpec] = []
        self._ladders: list[tuple[str, ...]] = []
        self._states: list[RadarState] = []
        self._last_actions = [0] * config.n_agents
        self._t = 0

    def encode_action(self, target: int, jam_type: int, power_level: int) -> int:
        return target * (3 * self._n_power) + jam_type * self._n_power + power_level

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        power_level = action % self._n_power
        jam_type = (action // self._n_power) % 3
        target = action // (3 * self._n_power)
        return target, jam_type, power_level

    def _radar_features(self, idx: int) -> NDArray[np.float32]:
        spec = self._emitters[idx].modes[self._ladders[idx][self._states[idx].mode_idx]]
        rf = 0.5 * (spec.rf_band[0] + spec.rf_band[1])
        pri = 0.5 * (spec.pri_range[0] + spec.pri_range[1])
        pw = 0.5 * (spec.pw_range[0] + spec.pw_range[1])
        eccm = 1.0 if self._states[idx].eccm_active else 0.0
        return np.array(
            [
                _normalize(rf, CONTINUOUS_RANGES[0]),
                _normalize(pri, CONTINUOUS_RANGES[4]),
                _normalize(pw, CONTINUOUS_RANGES[1]),
                min(1.0, spec.scan_period / _SCAN_MAX),
                eccm,
            ],
            dtype=np.float32,
        )

    def _obs(self) -> dict[int, NDArray[np.float32]]:
        radar_feats = np.concatenate([self._radar_features(i) for i in range(self.n_radars)])
        obs: dict[int, NDArray[np.float32]] = {}
        for a in range(self.n_agents):
            agent_onehot = np.zeros(self.n_agents, dtype=np.float32)
            agent_onehot[a] = 1.0
            obs[a] = np.concatenate([radar_feats, agent_onehot]).astype(np.float32)
        return obs

    def _global_state(self) -> NDArray[np.float32]:
        parts: list[NDArray[np.float32]] = []
        for i in range(self.n_radars):
            state = self._states[i]
            mode_oh = np.zeros(_N_MODES, dtype=np.float32)
            mode_oh[MODES.index(self._ladders[i][state.mode_idx])] = 1.0
            parts.append(mode_oh)
            parts.append(
                np.array(
                    [state.lock_energy, 1.0 if state.eccm_active else 0.0], dtype=np.float32
                )
            )
        last = np.array(
            [a / (self.action_dim - 1) for a in self._last_actions], dtype=np.float32
        )
        parts.append(last)
        return np.concatenate(parts).astype(np.float32)

    def _info(self, outcome: str, suppressed_count: int) -> dict[str, Any]:
        return {"outcome": outcome, "suppressed_fraction": suppressed_count / self.n_radars}

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[int, NDArray[np.float32]], NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idxs = self._rng.integers(0, len(self._candidates), size=self.n_radars)
        self._emitters = [self._candidates[int(i)] for i in idxs]
        self._ladders = [tuple(m for m in MODES if m in e.modes) for e in self._emitters]
        self._states = [RadarState() for _ in range(self.n_radars)]
        self._last_actions = [0] * self.n_agents
        self._t = 0
        return self._obs(), self._global_state(), self._info("ongoing", 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/ruff check src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/mypy src/cog_ew/marl_formation/env.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
git commit -m "feat(marl): IADSFormationEnv reset + obs/estado CTDE + codificación"
```

---

## Task 3: `IADSFormationEnv.step` (resolución por radar + recompensa de equipo)

**Files:**
- Modify: `src/cog_ew/marl_formation/env.py`
- Test: `tests/marl_formation/test_env.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/marl_formation/test_env.py` (añadir `from dataclasses import replace` arriba):

```python
from dataclasses import replace


def _small_env():
    config = replace(IADSEnvConfig.from_yaml(CONFIG), n_agents=2, n_radars=2)
    return IADSFormationEnv(config)


def _all_actions(env, target, jam_type, power_level):
    action = env.encode_action(target, jam_type, power_level)
    return {a: action for a in range(env.n_agents)}


def test_step_returns_ctde_tuple_with_shared_reward():
    env = _small_env()
    env.reset(seed=0)
    obs, state, rewards, terminated, truncated, info = env.step(_all_actions(env, 0, 2, 3))
    assert set(obs) == {0, 1}
    assert obs[0].shape == (2 * 5 + 2,)
    assert len(set(rewards.values())) == 1
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 0.0 <= info["suppressed_fraction"] <= 1.0


def test_splitting_targets_covers_more_than_concentrating():
    split_env = _small_env()
    split_env.reset(seed=0)
    split_actions = {
        0: split_env.encode_action(0, 2, 3),
        1: split_env.encode_action(1, 2, 3),
    }
    _, _, _, _, _, split_info = split_env.step(split_actions)

    conc_env = _small_env()
    conc_env.reset(seed=0)
    _, _, _, _, _, conc_info = conc_env.step(_all_actions(conc_env, 0, 2, 3))

    assert split_info["suppressed_fraction"] >= conc_info["suppressed_fraction"]


def test_uncovered_radars_lead_to_loss():
    env = _small_env()
    env.reset(seed=0)
    passive = _all_actions(env, 0, 0, 0)  # jam_type none
    outcome = "ongoing"
    for _ in range(env.config.horizon_t):
        _, _, _, terminated, truncated, info = env.step(passive)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome == "lose"


def test_rollout_is_deterministic_by_seed():
    def rollout():
        env = _small_env()
        env.reset(seed=1)
        rewards = []
        for _ in range(10):
            _, _, r, term, trunc, _ = env.step(_all_actions(env, 0, 2, 2))
            rewards.append(r[0])
            if term or trunc:
                break
        return rewards

    assert rollout() == rollout()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: FAIL con `AttributeError: 'IADSFormationEnv' object has no attribute 'step'`.

- [ ] **Step 3: Write minimal implementation**

Añadir el método `_technique_for` y `step` dentro de `IADSFormationEnv` (después de `reset`):

```python
    def _technique_for(self, jam_type: int, mode: str) -> JammingTechnique:
        if jam_type == 1:
            return JammingTechnique.DECEPTION
        best = max(
            self.config.suppression_techniques,
            key=lambda t: self.config.effectiveness[t][mode],
        )
        return JammingTechnique(best)

    def step(
        self, actions: dict[int, int]
    ) -> tuple[
        dict[int, NDArray[np.float32]],
        NDArray[np.float32],
        dict[int, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        decoded = {a: self._decode_action(int(actions[a])) for a in range(self.n_agents)}
        suppressed_count = 0
        advanced_count = 0
        total_power = 0.0

        for i in range(self.n_radars):
            mode = self._ladders[i][self._states[i].mode_idx]
            attempts: list[tuple[float, int, bool]] = []
            for a in range(self.n_agents):
                target, jam_type, power_level = decoded[a]
                if target != i or jam_type == 0:
                    continue
                technique = self._technique_for(jam_type, mode)
                tech_idx = self._techniques.index(technique)
                band_match = not (
                    self._states[i].eccm_active and tech_idx == self._states[i].eccm_technique_idx
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
                attempts.append((j_s, tech_idx, suppressed))

            radar_suppressed = any(s for _, _, s in attempts)
            if attempts:
                pool = [(j, t) for j, t, s in attempts if s] or [(j, t) for j, t, _ in attempts]
                best_tech_idx = max(pool, key=lambda x: x[0])[1]
            else:
                best_tech_idx = self._none_idx

            self._states[i] = advance_threat(
                self._states[i],
                best_tech_idx,
                radar_suppressed,
                len(self._ladders[i]),
                lock_gain=self.config.lock_gain,
                lock_decay=self.config.lock_decay,
                n_eccm=self.config.n_eccm,
            )
            if radar_suppressed:
                suppressed_count += 1
            else:
                advanced_count += 1

        for a in range(self.n_agents):
            _, jam_type, power_level = decoded[a]
            if jam_type != 0 and self._n_power > 1:
                total_power += power_level / (self._n_power - 1)

        self._last_actions = [int(actions[a]) for a in range(self.n_agents)]
        self._t += 1

        lose = any(
            self._states[i].mode_idx == len(self._ladders[i]) - 1
            and self._states[i].lock_energy >= 1.0
            for i in range(self.n_radars)
        )
        terminated = lose
        truncated = (not lose) and self._t >= self.config.horizon_t
        if lose:
            outcome = "lose"
        elif truncated:
            outcome = "win"
        else:
            outcome = "ongoing"

        reward = (
            -self.config.w_lock * advanced_count
            - self.config.lambda_power * total_power
            + self.config.w_supp * suppressed_count
        )
        if outcome == "win":
            reward += self.config.r_win
        elif outcome == "lose":
            reward -= self.config.r_lose

        rewards = {a: float(reward) for a in range(self.n_agents)}
        return (
            self._obs(),
            self._global_state(),
            rewards,
            terminated,
            truncated,
            self._info(outcome, suppressed_count),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_env.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/ruff check src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
.venv/bin/mypy src/cog_ew/marl_formation/env.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/env.py tests/marl_formation/test_env.py
git commit -m "feat(marl): IADSFormationEnv.step (resolución por radar + recompensa de equipo)"
```

---

## Task 4: Verificación final (lint, tipos, suite completa)

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: todos los tests pasan (los anteriores + los nuevos de `marl_formation`).

- [ ] **Step 2: Lint y formato**

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```
Expected: `All checks passed!` y todos los ficheros formateados.

- [ ] **Step 3: Tipos sobre `src/`**

Run: `.venv/bin/mypy src`
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit (si hubo algún ajuste)**

```bash
git add -A && git commit -m "chore(marl): verificación final del entorno IADS" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **Interfaz CTDE (obs local por agente + estado global + step):** Tasks 2-3; `reset`/`step` con dict de obs y `state` global. ✅
- **Acción discreta compuesta `target × {none,deception,suppression} × power`:** Task 2 `encode_action`/`_decode_action`; `action_dim = M*3*P`. ✅
- **Lock abstracto por radar reutilizando Modelo 1:** Task 3 usa `advance_threat` + `jamming_effectiveness`. ✅
- **Resolución mejor-de-los-que-apuntan:** Task 3 (`pool` de intentos, prioriza los que suprimen). ✅
- **Mapeo de tipo a técnica (suppression = mejor supresora del modo):** Task 3 `_technique_for`. ✅
- **Recompensa de equipo compartida (lock+potencia+supresión+terminal):** Task 3; `rewards[a]` iguales. ✅
- **`suppressed_fraction` / `outcome` para el ancla:** Task 2/3 `_info`. ✅
- **Coordinación cubre mejor / radar sin cubrir avanza:** Task 3 tests. ✅
- **Determinismo por seed; config en YAML:** Tasks 1-3. ✅
- **Fuera de alcance respetado:** sin QMIX/entrenamiento (B), sin comparación/+45 % (C), sin geometría 2D. ✅
- **Consistencia de tipos:** `IADSEnvConfig`/`IADSFormationEnv` y firmas `reset()->(dict, ndarray, dict)`, `step(dict)->(dict, ndarray, dict, bool, bool, dict)` idénticas entre Tasks 2-3; `encode_action`/`_decode_action` inversas; reutiliza `advance_threat`/`jamming_effectiveness` con sus firmas exactas. ✅
