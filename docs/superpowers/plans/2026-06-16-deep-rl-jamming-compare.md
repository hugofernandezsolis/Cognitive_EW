# Arnés de comparación cognitivo vs baseline (Modelo 1, sub-pieza C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar el arnés que enfrenta la política cognitiva (D3QN) y el baseline rule-based (Modelo 5) sobre los mismos episodios del entorno y cuantifica el delta de win rate (ancla >92 % vs 58 %).

**Architecture:** `compare.py` define una `Policy` (Protocol), adaptadores `AgentPolicy` (D3QN greedy) y `BaselinePolicy` (técnica top de `EWResponseLibrary.select` a potencia máxima), y las funciones `evaluate_policy` y `compare` (deterministas por seed, mismos episodios para ambas políticas). Requiere exponer el nombre del emisor en `RadarJammingEnv.info` y una propiedad pública `n_power_levels`.

**Tech Stack:** Python 3.11, NumPy, Gymnasium, PyTorch, pytest, ruff, mypy. Herramientas vía `.venv/bin/<tool>`.

---

## File Structure

- Modify: `src/cog_ew/deep_rl_jamming/env.py` — añadir `"emitter"` a `_info` y una propiedad `n_power_levels`.
- Create: `src/cog_ew/deep_rl_jamming/compare.py` — `Policy`, `AgentPolicy`, `BaselinePolicy`, `evaluate_policy`, `compare`.
- Test: `tests/deep_rl_jamming/test_env.py` (un test nuevo), `tests/deep_rl_jamming/test_compare.py` (nuevo).

**Piezas disponibles (no modificar salvo lo indicado):** `RadarJammingEnv` (Gymnasium) con `encode_action(technique, power_level)->int`, `reset(seed)->(obs, info)`, `step(action)->(obs, reward, terminated, truncated, info)`, `info` con `real_mode`/`eccm_active`/`outcome`; `D3QNAgent.select_action(obs, epsilon)->int`; `EWResponseLibrary.from_yaml(path)` + `select(emitter, mode)->tuple[JammingTechnique, ...]`; `JammingTechnique` (Enum). El entorno guarda internamente `self._emitter` (con `.name`) y `self._n_power` (entero, nº de niveles de potencia).

---

## Task 1: `env.py` — exponer `emitter` en `info` + propiedad `n_power_levels`

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/env.py`
- Test: `tests/deep_rl_jamming/test_env.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_env.py` (el fichero ya tiene `import numpy as np`, `from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv`, la constante `CONFIG` y el helper `_env()`):

```python
def test_info_exposes_emitter_name():
    from cog_ew.data.pdw_library import EmitterLibrary

    env = _env()
    _, info = env.reset(seed=0)
    names = EmitterLibrary.from_yaml("configs/temporal_cnn_elint/emitters.yaml").emitter_names()
    assert info["emitter"] in names


def test_env_exposes_n_power_levels():
    env = _env()
    assert env.n_power_levels == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: FAIL — `KeyError: 'emitter'` y/o `AttributeError: ... has no attribute 'n_power_levels'`.

- [ ] **Step 3: Write minimal implementation**

En `src/cog_ew/deep_rl_jamming/env.py`, en el método `_info`, añadir la clave `"emitter"`:

```python
    def _info(self, outcome: str, j_s: float) -> dict[str, Any]:
        return {
            "real_mode": self._ladder[self._state.mode_idx],
            "emitter": self._emitter.name,
            "j_s": j_s,
            "eccm_active": self._state.eccm_active,
            "outcome": outcome,
        }
```

Y añadir esta propiedad dentro de `RadarJammingEnv` (justo después de `encode_action`):

```python
    @property
    def n_power_levels(self) -> int:
        return self._n_power
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_env.py -v`
Expected: PASS (los tests previos + los 2 nuevos).

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
git commit -m "feat(deep-rl): exponer emitter en info + n_power_levels"
```

---

## Task 2: `compare.py` — `Policy` + `AgentPolicy` + `BaselinePolicy`

**Files:**
- Create: `src/cog_ew/deep_rl_jamming/compare.py`
- Test: `tests/deep_rl_jamming/test_compare.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_compare.py`:

```python
import numpy as np

from cog_ew.deep_rl_jamming.compare import BaselinePolicy
from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.ew_library.library import EWResponseLibrary, JammingTechnique

ENV_CONFIG = "configs/deep_rl_jamming/env.yaml"
LIB_CONFIG = "configs/ew_library/responses.yaml"


def _env():
    return RadarJammingEnv(RadarEnvConfig.from_yaml(ENV_CONFIG))


def _library():
    return EWResponseLibrary.from_yaml(LIB_CONFIG)


def test_baseline_policy_selects_top_technique_at_max_power():
    env = _env()
    library = _library()
    policy = BaselinePolicy(library, env)
    info = {"emitter": "S-400", "real_mode": "missile_guidance"}
    top = library.select("S-400", "missile_guidance")[0]
    expected = env.encode_action(top, env.n_power_levels - 1)
    assert policy.act(np.zeros((8, 5), dtype=np.float32), info) == expected


def test_baseline_policy_lpi_uses_poor_technique():
    env = _env()
    library = _library()
    policy = BaselinePolicy(library, env)
    info = {"emitter": "LPI-FMCW", "real_mode": "track"}
    top = library.select("LPI-FMCW", "track")[0]
    assert top == JammingTechnique.EVASIVE
    expected = env.encode_action(JammingTechnique.EVASIVE, env.n_power_levels - 1)
    assert policy.act(np.zeros((8, 5), dtype=np.float32), info) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_compare.py -v`
Expected: FAIL con `cannot import name 'BaselinePolicy'`.

- [ ] **Step 3: Write minimal implementation**

Crear `src/cog_ew/deep_rl_jamming/compare.py`:

```python
"""Comparación entre la política cognitiva (D3QN) y el baseline rule-based (Modelo 5)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cog_ew.deep_rl_jamming.agent import D3QNAgent
from cog_ew.deep_rl_jamming.env import RadarJammingEnv
from cog_ew.ew_library.library import EWResponseLibrary


class Policy(Protocol):
    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int: ...


class AgentPolicy:
    def __init__(self, agent: D3QNAgent) -> None:
        self.agent = agent

    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int:
        return self.agent.select_action(obs, epsilon=0.0)


class BaselinePolicy:
    def __init__(self, library: EWResponseLibrary, env: RadarJammingEnv) -> None:
        self.library = library
        self.env = env

    def act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int:
        techniques = self.library.select(info["emitter"], info["real_mode"])
        return self.env.encode_action(techniques[0], self.env.n_power_levels - 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_compare.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/compare.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
git commit -m "feat(deep-rl): Policy + AgentPolicy + BaselinePolicy"
```

---

## Task 3: `compare.py` — `evaluate_policy` + `compare`

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/compare.py`
- Test: `tests/deep_rl_jamming/test_compare.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_compare.py` (ampliar el import de `compare` con `AgentPolicy, compare, evaluate_policy` y añadir los imports de agente):

```python
from cog_ew.deep_rl_jamming.agent import D3QNAgent, D3QNConfig
from cog_ew.deep_rl_jamming.compare import (
    AgentPolicy,
    BaselinePolicy,
    compare,
    evaluate_policy,
)

_OPTIMAL = {
    "search": [JammingTechnique.NOISE, JammingTechnique.DECEPTION],
    "tws": [JammingTechnique.DECEPTION, JammingTechnique.NOISE],
    "track": [JammingTechnique.CROSS_EYE, JammingTechnique.VGPO],
    "missile_guidance": [JammingTechnique.CHAFF, JammingTechnique.RGPO],
}


class _OraclePolicy:
    def __init__(self, env):
        self.env = env
        self.last = None

    def act(self, obs, info):
        ranked = _OPTIMAL[info["real_mode"]]
        technique = ranked[1] if info["eccm_active"] and self.last == ranked[0] else ranked[0]
        self.last = technique
        return self.env.encode_action(technique, self.env.n_power_levels - 1)


def _baseline(env):
    return BaselinePolicy(_library(), env)


def test_evaluate_policy_returns_bounded_metrics_deterministically():
    a = evaluate_policy(_env(), _baseline(_env_holder := _env()), episodes=5, seed=0)
    assert 0.0 <= a["win_rate"] <= 1.0
    assert np.isfinite(a["mean_reward"])


def test_evaluate_policy_is_deterministic_by_seed():
    env_a = _env()
    env_b = _env()
    a = evaluate_policy(env_a, _baseline(env_a), episodes=5, seed=0)
    b = evaluate_policy(env_b, _baseline(env_b), episodes=5, seed=0)
    assert a == b


def test_oracle_beats_or_matches_fixed_baseline():
    base_env = _env()
    oracle_env = _env()
    base_metrics = evaluate_policy(base_env, _baseline(base_env), episodes=8, seed=0)
    oracle_metrics = evaluate_policy(oracle_env, _OraclePolicy(oracle_env), episodes=8, seed=0)
    assert oracle_metrics["win_rate"] >= base_metrics["win_rate"]


def test_compare_delta_is_difference():
    env = _env()
    cognitive = AgentPolicy(
        D3QNAgent(40, 40, D3QNConfig(hidden=16), "cpu", np.random.default_rng(0))
    )
    result = compare(env, cognitive, _baseline(env), episodes=5, seed=0)
    assert set(result) == {"cognitive", "baseline", "delta"}
    assert result["delta"]["win_rate"] == (
        result["cognitive"]["win_rate"] - result["baseline"]["win_rate"]
    )
```

(Nota: en `test_evaluate_policy_returns_bounded_metrics_deterministically`, la política y el entorno deben ser
el mismo objeto `env`; usar `env = _env(); evaluate_policy(env, _baseline(env), ...)`. Reescribir esa primera
línea así para mayor claridad:)

```python
def test_evaluate_policy_returns_bounded_metrics_deterministically():
    env = _env()
    a = evaluate_policy(env, _baseline(env), episodes=5, seed=0)
    assert 0.0 <= a["win_rate"] <= 1.0
    assert np.isfinite(a["mean_reward"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_compare.py -v`
Expected: FAIL con `cannot import name 'evaluate_policy'`.

- [ ] **Step 3: Write minimal implementation**

Añadir al final de `src/cog_ew/deep_rl_jamming/compare.py`:

```python
def evaluate_policy(
    env: RadarJammingEnv, policy: Policy, episodes: int, seed: int
) -> dict[str, float]:
    wins = 0
    total_reward = 0.0
    total_steps = 0
    obs, info = env.reset(seed=seed)
    for _ in range(episodes):
        done = False
        while not done:
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        obs, info = env.reset()
    return {
        "win_rate": wins / episodes,
        "mean_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
    }


def compare(
    env: RadarJammingEnv,
    cognitive: Policy,
    baseline: Policy,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    cognitive_metrics = evaluate_policy(env, cognitive, episodes, seed)
    baseline_metrics = evaluate_policy(env, baseline, episodes, seed)
    return {
        "cognitive": cognitive_metrics,
        "baseline": baseline_metrics,
        "delta": {
            "win_rate": cognitive_metrics["win_rate"] - baseline_metrics["win_rate"],
            "mean_reward": cognitive_metrics["mean_reward"] - baseline_metrics["mean_reward"],
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_compare.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/compare.py
```
Expected: sin errores. (Los helpers de test sin anotaciones de tipo son aceptables; si mypy se ejecuta solo
sobre `src/`, no afecta. El proyecto corre `mypy src`.)

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/compare.py tests/deep_rl_jamming/test_compare.py
git commit -m "feat(deep-rl): evaluate_policy + compare (delta de win rate)"
```

---

## Task 4: Verificación final (lint, tipos, suite completa)

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: todos los tests pasan (los anteriores + los nuevos de `compare`/`env`).

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
git add -A && git commit -m "chore(deep-rl): verificación final del arnés de comparación" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **Cambio en `env.py` (`emitter` en `info`):** Task 1 + test `test_info_exposes_emitter_name`. ✅
- **Propiedad `n_power_levels`:** Task 1 (la usa `BaselinePolicy`); test `test_env_exposes_n_power_levels`. ✅
- **`Policy` (Protocol) + `AgentPolicy` + `BaselinePolicy` (técnica top a potencia máxima):** Task 2; tests de técnica top y LPI. ✅
- **`evaluate_policy` (win_rate/mean_reward/mean_steps, determinista):** Task 3; tests de cotas y determinismo. ✅
- **`compare` con `delta`:** Task 3; `delta.win_rate == cognitive − baseline`. ✅
- **Contraste de dinámica (adaptativo ≥ baseline fijo) sin entrenar el D3QN:** Task 3 `test_oracle_beats_or_matches_fixed_baseline`. ✅
- **Reproducibilidad (determinista por seed, sin estado global ni I/O):** Tasks 2-3. ✅
- **Fuera de alcance respetado:** sin entrenar a >92 %, sin Colab, sin persistencia de informes, sin multi-seed. ✅
- **Consistencia de tipos:** `Policy.act(obs, info)->int` idéntica en `AgentPolicy`/`BaselinePolicy` y en las llamadas de `evaluate_policy`; `compare` usa `evaluate_policy` con la misma firma; `BaselinePolicy(library, env)` y `AgentPolicy(agent)` consistentes entre Tasks 2-3 y sus usos. ✅
