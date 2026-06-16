# Diseño — Arnés de comparación cognitivo vs baseline (Modelo 1, sub-pieza C)

**Fecha:** 2026-06-16
**Ámbito:** `src/cog_ew/deep_rl_jamming/compare.py` + `src/cog_ew/deep_rl_jamming/env.py` (1 línea) + `tests/deep_rl_jamming/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 1 (Deep RL jamming) debe superar al baseline rule-based en
**>92 %** de victorias de espectro vs. **58 %** (Modelo 5). Esta sub-pieza C cierra la descomposición del
Modelo 1 (A entorno, B agente+train, **C comparación**) aportando el **arnés que cuantifica ese delta**:
enfrenta la política cognitiva (D3QN entrenado) y el baseline (Modelo 5) sobre los mismos episodios del
entorno y reporta win rate, recompensa media y su diferencia.

Piezas ya disponibles (mergeadas):
- `RadarJammingEnv` (Gymnasium): `reset(seed)->(obs, info)`, `step(action)->(obs, reward, terminated,
  truncated, info)`, `info` con `real_mode`, `j_s`, `eccm_active`, `outcome ∈ {ongoing, win, lose}`;
  `encode_action(technique, power_level)->int`; `action_space = Discrete(40)`.
- `D3QNAgent.select_action(obs, epsilon)->int` (greedy con `epsilon=0`).
- `EWResponseLibrary.select(emitter, mode)->tuple[JammingTechnique, ...]` (combinación priorizada; Modelo 5).

Decisiones tomadas en brainstorming:
- **Acción del baseline:** la **técnica top** de la combinación a **potencia máxima**, vía `encode_action`.
  Fija, sin readaptación al ECCM — la debilidad que el cognitivo debe explotar.
- **Emisor para el lookup:** se **expone el nombre del emisor en `info`** (cambio mínimo en `env.py`); el
  baseline keyea por `(emitter, mode)` → baseline con "clasificación perfecta" de la amenaza.
- **"Done" del slice:** el arnés de comparación (máquina de medida); entrenar el agente al >92 % real y la
  ejecución en Colab son Fase 6.

## Cambio en el entorno (sub-pieza A)

En `RadarJammingEnv._info`, añadir la clave `"emitter": self._emitter.name`. No altera spaces, dinámica ni
recompensa; los tests existentes siguen pasando. Es un dato de logging/eval (como `real_mode`), no entra en
la observación del agente.

## Componentes (`compare.py`)

- **`Policy` (Protocol):** `act(self, obs: NDArray[np.float32], info: dict[str, Any]) -> int`. Unifica las
  políticas bajo una firma común (recibe la observación y el `info` del paso anterior / del `reset`).
- **`AgentPolicy`:** envuelve un `D3QNAgent`; `act(obs, info)` = `agent.select_action(obs, epsilon=0.0)`
  (greedy; ignora `info`). El agente puede estar entrenado o no — el arnés es agnóstico.
- **`BaselinePolicy`:** envuelve una `EWResponseLibrary`, el `encode_action` del entorno y `n_power`;
  `act(obs, info)` = `library.select(info["emitter"], info["real_mode"])[0]` (técnica prioritaria) a potencia
  máxima → `encode_action(technique, n_power - 1)`. Determinista, sin estado.
- **`evaluate_policy(env, policy, episodes, seed) -> dict[str, float]`:** corre `episodes` episodios; siembra
  el entorno con `seed` al inicio (`reset(seed=seed)`), y en cada episodio usa el `info` del paso previo (el
  de `reset` para la primera acción) para `policy.act`. Acumula `outcome`, recompensa total y nº de pasos.
  Devuelve `{"win_rate", "mean_reward", "mean_steps"}`.
- **`compare(env, cognitive, baseline, episodes, seed) -> dict[str, Any]`:** evalúa ambas políticas con el
  **mismo `seed`** (mismas secuencias de episodios → comparación justa) y devuelve
  `{"cognitive": {...}, "baseline": {...}, "delta": {"win_rate": cog−base, "mean_reward": cog−base}}`.

## Flujo de datos

```
env(reset seed) → (obs, info) → policy.act(obs, info) → action → env.step → (obs, reward, term, trunc, info)
  evaluate_policy agrega outcome/reward/steps por episodio → métricas
  compare(cognitive, baseline) con el mismo seed → delta de win rate (ancla >92% vs 58%)
```

`BaselinePolicy` usa `info["emitter"]` + `info["real_mode"]` (clasificación perfecta de la amenaza);
`AgentPolicy` usa solo `obs` (observación parcial POMDP). El contraste es exactamente respuesta fija conocida
vs. política adaptativa aprendida.

## Reproducibilidad

- `evaluate_policy`/`compare` deterministas por `seed` (reseed del entorno; el agente greedy es determinista,
  el baseline no tiene aleatoriedad).
- Sin estado global, sin I/O (el caller decide qué hacer con el dict resultante).

## Dependencias

Ninguna nueva. Reutiliza el entorno, el agente, `EWResponseLibrary` y `env.yaml`/`responses.yaml`.

## Tests (`tests/deep_rl_jamming/`)

- `test_compare.py`:
  - `BaselinePolicy.act` devuelve la acción de la técnica top a potencia máxima para un `(emitter, mode)`
    conocido (verificable contra `EWResponseLibrary.select` + `encode_action`).
  - `BaselinePolicy` con un emisor **LPI** (p. ej. `LPI-FMCW`) elige una técnica deliberadamente pobre
    (coherente con el baseline del Modelo 5).
  - `evaluate_policy` devuelve `win_rate ∈ [0, 1]` y `mean_reward` finito; determinista por seed.
  - **contraste de dinámica:** una política "oráculo" que cambia de técnica ante `eccm_active` obtiene
    `win_rate ≥` el de `BaselinePolicy` (fija) sobre los mismos episodios — el arnés detecta la ventaja del
    comportamiento adaptativo, **sin** depender de entrenar el D3QN real.
  - `compare` devuelve `cognitive`/`baseline`/`delta` con `delta["win_rate"] == cognitive["win_rate"] −
    baseline["win_rate"]`.
- `test_env.py`: `info["emitter"]` es uno de los nombres de la librería de emisores.

## Fuera de alcance (YAGNI)

- Entrenar el agente al >92 % real y la ejecución en Colab (Fase 6, `/experiment-run`).
- Métricas J/S/burnthrough agregadas más allá de la recompensa del entorno.
- Persistencia de informes/tablas comparativas (el caller decide).
- Significancia estadística / múltiples seeds agregados (se puede añadir luego sobre `compare`).

## Decisiones clave (resumen)

1. **`info["emitter"]`** expuesto (cambio mínimo en `env.py`) → baseline keyea por `(emitter, mode)`.
2. **`BaselinePolicy`** = técnica top de `select(...)` a potencia máxima, fija (sin readaptación a ECCM).
3. **`AgentPolicy`** = D3QN greedy (observación parcial).
4. **`evaluate_policy`/`compare`** deterministas por seed, mismos episodios para ambas políticas.
5. **El delta de win rate** materializa el ancla de `Propuesta.md`; las cifras reales salen en Fase 6.
6. **Cierra el Modelo 1** (A+B+C) como slice entrenable + evaluable.
