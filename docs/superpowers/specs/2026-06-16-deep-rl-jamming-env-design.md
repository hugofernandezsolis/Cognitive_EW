# Diseño — Entorno del ciclo radar para jamming adaptativo (Modelo 1, sub-pieza A)

**Fecha:** 2026-06-16
**Ámbito:** `src/cog_ew/deep_rl_jamming/{threat,reward,env}.py` + `configs/deep_rl_jamming/env.yaml` + `tests/deep_rl_jamming/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 1 es un **agente Deep RL para jamming adaptativo** que observa la
señal del radar amenaza y genera en tiempo real la técnica de jamming óptima, adaptándose cuando el radar
cambia de modo o activa ECCM, con **latencia <5 ms**. Ancla cuantitativa: **>92 %** de victorias de espectro
vs. **58 %** del baseline rule-based (Modelo 5).

El Modelo 1 se **descompone** en tres sub-piezas con ciclos `spec → plan` independientes:

- **A — Entorno del ciclo radar** (este spec): simulador del radar amenaza con interfaz Gymnasium, testable
  sin RL. Es la dependencia raíz.
- **B — Agente RL + entrenamiento** (`agent.py`, `train.py`): algoritmo (D3QN/SAC), red de política, replay,
  bucle de entrenamiento, perfilado de latencia, checkpoints. Depende de la interfaz de A.
- **C — Arnés de comparación vs baseline** (Modelo 5) y métricas (win rate, J/S): se pliega en la evaluación
  de B o se difiere a Fase 6.

Estado del arte (`docs/research/estado-del-arte.md` §1): la formulación es MDP/POMDP; D3QN para acción
discreta e Improved-SAC+Wolpertinger para acción discreta grande. El hueco del TFM (§1.4): **latencia <5 ms**
documentada, **espacio de acción discreto-compuesto con técnicas EW reales** (noise, DRFM, cross-eye, VGPO,
RGPO), y **observabilidad parcial (POMDP con historial)**. Este spec materializa ese hueco en el diseño del
entorno.

Decisiones tomadas en brainstorming:
- **Interfaz:** entorno propio conforme a la **API Gymnasium** (dep nueva `gymnasium`); deja la puerta abierta
  a Stable-Baselines3 en la sub-pieza B.
- **Observación:** **parcial + historial (POMDP)** — el agente ve solo los parámetros emitidos medibles del
  radar, apilados en una ventana de K pasos; no ve el modo interno ni el estado de lock.
- **Acción:** discreta compuesta **técnica × nivel de potencia** (~40), reusando `JammingTechnique` (Modelo 5).
- **Recompensa:** **densa** — efectividad (J/S vs burnthrough) − penalización de potencia + término terminal
  (victoria por agotar horizonte / derrota por launch).

## Modelo de dominio

### Estado oculto del radar (no observable por el agente)

- `mode` ∈ `MODES` = (`search`, `tws`, `track`, `missile_guidance`) (reusa `pdw_library.MODES`).
- `lock_energy: float` — contador en [0, 1]. Si el jamming **no** suprime el modo actual, se incrementa; al
  cruzar umbrales promociona el modo (search→tws→track→missile_guidance). Si el jamming **sí** suprime, decae
  y el radar puede regresar de modo.
- `eccm_active: bool` — cuando el radar acumula `N_eccm` pasos consecutivos de jamming efectivo, activa ECCM
  (salto de banda/cambio de waveform) que **anula la efectividad de la técnica actual** hasta que el agente
  cambia de técnica o de banda. Modela la adaptación del radar.
- `emitter: EmitterSpec` — emisor objetivo muestreado de `EmitterLibrary` (de `configs/.../emitters.yaml`).
  Sus `ModeSpec` (rf_band, pri, pw, scan) generan los parámetros emitidos observables del modo actual →
  coherencia con la taxonomía de amenazas de los Modelos 2 y 5.

### Efectividad (J/S)

- Matriz configurable `efectividad[técnica][modo] ∈ [0, 1]` (p. ej. RGPO/VGPO/DRFM altos vs `track`; `noise`
  vs `search`; `chaff` vs `missile_guidance`; `cross_eye` vs `track` monopulso; `none` = 0).
- `jamming_effectiveness(technique, power_level, mode, eccm_active, band_match) -> tuple[float, bool]`:
  - `j_s = base_js[power_level] + efectividad[técnica][modo] · escala` (dB), reducido si `band_match` es falso
    (el jammer no cubre la banda actual del radar, p. ej. tras un salto ECCM) y anulado si `eccm_active` y la
    técnica no se ha readaptado.
  - `suppressed = (j_s >= umbral_burnthrough) and efectividad[técnica][modo] > umbral_técnica`.
  - Devuelve `(j_s, suppressed)`.

### Recompensa

`compute_reward(j_s, suppressed, power_level, terminal) -> float`:
- término de efectividad: `+ w_eff · (j_s − umbral_burnthrough)` si `suppressed`, pequeño negativo si no.
- penalización de potencia: `− λ · power_level_normalizado`.
- término terminal: `+ R_win` si el episodio termina por agotar el horizonte sin launch; `− R_lose` si el
  radar alcanza `missile_guidance` sostenido (launch).

## Interfaz Gymnasium (`RadarJammingEnv`)

### Espacios

- **Observación:** `gymnasium.spaces.Box(low, high, shape=(K, F), dtype=float32)` — ventana de los últimos K
  pasos de F parámetros observables: `rf, pri, pw, scan, eccm_detectado`. Normalizados con
  `pdw_library.CONTINUOUS_RANGES` (rf/pri/pw) y rangos propios para scan y el flag binario. La ventana se
  rellena con ceros antes del paso K.
- **Acción:** `gymnasium.spaces.Discrete(n_techniques * n_power_levels)`, decodificada a
  `(JammingTechnique, power_level)` mediante `divmod`.

### Ciclo

- `reset(seed=None, options=None) -> tuple[obs, info]`: siembra el RNG (`super().reset(seed=seed)`), muestrea
  emisor objetivo, fija `mode=search`, `lock_energy=0`, `eccm_active=False`, historial a cero. **Determinista
  por seed.**
- `step(action) -> tuple[obs, reward, terminated, truncated, info]`:
  1. decodifica `(technique, power_level)`.
  2. calcula `(j_s, suppressed)` con `reward.jamming_effectiveness`.
  3. actualiza el estado oculto (`lock_energy`, `mode`, `eccm_active`) con `threat`.
  4. calcula la recompensa con `reward.compute_reward`.
  5. construye la observación apilando los últimos K parámetros emitidos.
  6. `terminated = launch` (derrota); `truncated = (t >= T)` (victoria por horizonte).
- `info`: expone **solo para logging/eval** (no para el agente) `real_mode`, `j_s`, `eccm_active`,
  `outcome` ∈ {`win`, `lose`, `ongoing`} → permite calcular el **win rate** en B/C.

### Latencia

`step` y la construcción de observación son O(K·F) en NumPy; el entorno no debe ser el cuello de botella del
objetivo <5 ms (que es del agente, sub-pieza B).

## Componentes y ficheros

- `src/cog_ew/deep_rl_jamming/threat.py` — `RadarState` (dataclass) + funciones puras de transición de modo y
  ECCM. Sin Gymnasium.
- `src/cog_ew/deep_rl_jamming/reward.py` — funciones puras `jamming_effectiveness` y `compute_reward`.
- `src/cog_ew/deep_rl_jamming/env.py` — `RadarEnvConfig` (dataclass + `from_yaml`) y `RadarJammingEnv`
  (`gymnasium.Env`); consume `threat` y `reward`.
- `configs/deep_rl_jamming/env.yaml` — `emitters` (path a la librería), `history_k`, `horizon_t`,
  `power_levels` (dB), `burnthrough_threshold`, matriz `effectiveness` (`técnica×modo`), parámetros ECCM
  (`n_eccm`), pesos de reward (`w_eff`, `lambda_power`, `r_win`, `r_lose`), `seed`.

## Reproducibilidad

- Parámetros solo en `env.yaml` versionado; nada hardcodeado.
- Entorno determinista por seed (RNG de Gymnasium); sin estado global ni aleatoriedad fuera del RNG sembrado.
- `threat` y `reward` son funciones puras → testables con casos conocidos.

## Dependencias

Nueva: `gymnasium` en `pyproject.toml`. `numpy` ya está. No requiere PyTorch (el entorno es NumPy puro; la
red vive en la sub-pieza B).

## Tests (`tests/deep_rl_jamming/`)

- `test_threat.py`: promoción de modo cuando el jamming no suprime; regresión cuando sí; activación de ECCM
  tras `n_eccm` pasos efectivos; determinismo de las transiciones.
- `test_reward.py`: `jamming_effectiveness` por casos (técnica correcta vs modo → `suppressed=True`; ECCM activo
  no readaptado → anula; `band_match=False` → J/S bajo; `none` → 0); penalización de potencia monótona;
  términos terminales `R_win`/`R_lose`.
- `test_env.py`: cumple la API Gymnasium (shapes/tipos de `reset`/`step`; usar `gymnasium.utils.env_checker`
  si está disponible); determinismo por seed (dos `reset(seed=0)` + misma secuencia de acciones → mismas
  observaciones/recompensas); un episodio con "jammer pasivo" (siempre `none`) termina en **derrota** (el
  radar consigue lock); un episodio con la técnica óptima por modo termina en **victoria** — fija la dinámica.

## Fuera de alcance (YAGNI)

- Agente RL, red de política, replay, bucle de entrenamiento y perfilado de latencia del agente (sub-pieza B).
- Arnés de comparación vs baseline y win rate agregado (sub-pieza B/C / Fase 6).
- Multi-agente / formación (Modelo 3).
- Co-simulación MATLAB, selección de banda como acción (técnica×potencia×banda), Wolpertinger.

## Decisiones clave (resumen)

1. **Entorno propio con API Gymnasium** (dep `gymnasium`), reset/step/spaces estándar.
2. **POMDP con historial:** observación parcial (parámetros emitidos) apilada en ventana de K pasos.
3. **Acción discreta compuesta técnica × potencia** (~40), reusa `JammingTechnique`.
4. **Estado oculto del radar** con promoción/regresión de modo por `lock_energy` y **ECCM** que anula la
   técnica hasta readaptación.
5. **Recompensa densa** J/S − penalización de potencia + terminal win/lose; el resultado terminal define el
   win rate del ancla.
6. **Emisor objetivo de `EmitterLibrary`** → coherencia con los Modelos 2 y 5.
7. **`threat` y `reward` puros**, `env` orquesta; parámetros en YAML versionado, determinista por seed.
