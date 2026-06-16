# Diseño — Entorno IADS multi-agente para coordinación EW en formación (Modelo 3, sub-pieza A)

**Fecha:** 2026-06-16
**Ámbito:** `src/cog_ew/marl_formation/env.py` + `configs/marl_formation/env.yaml` + `tests/marl_formation/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 3 es **Multi-Agent RL para coordinación de EW en formación**:
cada aeronave es un agente EW que coordina emisiones para maximizar la supresión del IADS adversario. Ancla
cuantitativa: **+45 %** de supresión del IADS coordinando **4 aeronaves** frente a actuación independiente.

El Modelo 3 se **descompone** en A (entorno IADS multi-agente, este spec), B (agentes QMIX + entrenamiento
CTDE) y C (comparación vs actuación independiente, +45 %).

Estado del arte (`docs/research/estado-del-arte.md` §3): formulación **Dec-POMDP / juego de Markov**, esquema
**CTDE** (entrenamiento centralizado, ejecución descentralizada), algoritmo de referencia **QMIX**. El caso de
estudio **MA-CJD** (4 jammers vs 4 radares, acción parametrizada objetivo+tipo × potencia, recompensa
lock+potencia+éxito de jamming) mapea ~1:1. Diferenciadores del TFM (§3.4): métricas EW, escenario NTTR con la
taxonomía SA-2…S-400 ya modelada (`EmitterLibrary`), deception coordinada, latencia documentada, y reutilizar
la receta validada.

Decisiones tomadas en brainstorming:
- **Interfaz:** **CTDE propia mínima** — `reset`/`step` que devuelven observación local por agente **y** un
  estado global explícito (lo que QMIX necesita); sin dependencia nueva.
- **Acción por agente:** **discreta compuesta** `(radar objetivo × tipo × nivel de potencia)` → QMIX puro
  (sin MP-DQN).
- **Fidelidad:** **lock abstracto por radar** reutilizando `threat.advance_threat` y
  `reward.jamming_effectiveness` del Modelo 1; sin geometría 2D.

## Modelo de dominio

- **N jammer-agentes vs M radares.** Cada radar tiene un `RadarState` (de `deep_rl_jamming.threat`) y un
  emisor objetivo muestreado de `EmitterLibrary`; su dinámica de modo/lock se rige por `advance_threat`.
- **Acción por agente:** `Discrete(M × 3 × P)`, decodificada a `(target, jam_type, power_level)` donde
  `jam_type ∈ {none, deception, suppression}` (índices 0/1/2) y `target ∈ [0, M)`, `power_level ∈ [0, P)`.
  Mapeo a técnica para la efectividad:
  - `none` → efectividad 0 (no jamming).
  - `deception` → `JammingTechnique.DECEPTION`.
  - `suppression` → el entorno elige la técnica supresora más efectiva para el **modo actual** del radar
    objetivo, entre un conjunto supresor configurable (p. ej. `noise, drfm_repeater, vgpo, rgpo, cross_eye,
    chaff`), tomando la de mayor valor en la matriz de efectividad.
- **Resolución por radar:** para cada radar, se considera el **mejor** `jamming_effectiveness` entre los
  agentes que lo apuntan (varios agentes al mismo radar **no** suman). El radar se suprime si ese mejor
  resultado da `suppressed=True`; `advance_threat` actualiza su estado (regresa si suprimido, avanza si no).
  Apuntar en exceso a un radar desperdicia potencia (penalización) y deja otros radares sin cubrir → la
  coordinación (repartir objetivos) emerge sin término de fratricidio explícito.

## Interfaz CTDE (`IADSFormationEnv`)

- **Agentes:** identificados por índice `0..N-1` (claves de los dicts).
- **Observación local por agente** (`dict[int, NDArray[np.float32]]`): por cada radar, sus parámetros
  emitidos `(rf, pri, pw, scan, eccm_detectado)` normalizados (`M × 5`), aplanados, concatenados con el
  one-hot del id del agente → vector de tamaño `M*5 + N`. Misma info de radares para todos los agentes (la
  localidad por sector se anota como mejora futura, YAGNI).
- **Estado global** (`NDArray[np.float32]`, solo entrenamiento, para la mixing network): por cada radar,
  `(modo one-hot [4], lock_energy [1], eccm_active [1])` + la última acción (índice normalizado) de cada
  agente → vector determinista.
- **`reset(seed=None) -> tuple[dict, NDArray, dict]`**: siembra el RNG; muestrea M emisores objetivo (de los
  candidatos de la librería); inicializa cada `RadarState` (modo search, lock 0); última acción a 0.
  Determinista por seed. Devuelve `(obs, state, info)` con `info["outcome"] = "ongoing"`.
- **`step(actions: dict[int, int]) -> tuple[dict, NDArray, dict, bool, bool, dict]`**:
  1. decodifica cada acción a `(target, jam_type, power_level)`.
  2. por radar: efectividad = mejor `jamming_effectiveness` entre los agentes que lo apuntan (band_match se
     resuelve por radar como en el Modelo 1); `advance_threat` actualiza su `RadarState`.
  3. recompensa de **equipo** (escalar, compartida — fully-cooperative QMIX):
     `r = −w_lock·(nº radares que avanzaron) − λ·Σ potencia_norm + w_supp·(nº radares suprimidos) + terminal`.
  4. `terminated` si **algún** radar alcanza `missile_guidance` con `lock_energy ≥ 1.0` (IADS fija → misión
     falla); `truncated` si `t ≥ horizon_t` (la formación aguanta → éxito).
  5. devuelve `(obs, state, rewards, terminated, truncated, info)` con `rewards[i] = r` para todo agente, e
     `info` con `suppressed_fraction` (radares suprimidos / M) y `outcome ∈ {ongoing, win, lose}`.

## Componentes y ficheros

- `src/cog_ew/marl_formation/env.py` — `IADSEnvConfig` (dataclass + `from_yaml`) e `IADSFormationEnv`.
  Reutiliza `deep_rl_jamming.threat.{RadarState, advance_threat}`, `deep_rl_jamming.reward.jamming_effectiveness`,
  `ew_library.library.JammingTechnique`, `data.pdw_library.{EmitterLibrary, MODES, CONTINUOUS_RANGES}`.
- `configs/marl_formation/env.yaml` — `library_path`, `emitters` (subconjunto, p. ej. SAMs con los 4 modos),
  `n_agents`, `n_radars`, `power_levels`, `effectiveness` (matriz técnica×modo, reusa la del Modelo 1),
  `suppression_techniques` (lista), `burnthrough`, `eff_threshold`, `js_scale`, `lock_gain`, `lock_decay`,
  `n_eccm`, `w_lock`, `lambda_power`, `w_supp`, `r_win`, `r_lose`, `horizon_t`, `seed`.

## Reproducibilidad

- Parámetros solo en `env.yaml` versionado.
- Determinista por seed (RNG propio sembrado en `reset`); sin estado global ni aleatoriedad fuera del RNG.

## Dependencias

Ninguna nueva (NumPy, PyYAML ya están). No requiere PyTorch (el entorno es NumPy puro). Importa primitivas EW
de `deep_rl_jamming` y `ew_library`/`data`.

## Tests (`tests/marl_formation/`)

- `test_env.py`:
  - `IADSEnvConfig.from_yaml` parsea los parámetros.
  - `reset` devuelve `obs` (dict de N agentes con la forma `(M*5 + N,)`), `state` global con la forma esperada
    e `info["outcome"] == "ongoing"`; determinista por seed.
  - `step` con un dict de acciones de los N agentes devuelve la tupla CTDE; `rewards` compartida (todos los
    valores iguales); shapes correctas.
  - **coordinación cubre mejor:** repartir agentes entre radares distintos da `suppressed_fraction` ≥ (y
    avance de lock ≤) que concentrar todos en un solo radar, sobre el mismo seed.
  - **radar sin cubrir avanza:** con todos los agentes en `none`, todos los radares progresan y el episodio
    termina en `lose` dentro del horizonte.
  - determinismo de un rollout por seed (misma secuencia de acciones → mismas recompensas).

## Fuera de alcance (YAGNI)

- Agentes QMIX y bucle de entrenamiento CTDE (sub-pieza B).
- Comparación vs actuación independiente y el +45 % (sub-pieza C / Fase 6).
- Geometría 2D / cinemática de radares, observación parcial por sector.
- MP-DQN / potencia continua, deception con modelado detallado de falsos blancos.

## Decisiones clave (resumen)

1. **Interfaz CTDE propia** (obs local por agente + estado global explícito), sin dep nueva.
2. **Acción discreta compuesta** `target × {none, deception, suppression} × power` → QMIX puro.
3. **Lock abstracto por radar** reutilizando `advance_threat` y `jamming_effectiveness` del Modelo 1.
4. **Mejor-de-los-que-apuntan por radar** → la coordinación emerge (sin término de fratricidio explícito).
5. **Recompensa de equipo compartida** (lock + potencia + supresión + terminal); `suppressed_fraction` para
   medir el ancla +45 %.
6. **Determinista por seed**, parámetros en YAML versionado.
