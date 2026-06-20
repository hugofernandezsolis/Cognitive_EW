# Diseño — Agentes QMIX + entrenamiento CTDE (Modelo 3, sub-pieza B)

**Fecha:** 2026-06-20
**Ámbito:** `src/cog_ew/marl_formation/{agents.py,train.py}` + `configs/marl_formation/qmix.yaml` + `tests/marl_formation/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 3 es **Multi-Agent RL para coordinación de EW en
formación**: cada aeronave es un agente EW que coordina emisiones para maximizar la supresión del IADS
adversario. Ancla cuantitativa: **+45 %** de supresión coordinando **4 aeronaves** frente a actuación
independiente (la comparación que mide ese ancla es la sub-pieza C, no esta).

El Modelo 3 se descompone en A (entorno IADS multi-agente ✅ mergeado), **B (agentes QMIX + entrenamiento
CTDE, este spec)** y C (comparación vs actuación independiente, +45 %).

Estado del arte (`docs/research/estado-del-arte.md` §3): formulación Dec-POMDP / juego de Markov, esquema
**CTDE** (entrenamiento centralizado, ejecución descentralizada), algoritmo de referencia **QMIX**. El caso
MA-CJD (4 jammers vs 4 radares) mapea ~1:1 con el entorno A ya construido.

El entorno A (`IADSFormationEnv`) ya expone exactamente lo que QMIX necesita:
- `reset`/`step` devuelven **obs local por agente** (`dict[int, NDArray]`, `obs_dim = M*5 + N`, con el
  one-hot del id del agente ya incluido → favorece parámetros compartidos), **estado global explícito**
  (`state_dim = M*(4+2) + N`), **recompensa de equipo** (escalar compartido, `rewards[i]` igual para todos),
  y `terminated`/`truncated`.

Decisiones tomadas en brainstorming:
- **Red de agente:** **GRU recurrente (DRQN)** con **parámetros compartidos** entre los N agentes (QMIX
  canónico; rastrea el `lock_energy` oculto no observable en la obs local). Implica **replay por episodios**.
- **Métrica de evaluación/selección de checkpoint:** **win-rate** (`outcome == "win"`), consistente con el
  Modelo 1; `suppressed_fraction` media se loguea como señal secundaria.

## Arquitectura general (CTDE)

Tres piezas nuevas en `src/cog_ew/marl_formation/`:

- **`agents.py`** — red de agente recurrente compartida (`AgentRNN`), red de mezcla monótona (`QMixer`),
  configuración (`QMIXConfig`) y orquestador (`QMIXLearner`).
- **`train.py`** — bucle de entrenamiento CTDE con replay por episodios, evaluación de win-rate,
  perfilado de latencia, y logging de reproducibilidad.
- **`configs/marl_formation/qmix.yaml`** — config de entrenamiento (apunta al `env.yaml` existente +
  hiperparámetros QMIX).

**Entrenamiento centralizado** (el mixer ve el estado global). **Ejecución descentralizada** (en inferencia
cada agente corre solo su `AgentRNN` sobre su obs local → la latencia perfilada es la de **un** agente,
forward con batch=1).

## `agents.py` — componentes

### `AgentRNN` (compartida entre los N agentes)

- `Linear(obs_dim → hidden) → ReLU → GRUCell(hidden → hidden) → Linear(hidden → action_dim)`.
- Entrada: obs local del agente `(B, obs_dim)` + hidden previo `(B, hidden)`. Salida: Q por acción
  `(B, action_dim)` y nuevo hidden `(B, hidden)`. `action_dim = M*3*P`.
- Parámetros compartidos entre agentes; el one-hot de id (ya en la obs) los diferencia.
- `init_hidden(batch)` devuelve ceros.

### `QMixer` (mixing network monótona, solo entrenamiento)

- Hypernetworks que mapean el **estado global** → pesos `W1, W2` (pasados por `abs()` → monotonicidad,
  garantiza `∂Q_tot/∂Q_i ≥ 0`) y biases `b1, b2`.
- Entrada: los N Q-valores de la acción elegida `(B, N)` + estado global `(B, state_dim)`. Salida:
  `Q_tot (B, 1)`.
- Arquitectura estándar QMIX: `|W1|` mezcla los N agentes a `embed_dim`, ELU, `|W2|` a 1, con bias de
  estado (la última capa de bias pasa por una pequeña red de estado).

### `QMIXConfig` (dataclass frozen)

`hidden`, `mixer_embed_dim`, `hypernet_hidden`, `gamma`, `lr`, `batch_episodes`, `buffer_episodes`,
`target_sync`, `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`, `learning_starts_episodes`,
`double_q=True`, `grad_clip`.

### `QMIXLearner`

- Mantiene `AgentRNN` online/target y `QMixer` online/target; un único optimizador Adam sobre los
  parámetros de ambas redes online.
- `select_actions(obs_dict, hidden, epsilon) → (actions_dict, new_hidden)`: ε-greedy **descentralizada**
  por agente (cada agente toma el argmax de su propia Q, compartiendo la red); avanza el hidden de cada
  agente.
- `update(batch) → loss`: desenrolla la GRU sobre T pasos para todos los agentes; calcula `Q_tot` con el
  mixer online sobre la Q de la acción tomada; el target usa **Double-DQN** (la `AgentRNN` online elige la
  acción del siguiente paso, la target la evalúa) + **target mixer**: `y = r + γ(1-done)·Q_tot'`. Huber
  loss **enmascarada** por la longitud real del episodio (`filled`), `clip_grad_norm_(grad_clip)`, sync
  periódico de ambos targets cada `target_sync` updates.

**Recompensa de equipo:** como es un escalar compartido, el batch guarda **un** reward por timestep (no por
agente). El mixer produce un único `Q_tot` por timestep → la pérdida es coherente con el reward de equipo.

## `train.py` — bucle CTDE, replay y reproducibilidad

### `EpisodeReplayBuffer`

Almacena episodios completos con padding a `horizon_t`. Por timestep guarda: obs por-agente
`(T, N, obs_dim)`, acciones `(T, N)`, reward de equipo `(T,)`, estado global `(T, state_dim)`, `done (T,)` y
`filled (T,)` (máscara de padding). `add(episode)` y `sample(batch_episodes, rng) → batch (B, T, ...)`.

### `TrainConfig` (dataclass + `from_yaml`)

Mismo patrón que el Modelo 1: `env: IADSEnvConfig`, `agent: QMIXConfig`, `total_episodes`, `eval_episodes`,
`eval_every`, `device="cpu"`, `seed`, `out_dir="runs/marl_formation"`, `tracking=False`. `from_yaml` hace
`pop("env_config") → IADSEnvConfig.from_yaml` y `pop("agent") → QMIXConfig(**...)`.

### Bucle `train(config) → dict`

1. `_set_seeds` (random/numpy/torch) + RNG propio; instancia env, eval_env, learner, buffer; crea `out_dir`
   y escribe `run_meta.json`.
2. Por episodio: `reset`, `init_hidden`; recolecta hasta `terminated/truncated` con ε decayente (schedule
   por episodios); almacena la trayectoria padeada en el buffer.
3. Tras `learning_starts_episodes` episodios, cada episodio muestrea `batch_episodes` y llama
   `learner.update`.
4. Cada `eval_every` episodios: `_evaluate` → **win-rate** (acción greedy descentralizada, ε=0) +
   `suppressed_fraction` media (logueada); guarda `best.pt` (state_dict de `AgentRNN`, lo único necesario en
   ejecución descentralizada) cuando mejora el win-rate.
5. Al final: perfila latencia de **un agente** con `profile_latency` (reutilizado de
   `temporal_cnn_elint.metrics`, forward de `AgentRNN` con batch=1) → media y p99; escribe `metrics.json`
   (`win_rate`, `suppressed_fraction`, `latency_mean_ms`, `latency_p99_ms`).

### Reproducibilidad y seguridad

- Parámetros solo en `qmix.yaml` versionado (nunca hardcodear hiperparámetros).
- `run_meta.json`: seed, hyperparams completos, hash de config, versiones (python/torch/numpy).
- Determinista por seed (mismo seed → misma curva de entrenamiento).
- No se exponen parámetros de amenazas reales (emisores sintéticos ya en `env.yaml`); si se recarga un
  checkpoint, `torch.load(..., weights_only=True)`.

## Componentes y ficheros

- `src/cog_ew/marl_formation/agents.py` — `AgentRNN`, `QMixer`, `QMIXConfig`, `QMIXLearner`.
- `src/cog_ew/marl_formation/train.py` — `EpisodeReplayBuffer`, `TrainConfig`, `train`.
- `configs/marl_formation/qmix.yaml` — `env_config: configs/marl_formation/env.yaml`, bloque `agent` con
  los campos de `QMIXConfig`, y los de entrenamiento (`total_episodes`, `eval_episodes`, `eval_every`,
  `device`, `seed`, `out_dir`).
- Reutiliza: `IADSFormationEnv`/`IADSEnvConfig` (sub-pieza A), `profile_latency`
  (`temporal_cnn_elint.metrics`).

## Dependencias

Ninguna nueva (PyTorch, NumPy, PyYAML, Gymnasium ya están). El entorno es NumPy puro; las redes son PyTorch.

## Tests (`tests/marl_formation/`)

- `test_agents.py`:
  - `AgentRNN` forward: Q shape `(B, action_dim)`, hidden `(B, hidden)`; determinista por seed.
  - `QMixer`: salida `(B, 1)`; **monotonicidad** `∂Q_tot/∂Q_i ≥ 0` (autograd sobre las Q de agente).
  - `QMIXLearner.select_actions`: dict de N acciones en `[0, action_dim)`; ε=0 determinista, ε=1 explora.
  - `QMIXLearner.update`: batch sintético de episodios → loss finito, parámetros cambian, sync del target.
  - `EpisodeReplayBuffer`: add/sample con shapes y máscara `filled` correctas.
- `test_train.py`:
  - `TrainConfig.from_yaml` parsea env + agent + entrenamiento.
  - `train` smoke test (`total_episodes` pequeño) → `metrics.json` con `win_rate ∈ [0,1]`, latencia finita,
    `best.pt` existe; determinista por seed.

## Fuera de alcance (YAGNI)

- Comparación coordinado vs actuación independiente y el **+45 %** → sub-pieza C.
- Ejecución real / barrido de hiperparámetros en Colab → Fase 6.
- Recurrencia sobre el estado del mixer, atención, comunicación explícita entre agentes, MP-DQN / potencia
  continua.

## Decisiones clave (resumen)

1. **Red de agente GRU (DRQN) con parámetros compartidos** → replay por episodios + BPTT.
2. **QMixer monótono** (hypernetwork condicionada al estado global, pesos por `abs()`).
3. **Double-DQN + target mixer**, Huber loss enmascarada por longitud de episodio, grad clipping.
4. **Recompensa de equipo escalar** → un `Q_tot` por timestep (fully-cooperative).
5. **Ejecución descentralizada** → latencia perfilada = un agente (batch=1), media + p99.
6. **Selección de best.pt por win-rate**; `suppressed_fraction` logueada.
7. **Determinista por seed**, hiperparámetros en YAML versionado, `run_meta.json` con hash + versiones.
