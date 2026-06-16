# Diseño — Agente D3QN y entrenamiento para jamming adaptativo (Modelo 1, sub-pieza B)

**Fecha:** 2026-06-16
**Ámbito:** `src/cog_ew/deep_rl_jamming/{agent,train}.py` + `configs/deep_rl_jamming/train.yaml` + `tests/deep_rl_jamming/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 1 es un agente Deep RL para jamming adaptativo; ancla
cuantitativa **>92 %** de victorias de espectro vs. **58 %** del baseline (Modelo 5), con **latencia <5 ms**.

El Modelo 1 se descompone en A (entorno), B (este spec: agente + entrenamiento) y C (comparación vs baseline).
La **sub-pieza A ya está implementada y mergeada**: `RadarJammingEnv` (Gymnasium) con `observation_space`
`Box(float32, shape=(8, 5))`, `action_space` `Discrete(40)` (técnica × nivel de potencia), `step` que
devuelve `(obs, reward, terminated, truncated, info)` con `info["outcome"] ∈ {ongoing, win, lose}`, y
determinismo por seed.

Estado del arte (`docs/research/estado-del-arte.md` §1): el caso de estudio §1.3 usa **Dueling Double DQN
(D3QN)** para acción discreta; el hueco §1.4 pide **latencia <5 ms documentada** (media + p99) y manejo de
**observabilidad parcial** — ambos materializados aquí (la observación ya es una ventana POMDP de K pasos).

Decisiones tomadas en brainstorming:
- **Algoritmo:** **D3QN propio en PyTorch** (Dueling + Double DQN + replay + ε-greedy). Sin dependencia nueva.
- **Red Q:** **MLP sobre la observación aplanada (8×5=40)** con dos cabezas dueling (V y A). La ventana de
  historial ya viene en la observación, así que no se usa red recurrente.
- **Replay:** **uniforme** (priorizado se anota como mejora futura, YAGNI).
- **"Done" del slice:** agente + bucle entrenable y reproducible; **alcanzar el >92 % real es la Fase 6**
  (Colab). El smoke test verifica que el bucle corre, aprende y vuelca checkpoint + métricas.

## Componentes

### `agent.py`

- **`QNetwork(nn.Module)`** — entrada `obs[B, 8, 5]` → aplanado a `[B, 40]` → tronco MLP `40→hidden→hidden`
  (ReLU) → dos cabezas lineales: `V(s)` (`[B, 1]`) y `A(s, a)` (`[B, n_actions]`). Combinación dueling:
  `Q = V + (A − mean_a A)`. `forward(obs) -> Q[B, n_actions]`. Acepta obs como tensor `float32`.
- **`ReplayBuffer`** — buffer circular uniforme en NumPy de `(obs, action, reward, next_obs, done)`; capacidad
  `buffer_size`; `add(...)` y `sample(batch_size, rng) -> tuple` de tensores. Determinista dado el `rng`.
- **`D3QNConfig`** (dataclass + `from_yaml`) — `n_actions=40`, `obs_dim=40`, `hidden=128`, `gamma=0.99`,
  `lr=1e-3`, `batch_size=64`, `buffer_size=50000`, `target_sync=500` (pasos entre copias a la target),
  `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay_steps=5000`, `learning_starts=1000`, `train_freq=1`.
- **`D3QNAgent`** — `online_net` y `target_net` (`QNetwork`); optimizador Adam.
  - `select_action(obs, epsilon, rng) -> int`: ε-greedy; con prob. ε acción aleatoria (vía `rng`), si no
    `argmax_a Q_online(obs)`. Greedy puro con `epsilon=0`.
  - `update(batch) -> float`: target **Double DQN** —
    `y = r + γ·(1−done)·Q_target(s', argmax_a Q_online(s'))`; pérdida **Huber** entre `Q_online(s, a)` e `y`;
    paso de optimizador; devuelve la pérdida. Sincroniza `target_net ← online_net` cada `target_sync` updates.
  - Determinista por seed (semilla de torch + el `rng` de NumPy inyectado).

### `train.py`

- **`TrainConfig`** (dataclass + `from_yaml`) — incrusta `RadarEnvConfig` (sección `env`) y `D3QNConfig`
  (sección `agent`); añade `total_steps`, `eval_episodes`, `eval_every`, `device`, `seed`, `out_dir`,
  `tracking`.
- **Seeds explícitos** (`random`, `numpy`, `torch`) al inicio; `env.reset(seed=config.seed)`.
- **Bucle de entrenamiento:** por paso, `select_action` con ε interpolada linealmente
  (`epsilon_start → epsilon_end` en `epsilon_decay_steps`), `env.step`, `buffer.add`; tras `learning_starts`
  y cada `train_freq` pasos, `agent.update(buffer.sample(...))`. Al terminar episodio, `env.reset()` y
  registrar `outcome`.
- **Evaluación periódica:** cada `eval_every` pasos, `eval_episodes` episodios en modo greedy (ε=0); **win
  rate** = fracción de `info["outcome"] == "win"`. Guarda el **mejor checkpoint** (`best.pt`) por win rate.
- **Perfilado de latencia:** al final, `profile_latency(agent.online_net, sample_obs, n_warmup, n_iter,
  device)` (reutilizado de `cog_ew.temporal_cnn_elint.metrics`) → media + p99 en ms, frente a **<5 ms**.
- **Logueo de reproducibilidad incondicional:** `run_meta.json` con seed, hiperparámetros completos
  (`dataclasses.asdict`), versiones (python/torch/numpy/gymnasium) y hash de la config de entrenamiento.
  `metrics.json` con win rate final y latencias. `trackio` opcional por flag `tracking`.
- **`train(config) -> dict[str, Any]`**: devuelve `{"win_rate_history": [...], "final": {...}}` con win rate
  final, `latency_mean_ms` y `latency_p99_ms`.

### `configs/deep_rl_jamming/train.yaml`

Secciones anidadas `env` (mismos parámetros que `env.yaml`) y `agent` (hiperparámetros de `D3QNConfig`), más
los de entrenamiento. Nada hardcodeado.

## Reproducibilidad

- Parámetros solo en `train.yaml` versionado.
- Seeds explícitos en torch/numpy/random; entorno y replay deterministas por seed.
- `run_meta.json` incondicional (no depende de tracking).
- `trackio` opcional y guardado por flag → tests sin red.

## Dependencias

Ninguna nueva (PyTorch, NumPy, Gymnasium ya están). Reutiliza `profile_latency` de
`cog_ew.temporal_cnn_elint.metrics`.

## Tests (`tests/deep_rl_jamming/`)

- `test_agent.py`:
  - `QNetwork.forward` devuelve `Q[B, 40]`.
  - combinación dueling correcta: con `V` y `A` fijados (pesos controlados o un caso construido),
    `Q == V + (A − mean(A))`.
  - `select_action` con `epsilon=0` devuelve `argmax_a Q`; con `epsilon=1` y un `rng` sembrado es determinista
    y dentro de `[0, 40)`.
  - `ReplayBuffer` `add`/`sample` con shapes correctas y muestreo determinista por `rng`.
  - `update` reduce la pérdida sobre un batch sintético repetido.
- `test_train.py`:
  - `TrainConfig.from_yaml` parsea las secciones `env`/`agent` anidadas.
  - **smoke test** end-to-end: config minúscula (`total_steps` bajo, buffer pequeño) corre, escribe `best.pt`,
    `metrics.json` y `run_meta.json`; `train()` devuelve `win_rate_history` y latencias (> 0).
  - determinismo: dos `train()` con el mismo seed dan el mismo `win_rate_history`.

## Fuera de alcance (YAGNI)

- Ejecución real en Colab y alcanzar el >92 % (Fase 6, `/experiment-run`).
- Arnés de comparación vs baseline Modelo 5 (sub-pieza C).
- Replay priorizado, Wolpertinger, acción de banda/frecuencia, multi-agente.
- Convolución/GRU temporal en la red (la ventana ya está en la observación).

## Decisiones clave (resumen)

1. **D3QN propio en PyTorch** (Dueling + Double DQN + replay uniforme + ε-greedy).
2. **Red Q = MLP sobre obs aplanada + cabezas dueling** (V, A) → `Q = V + (A − mean A)`.
3. **Bucle con evaluación por win rate** (greedy) y **mejor checkpoint** por win rate.
4. **Perfilado de latencia** (media + p99) reutilizando `profile_latency`, objetivo <5 ms.
5. **Reproducibilidad**: seeds, `train.yaml` versionado, `run_meta.json` incondicional, `trackio` opcional.
6. **Slice entrenable**; el >92 % real y la comparación vs baseline son Fase 6 / sub-pieza C.
