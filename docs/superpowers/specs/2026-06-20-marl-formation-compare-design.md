# Diseño — Comparación QMIX coordinado vs learners independientes (Modelo 3, sub-pieza C)

**Fecha:** 2026-06-20
**Ámbito:** `src/cog_ew/marl_formation/{agents.py,train.py,compare.py}` + `configs/marl_formation/iql.yaml` + `tests/marl_formation/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 3 es **Multi-Agent RL para coordinación de EW en
formación**. Ancla cuantitativa: **mejora la supresión del IADS un 45 % coordinando 4 aeronaves frente a
actuación independiente**.

El Modelo 3 se descompone en A (entorno IADS ✅ mergeado), B (agentes QMIX + entrenamiento CTDE ✅
implementado en rama) y **C (comparación coordinado vs actuación independiente, este spec)**.

Estado del arte (`docs/research/estado-del-arte.md` §3): la ablación canónica del paper de QMIX es **QMIX vs
Independent Q-Learning (IQL)**; el caso MA-CJD contrasta jamming cooperativo coordinado frente a aprendices
independientes. Esta sub-pieza modela la **"actuación independiente"** del ancla como **learners
independientes (IQL)** y el régimen coordinado como el **QMIX conjunto** de la sub-pieza B.

Decisiones tomadas en brainstorming:
- **Baseline "independiente" = IQL** (no rule-based): cada agente aprende su propia política sin mezcla
  centralizada.
- **IQL = mismo `AgentRNN` compartido, sin `QMixer` ni estado global, pérdida Double-DQN por agente** sobre
  la recompensa de equipo. Aísla exactamente lo que aporta la mezcla centralizada.
- **Ejecución idéntica en ambos regímenes** (AgentRNN por agente, greedy descentralizado) → la comparación
  es entre dos checkpoints; el contraste es solo de entrenamiento.
- **Ancla expresada como mejora relativa** sobre `suppressed_fraction`: `(coord − indep) / indep`.

## Componentes y ficheros

Cuatro piezas en `src/cog_ew/marl_formation/`:

### 1. `IQLLearner` (en `agents.py`, hermana de `QMIXLearner`)

- **Constructor:** `IQLLearner(obs_dim, action_dim, n_agents, config, device, rng)`. **No recibe
  `state_dim`** (no hay estado global ni mixer). Reutiliza `QMIXConfig` (los campos `mixer_embed_dim` y
  `hypernet_hidden` quedan inertes — evita un dataclass casi idéntico).
- **Redes:** `AgentRNN` online + target con **parámetros compartidos** (igual que QMIX). Optimizador Adam
  **solo** sobre los parámetros del `AgentRNN`.
- **`init_hidden() -> dict[int, Tensor]` y `select_actions(obs_dict, hidden, epsilon) ->
  tuple[dict[int,int], dict[int,Tensor]]`:** idénticos a `QMIXLearner` (ε-greedy descentralizada por agente).
  Garantiza ejecución indistinguible entre regímenes.
- **`update(batch) -> float`:** desenrolla la GRU sobre T pasos para los N agentes (igual que QMIX), pero
  **sin mezclar**. Para cada agente i, Double-DQN sobre la recompensa de equipo broadcast `r`:
  `y_i = r + γ(1−done)·Q_target_i(next, argmax_a Q_online_i(next))`. Pérdida = media sobre agentes de
  `Huber(Q_online_i(chosen), y_i)`, **enmascarada por `filled`** (longitud de episodio), `clip_grad_norm_`,
  sync periódico del target cada `target_sync`. El `batch` tiene el mismo formato que QMIX (incluye `states`,
  que IQL **ignora**), para reutilizar `EpisodeReplayBuffer` sin cambios.

La única diferencia con `QMIXLearner.update`: donde QMIX hace `Q_tot = mixer(chosen, state)` y un TD sobre
`Q_tot`, IQL hace el TD por agente y promedia, sin tocar el estado global.

### 2. Generalización del bucle de entrenamiento (`train.py`)

- `TrainConfig` gana `regime: str = "qmix"` (valores `"qmix"` | `"iql"`); `from_yaml` lo lee del YAML.
- Función interna `_build_learner(regime, env, config, device, rng)` que instancia `QMIXLearner` (con
  `state_dim`) o `IQLLearner` (sin `state_dim`). El resto del bucle (`_rollout`, `_evaluate`, checkpoint por
  `(win_rate, suppressed_fraction)`, perfilado de latencia, `run_meta`/`metrics`) queda **idéntico**: ya
  opera contra la interfaz común `{init_hidden, select_actions, update}`.
- `run_meta.json` registra el `regime` (qué régimen produjo cada checkpoint).
- `out_dir` por defecto puede incluir el régimen (`runs/marl_formation/{regime}`) o fijarse en cada YAML para
  no pisar checkpoints.

### 3. `compare.py` (reescritura)

- **`FormationPolicy` (Protocol):** `reset(env)` + `act(env, obs, state, info) -> dict[int,int]`.
- **`AgentPolicy`** (generaliza el anterior `QMIXPolicy`; renombrado porque ya no es específico de QMIX):
  envuelve un `AgentRNN`, ejecución greedy descentralizada con hidden por agente.
  `from_checkpoint(path, *, obs_dim, action_dim, hidden, n_agents, device)` con
  `torch.load(..., weights_only=True)`. Sirve para ambos regímenes (carga `qmix/best.pt` o `iql/best.pt`).
- **Baselines rule-based deterministas** (conservadas como sanity sin-entrenamiento y fixtures de test):
  `ConcentratedSuppressionPolicy` (todos los agentes al mismo radar, supresión a máxima potencia) y
  `SpreadSuppressionPolicy` (reparto round-robin de radares).
- **`evaluate_policy(env, policy, *, episodes, seed) -> dict[str,float]`:** `win_rate`, `mean_reward`,
  `mean_steps`, `suppressed_fraction`. Determinista por seed (solo el primer `reset` lleva seed).
- **`compare_policies(env, *, coordinated, independent, episodes, seed) -> dict`:** devuelve
  `{"coordinated": {...}, "independent": {...}, "delta": {...}, "relative_improvement": {...}}`.
  - `delta`: deltas absolutos (`win_rate`, `mean_reward`, `suppressed_fraction`).
  - `relative_improvement["suppressed_fraction"] = (coord − indep) / indep`, **la cifra del +45 %**. Guarda
    para `indep == 0`: devuelve `float("inf")` (documentado).
  - Comparación canónica: `coordinated = AgentPolicy(qmix/best.pt)` vs `independent = AgentPolicy(iql/best.pt)`.

### 4. Config `configs/marl_formation/iql.yaml`

Clon de `qmix.yaml` con `regime: iql`, `out_dir: runs/marl_formation/iql` y el mismo bloque `agent` (campos
del mixer inertes). `qmix.yaml` gana `regime: qmix` explícito.

## Reproducibilidad y seguridad

- Parámetros solo en YAML versionado; determinista por seed.
- `run_meta.json` registra `regime` + seed + hyperparams + hash + versiones.
- `torch.load(..., weights_only=True)` al recargar checkpoints.
- No se exponen parámetros de amenazas reales (emisores sintéticos del `env.yaml`).

## Dependencias

Ninguna nueva (PyTorch, NumPy, PyYAML, Gymnasium ya presentes). Reutiliza `AgentRNN`, `QMIXConfig`,
`EpisodeReplayBuffer`, `IADSFormationEnv` y `profile_latency`.

## Tests (`tests/marl_formation/`)

Ninguno entrena de verdad ni aserta el +45 % (eso es Fase 6); validan mecánica y contrato.

- `test_agents.py` (añade IQL junto a QMIX):
  - `IQLLearner.select_actions`: N acciones válidas en `[0, action_dim)`; ε=0 determinista, ε=1 explora.
  - `IQLLearner.update`: batch sintético → loss finita, parámetros del `AgentRNN` cambian, sync del target.
  - `IQLLearner` entrena sin estado global (constructor sin `state_dim`; `update` ignora `states`).
- `test_compare.py` (reescrito):
  - Baselines rule-based: concentrada apunta todos al mismo radar; reparto round-robin distribuye.
  - `AgentPolicy`: acciones válidas; `from_checkpoint` carga pesos (`weights_only=True`).
  - `evaluate_policy`: métricas acotadas y finitas; determinista por seed.
  - `compare_policies`: estructura `{coordinated, independent, delta, relative_improvement}`; delta absoluto
    = resta; `relative_improvement` = `(coord − indep)/indep`; guarda de `indep == 0`.
  - Sanity de coordinación con baselines rule-based: el reparto cubre ≥ que la concentración (sin entrenar).
- `test_train.py`: smoke test del régimen `iql` (pocos episodios → `best.pt` + `metrics.json`; determinista
  por seed), en paralelo al de QMIX.

## Fuera de alcance (YAGNI)

- Ejecución real / barrido de hiperparámetros y el valor numérico del +45 % → Fase 6 (Colab).
- VDN u otros mezcladores intermedios; comunicación explícita entre agentes; potencia continua.

## Decisiones clave (resumen)

1. **"Actuación independiente" = IQL** (learners independientes), no rule-based.
2. **IQL = `AgentRNN` compartido sin `QMixer`, Double-DQN por agente** sobre recompensa de equipo; reutiliza
   `QMIXConfig` y `EpisodeReplayBuffer`.
3. **Ejecución idéntica en ambos regímenes** → comparación entre dos checkpoints con la misma `AgentPolicy`.
4. **Un bucle de entrenamiento, dos regímenes** (`regime` en `TrainConfig`) → DRY.
5. **Ancla = mejora relativa** `(coord − indep)/indep` sobre `suppressed_fraction`.
6. **Baselines rule-based conservadas** como sanity sin-entrenamiento y fixtures de test.
7. **El valor real del +45 %** sale de entrenar ambos regímenes en Colab (Fase 6); los tests validan mecánica.
