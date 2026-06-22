# Fase 6 · Sub-pieza A — Arnés unificado de experimentos + reporte de anclas Q1

> Fecha: 2026-06-22 · Estado: aprobado · Fuente de verdad del *qué*: `Propuesta.md` (anclas Q1).
> Depende de: los 5 modelos completos y mergeados (M1–M5).

## Contexto y objetivo

Los 5 modelos están implementados y mergeados. La **Fase 6** produce las cifras ancla Q1 mediante
entrenamiento real en Colab/GPU y redacta el paper. La **sub-pieza A** construye la **infraestructura
reproducible** que orquesta el pipeline de cada modelo entrenable y agrega sus métricas ancla en un
único reporte — la pieza testeable ahora (modo `quick`/CPU), lista para correr a escala en Colab.

**Anclas Q1 (Propuesta.md):**

| Ancla | Modelo | Pipeline | Métrica | Objetivo |
|------|--------|----------|---------|----------|
| `jamming` | M1 Deep RL | `train` D3QN → `compare` cognitivo vs librería | `cognitive.win_rate` | ≥ 0.92 (baseline ≈ 0.58) |
| `elint` | M2 Temporal CNN | `train` | `lpi_accuracy` (y `macro_acc_type`) | ≥ 0.96 |
| `marl` | M3 MARL | `train` qmix + `train` iql → `compare_policies` | `relative_improvement.suppressed_fraction` | ≥ 0.45 |
| `gan` | M4 GAN | `train` GAN → `export_synthetic` → `run_robustness_experiment` | `relative_improvement` | ≥ 0.22 |

### Fuentes de cada métrica (verificadas en el código)

- **M1:** `deep_rl_jamming.compare.compare(env, cognitive, baseline, episodes, seed)` →
  `{"cognitive": {"win_rate", ...}, "baseline": {"win_rate", ...}, "delta": {...}}`. La policy cognitiva
  envuelve el agente D3QN entrenado; la baseline es una `LibraryPolicy` sobre `EWResponseLibrary`.
- **M2:** `temporal_cnn_elint.train.train(config)` → `{"test": {"macro_acc_type", "macro_acc_mode",
  "lpi_accuracy", ...}}`.
- **M3:** `marl_formation.compare.compare_policies(env, coordinated=, independent=, episodes, seed)` →
  `{..., "relative_improvement": {"suppressed_fraction"}}`. Coordinado = `AgentPolicy` del checkpoint
  QMIX; independiente = `AgentPolicy` del checkpoint IQL (ambos vía
  `AgentPolicy.from_checkpoint`).
- **M4:** `gan_signals.robustness.run_robustness_experiment(config)` → `{"baseline", "augmented",
  "delta", "relative_improvement", "global"}`. Requiere antes `gan_signals.train.train` (escribe
  `best.pt`) y `gan_signals.export.export_synthetic` (escribe el HDF5).

## Decisiones de diseño fijadas

1. **Baseline del ancla `marl`:** coordinado **QMIX** vs independiente **IQL** (ambos aprendidos →
   aísla el efecto de la coordinación, alineado con "coordinando 4 aeronaves vs actuación
   independiente" de la Propuesta).
2. **El arnés orquesta + agrega:** ejecuta el pipeline de cada modelo (que escribe su propio
   `run_meta`/`metrics.json`) y agrega las anclas en un `anchors_report.json`.
3. **Perfiles `quick` / `full`:** `quick` = configs diminutas (CPU, para testear el arnés; las anclas
   **no** alcanzan el objetivo); `full` = configs reales de `configs/` (Colab GPU). El trabajo del
   arnés es producir el reporte de forma reproducible, no aprobar las anclas.

## Arquitectura

Paquete nuevo **`src/cog_ew/experiments/`** (importable y testeable) + un notebook fino de entrada.

### `src/cog_ew/experiments/anchors.py`

- `@dataclass(frozen=True) AnchorResult`: `name: str`, `target: float`, `achieved: float`,
  `baseline: float | None`, `passed: bool`, `run_dir: str`.
- Cuatro funciones runner, cada una ejecuta su pipeline en un `out_dir` propio y devuelve un
  `AnchorResult`:
  - `run_jamming_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult`
  - `run_elint_anchor(profile, out_dir) -> AnchorResult`
  - `run_marl_anchor(profile, out_dir) -> AnchorResult`
  - `run_gan_anchor(profile, out_dir) -> AnchorResult`
- `passed = math.isfinite(achieved) and achieved >= target` (todas las anclas son "mayor o igual";
  el guard de finitud evita que un `relative_improvement == inf` —baseline 0 en `quick`— apruebe de
  forma vacua).

### `src/cog_ew/experiments/report.py`

- `ExperimentProfile` (dataclass + `from_yaml`): selecciona las configs concretas por modelo
  (rutas a los YAML reales en perfil `full`, o overrides diminutos en `quick`), `device`, `seed`,
  `episodes`/`epochs` reducidos en `quick`.
- `ANCHOR_RUNNERS: dict[str, Callable]` mapea `name → runner`.
- `run_anchors(names: tuple[str, ...], profile: ExperimentProfile, out_dir: Path) -> dict[str, Any]`:
  ejecuta los runners seleccionados, agrega los `AnchorResult` y escribe
  `out_dir/anchors_report.json` con: por ancla `{target, achieved, baseline, passed, run_dir}`, más
  metadatos de reproducibilidad (`profile_name`, `seed`, `config_hash`, versiones de dependencias).
  Devuelve el dict del reporte.

### `configs/experiments/{quick,full}.yaml`

Perfiles versionados. `quick` reduce drásticamente episodios/épocas/tamaños y usa `device: cpu`;
`full` apunta a las configs reales por modelo y `device: cuda`. Hiperparámetros solo en YAML.

### `notebooks/run_anchors.py`

Entrada fina para Colab: parsea `--profile {quick,full}`, `--anchors {all|jamming,elint,...}`,
`--out-dir`, monta Drive opcionalmente, y llama `experiments.report.run_anchors`. Sustituye/eleva el
`notebooks/colab_train_models.py` actual (que solo cubría M1–M3 parcialmente). El runner antiguo se
elimina para no dejar dos caminos divergentes.

### GPU-readiness (prerrequisito transversal)

Añadir `torch.cuda.manual_seed_all(seed)` al `_set_seeds` de los modelos entrenables
(`deep_rl_jamming`, `temporal_cnn_elint`, `marl_formation`, `gan_signals/train`,
`gan_signals/robustness`) para reproducibilidad en runs GPU. Cambio mínimo y aislado.

## Reproducibilidad y salidas

- Seeds explícitos por perfil; cada pipeline ya escribe su `run_meta.json`/`metrics.json` (el reporte
  los referencia por `run_dir`).
- `anchors_report.json`: anclas + `profile_name`, `seed`, `config_hash` (sha256 del perfil), versiones
  de dependencias (python/torch/numpy).
- `weights_only=True` en toda carga de checkpoint (M3 `AgentPolicy.from_checkpoint`, M4 `load_generator`
  ya lo cumplen).
- No se exponen parámetros de amenazas reales (solo catálogos sintéticos del proyecto).

## Plan de tests (TDD, perfil `quick`/CPU)

1. `run_jamming_anchor` (config diminuta) devuelve un `AnchorResult` con `0 ≤ achieved ≤ 1`,
   `baseline` poblado, `name == "jamming"`, y `run_dir` existente.
2. `run_elint_anchor` devuelve `AnchorResult` con `achieved` en [0,1] (`lpi_accuracy`).
3. `run_marl_anchor` entrena qmix+iql diminutos y devuelve `AnchorResult` con `achieved` finito
   (relative_improvement; puede ser `inf` si el baseline es 0 — el arnés lo trata como no-pass).
4. `run_gan_anchor` corre el pipeline GAN→export→robustness diminuto y devuelve `AnchorResult`
   con `achieved` finito.
5. `run_anchors(("elint",), profile_quick, tmp)` escribe `anchors_report.json` con la clave `elint` y
   los metadatos de reproducibilidad; `passed` consistente con `achieved >= target`.
6. `run_anchors` con todas las anclas agrega las 4 entradas.
7. Reproducible por seed: dos ejecuciones del mismo perfil dan `achieved` idénticos para un ancla
   determinista (p. ej. `elint`).

## Fuera de alcance de la sub-pieza A

- La ejecución GPU real a escala que produce las cifras finales (6B, ejecutada por el usuario en Colab).
- La redacción del paper Q1 (6C).
- `relative_improvement == inf` en `quick` no es un fallo del arnés: con entrenamientos diminutos los
  baselines pueden ser 0; el reporte registra `achieved=inf` y, por el guard de finitud, `passed=False`.
- El Modelo 5 (baseline EW) no tiene ancla propia: es la referencia de comparación dentro de A1.
