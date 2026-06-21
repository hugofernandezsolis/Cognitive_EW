# Modelo 4 · Sub-pieza C — Evaluación de robustez (+22%)

> Fecha: 2026-06-21 · Estado: aprobado · Fuente de verdad del *qué*: `Propuesta.md` (Modelo 4).
> Depende de: sub-pieza A (cWGAN-GP) y sub-pieza B (export sintético HDF5), ambas mergeadas; y del
> Modelo 2 (Temporal CNN ELINT), completo y mergeado.

## Contexto y objetivo

La sub-pieza C cierra el Modelo 4: demuestra que los datos sintéticos generados por el cWGAN-GP
**mejoran la robustez del Modelo 2 frente a sistemas no catalogados**. Es la métrica ancla del modelo.

**Ancla (Propuesta.md):** la GAN, generando 50+ tipos (incluyendo sistemas nunca catalogados), mejora
la robustez del clasificador (Modelo 2) un **+22 %** frente a sistemas no catalogados.

### Artefactos y restricciones heredadas

- **Modelo 2 (`TemporalCNN`, `cog_ew.temporal_cnn_elint.model`)**: dos cabezas — `type` (8 emisores) y
  `mode` (4 modos: search/tws/track/missile_guidance); `threat` se deriva de `mode`. Entrena con CE
  conjunta type+mode sobre `PDWSyntheticDataset`; métricas `macro_accuracy`, `lpi_accuracy`.
- **HDF5 sintético de B** (`data/synthetic/<name>.h5`): datasets `X (N,10,64)`, `type_id`, `source_a`,
  `source_b`, `alpha`, `is_known`. **No tiene etiqueta de `mode`** (la GAN se condicionó por emisor, no
  por modo). → Los sintéticos solo pueden supervisar la cabeza de **tipo**.
- Las etiquetas de tipo son **índices globales de emisor** consistentes entre el dataset real
  (`PDWSyntheticDataset`, donde `type_idx = library.emitter_names().index(name)`) y el `source_a` del
  HDF5 sintético (`build_type_catalog` con `n_emitters=8` → ids 0–7 globales). Encajan sin remapeo.
- `mode` es un label compartido entre emisores; `type` es closed-set (8 clases): un emisor retenido no
  es predecible como tipo salvo que M2 lo haya visto (real o sintético).
- `PDWConfig.emitters` (tupla de nombres) filtra `PDWSyntheticDataset` por emisor → permite construir el
  set real catalogado (no retenidos) y el set real de test (retenidos).

## Decisiones de diseño fijadas

1. **Experimento:** leave-emitters-out + reconocimiento de tipo.
2. **Supervisión de los sintéticos:** solo la cabeza de tipo; su `mode = -1` y la CE de modo usa
   `ignore_index=-1` (los reales aportan type+mode, los sintéticos solo type).
3. **Métrica ancla:** `macro_acc_type` sobre las señales **reales de los emisores retenidos**, baseline
   vs aumentado.
4. **Emisores retenidos por defecto:** los 2 LPI (`LPI-FMCW`, `LPI-polyphase`) — radares cognitivos no
   catalogados (config).
5. **Aislamiento:** la única diferencia entre baseline y aumentado son los sintéticos de los emisores
   retenidos (`augment_held_out_only=true`, conmutable).

## Experimento de robustez

- **Emisores retenidos** (`held_out`, config, **nombres**): sus señales reales se excluyen del
  entrenamiento y solo aparecen en el set de test. El harness resuelve los nombres a **ids globales**
  vía `library.emitter_names().index(name)`: usa nombres para filtrar `PDWConfig.emitters` (que filtra
  por nombre) e ids para filtrar `SyntheticPDWDataset` (que filtra por `source_a`).
- **Real catalogado (train/val):** `PDWSyntheticDataset` filtrado a los emisores NO retenidos (por
  nombre), partido train/val con `split_dataset`.
- **Real retenido (test):** `PDWSyntheticDataset` filtrado a los emisores retenidos (por nombre).
- **Sintético de aumento:** `SyntheticPDWDataset` del HDF5 de B, filtrado a los emisores retenidos
  (`is_known=True`, `source_a ∈ held_out`), etiqueta `type = source_a`, `mode = -1`.
- **Baseline:** M2 entrenado sobre el real catalogado.
- **Aumentado:** M2 entrenado sobre real catalogado + sintético de aumento, con la **misma seed** (la
  única diferencia es la data sintética).
- **Evaluación (ancla):** `macro_acc_type` sobre el real retenido para baseline y aumentado → `delta`
  y `relative_improvement = (aug − base) / base` (guard `base == 0 → inf`, como en el harness del
  Modelo 3).
- **Contexto (`global`):** `macro_acc_type` sobre un set de evaluación de todos los tipos = (split de
  validación del catalogado) ∪ (real retenido), para baseline y aumentado → dict
  `{"baseline", "augmented"}`. Da una vista global (8 clases) sin necesidad de un test split extra del
  catalogado.

## Módulos

### `src/cog_ew/data/synthetic_loader.py`

- `SyntheticPDWDataset(hdf5_path, *, emitters: tuple[int, ...] | None = None, known_only: bool = True)`:
  `torch.utils.data.Dataset`. Lee `X`, `source_a`, `is_known` del HDF5; filtra por `is_known` (si
  `known_only`) y por `source_a ∈ emitters` (si `emitters` no es `None`); `__getitem__` devuelve
  `(x: torch.Tensor (10,64), type: int = source_a, mode: int = -1, threat: int = -1)` — mismo 4-tuple
  que `PDWSyntheticDataset`, para reutilizar el bucle de entrenamiento.

### `src/cog_ew/gan_signals/robustness.py`

- `RobustnessConfig` (dataclass + `from_yaml`): `synthetic_path: str`, `library_path: str`,
  `held_out: tuple[str, ...]`, `model: TemporalCNNConfig`, `pdw: PDWConfig` (para el real),
  `augment_held_out_only: bool = True`, `epochs`, `batch_size`, `lr`, `weight_decay`, `seed`,
  `device`, `out_dir`.
- `_fit_classifier(model_config, train_ds, val_ds, *, hp..., device) -> TemporalCNN`: bucle de
  entrenamiento type+mode con `F.cross_entropy(mode_logits, y_mode, ignore_index=-1)`; selección por
  mejor val; mirror del bucle del Modelo 2 con la máscara de modo.
- `evaluate_type_accuracy(model, test_ds, n_types, device) -> float`: `macro_acc_type` sobre `test_ds`.
- `run_robustness_experiment(config) -> dict[str, Any]`: construye los datasets (real catalogado,
  real retenido, sintético de aumento), entrena baseline y aumentado con la misma seed, evalúa, y
  devuelve `{"baseline", "augmented", "delta", "relative_improvement", "global"}`; escribe
  `run_meta.json` + `metrics.json`.

### `configs/gan_signals/robustness.yaml`

Hiperparámetros solo en YAML: `synthetic_path`, `library_path`, `held_out`, bloque `model`, bloque
`pdw`, `augment_held_out_only`, `epochs`, `batch_size`, `lr`, `weight_decay`, `seed`, `device`,
`out_dir`. **Valores por defecto sugeridos:** `held_out=[LPI-FMCW, LPI-polyphase]`,
`augment_held_out_only=true`, `epochs=30`, `batch_size=64`, `lr=1e-3`, `seed=0`, `device=cpu`,
`synthetic_path=data/synthetic/wgan_gp.h5`, `library_path=configs/temporal_cnn_elint/emitters.yaml`.

## Reproducibilidad y salidas

- `_set_seeds(seed)` (random/numpy/torch); baseline y aumentado con la misma seed.
- `run_meta.json`: seed, hiperparámetros (`asdict`), `config_hash`, `synthetic_hash` (sha256 del HDF5),
  versiones (python/torch/numpy/h5py).
- `metrics.json`: `baseline`, `augmented`, `delta`, `relative_improvement`, `global`, latencia del M2.
- `weights_only=True` en toda carga; no se exponen parámetros de amenazas reales (solo catálogo
  sintético + checkpoints entrenados sobre él).

## Plan de tests (TDD)

1. `SyntheticPDWDataset` filtra por emisor y `is_known`, y `__getitem__` devuelve el 4-tuple con
   `type == source_a`, `mode == -1`, `threat == -1`, `x` de forma `(10,64)`.
2. `SyntheticPDWDataset` con `emitters=(6,7)` solo devuelve muestras de esos emisores; longitud
   coherente.
3. `_fit_classifier` reduce el loss y deja un `TemporalCNN`; un batch puramente sintético
   (`mode=-1`) no propaga gradiente a `head_mode` (la máscara `ignore_index` funciona).
4. `evaluate_type_accuracy` sobre un set conocido devuelve un float en [0,1].
5. `run_robustness_experiment` (config diminuta) devuelve `{"baseline","augmented","delta",
   "relative_improvement","global"}`; `delta == augmented − baseline`;
   `relative_improvement` consistente con el guard `base==0 → inf`; escribe `run_meta.json` y
   `metrics.json`.
6. Reproducible por seed: dos ejecuciones con la misma config dan métricas idénticas.

## Fuera de alcance de la sub-pieza C

- El entrenamiento real en Colab (muchas épocas, GPU) que produce el +22 % numérico final (Fase 6).
- Regenerar o reetiquetar los sintéticos con `mode` (requeriría recondicionar la GAN — sub-pieza A).
- Los tipos interpolados `is_known=False` (sin etiqueta de tipo única) no se usan en esta supervisión.
- Detección de novedad open-set / umbrales de rechazo (experimento alternativo descartado).
