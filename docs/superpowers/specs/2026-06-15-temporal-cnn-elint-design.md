# Diseño — Temporal CNN multi-tarea para clasificación ELINT (Modelo 2)

**Fecha:** 2026-06-15
**Ámbito:** `src/cog_ew/temporal_cnn_elint/` (model, metrics, train) + `configs/temporal_cnn_elint/train.yaml` + `tests/temporal_cnn_elint/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 2 es una **Temporal CNN** que procesa la secuencia de pulsos
interceptada por el RWR y clasifica en tiempo real, **multi-tarea**: (1) **tipo de emisor**, (2) **modo de
operación** y (3) **estado de amenaza**, con **latencia <1 ms**, **incluyendo radares LPI**. Contribución:
accuracy >96 % incl. LPI (vs. <65 % convencional), y reportar **latencia (media + p99)** en hardware fijo —
los trabajos publicados dan accuracy pero no tiempos (el estado del arte parte de ~15 ms en GTX 1060, ver
`docs/research/estado-del-arte.md` §2.2).

Este spec consume el pipeline de datos PDW ya implementado (`PDWSyntheticDataset`, que entrega
`(Tensor[10,64], type, mode, threat)`). Cubre el **slice entrenable completo** del Modelo 2: modelo,
métricas, perfilado de latencia y bucle de entrenamiento. La **ejecución real** del entrenamiento en Colab
es un paso posterior (skill `/experiment-run`).

Decisiones tomadas en brainstorming:
- **Arquitectura:** TCN dilatada con **backbone único compartido + cabezas** (la más fiel a "Temporal CNN"
  de Propuesta.md y la más alineada con el objetivo <1 ms; descartadas CNN 1D simple por campo receptivo
  corto y multi-stream por coste de latencia).
- **Amenaza derivada del modo:** dado que en el pipeline PDW `threat = mode_to_threat(mode)` es determinista,
  el modelo entrena **2 cabezas** (tipo, modo) y la amenaza se obtiene en inferencia aplicando
  `mode_to_threat` al modo predicho → 3 salidas expuestas (como pide Propuesta.md) y amenaza **siempre
  coherente** con el modo, sin redundancia en el loss.

## Representación y flujo de datos

```
PDWConfig → PDWSyntheticDataset → split_dataset(train/val/test, seed) → DataLoader
  → TemporalCNN → (type_logits[B,8], mode_logits[B,4])
  → loss = w_type·CE(type) + w_mode·CE(mode)
  → eval (test): macro-accuracy por cabeza + accuracy LPI + matriz de confusión + latencia (media, p99)
```

Entrada al modelo: tensor `[B, C=10, L=64]` (channels-first, 5 features continuas normalizadas + 5 canales
one-hot de modulación intra-pulso). Etiquetas: `type` (8 clases), `mode` (4 clases); `threat` (4 niveles)
se deriva.

## Componentes

### `model.py` — `TemporalCNN` + `TemporalCNNConfig`

Backbone TCN dilatado (campo receptivo que cubre 64 posiciones con pocas capas → red ligera):

```
x: [B, 10, 64]
  stem:   Conv1d(10→H, k=3, pad=same) + BatchNorm1d + GELU
  blocks: R bloques residuales TCN con dilations cíclicas [1, 2, 4, 8]:
          Conv1d(H,H,k=3,dil=d,pad=same) + BN + GELU
          + Conv1d(H,H,k=3,dil=d,pad=same) + BN
          + skip (residual) → GELU
  pool:   GlobalAvgPool1d sobre el tiempo → feat[H]
  heads:  Linear(H→8)  (tipo)
          Linear(H→4)  (modo)
```

- `forward(x) -> tuple[Tensor, Tensor]`: devuelve `(type_logits, mode_logits)`.
- `predict(x) -> tuple[Tensor, Tensor, Tensor]`: `type = argmax(type_logits)`, `mode = argmax(mode_logits)`,
  `threat = THREAT_FROM_MODE[mode]`, donde `THREAT_FROM_MODE` es un **buffer constante** (registrado con
  `register_buffer`) construido desde `mode_to_threat` sobre `MODES` → amenaza coherente por construcción.
- **Padding "same", no causal:** la ventana de 64 pulsos está completa en inferencia (clasificación, no
  streaming), así que la convolución no necesita ser causal. Se anota como opción futura para streaming.
- `TemporalCNNConfig` (dataclass + `from_yaml`): `in_channels=10`, `seq_len=64`, `hidden=64`,
  `n_blocks=4`, `dilations=(1,2,4,8)`, `n_types=8`, `n_modes=4`, `dropout`. Objetivo ~100–200k params.

### `metrics.py` — funciones puras (sin I/O ni estado)

- `macro_accuracy(preds, targets, num_classes) -> float`: media del recall por clase (métrica correcta con
  clases potencialmente desbalanceadas; SoA §2.3). Clases sin soporte se omiten del promedio.
- `confusion_matrix(preds, targets, num_classes) -> Tensor[num_classes, num_classes]`.
- `lpi_accuracy(type_preds, type_targets, lpi_indices) -> float`: accuracy restringida a los pulsos cuyo
  tipo real es un emisor LPI. `lpi_indices` se deriva de la librería de emisores (los que declaran modos
  con `lpi=True`). Sostiene la métrica ">96 % incl. LPI".
- `profile_latency(model, sample, *, n_warmup, n_iter, device) -> tuple[float, float]`: devuelve
  **(media, p99) en milisegundos** midiendo inferencia con **batch=1** (real-time es por ventana). Hace
  `n_warmup` iteraciones de calentamiento, sincroniza CUDA si el device es GPU, y mide `n_iter` pasadas.
  Métrica ancla de Propuesta.md (<1 ms).

### `train.py` — `TrainConfig` + bucle + evaluación final

- `TrainConfig` (dataclass + `from_yaml`): incrusta `PDWConfig` (datos) y `TemporalCNNConfig`; añade
  `splits=(0.7,0.15,0.15)`, `batch_size`, `epochs`, `lr`, `weight_decay`, `loss_weights=(w_type,w_mode)`,
  `device`, `seed`, `out_dir`, `tracking: bool`.
- **Seeds explícitos** al inicio: `torch.manual_seed`, `numpy.random.seed`, `random.seed`.
- Construye `PDWSyntheticDataset(config.data)` → `split_dataset(ds, config.splits, seed)` → 3 `DataLoader`
  (train con `shuffle=True`).
- Optimizer **Adam** (`lr`, `weight_decay`). Loop por época: forward → `loss = w_type·CE(type_logits,type)
  + w_mode·CE(mode_logits,mode)` → backward → step.
- Por época: calcula train/val loss y **macro-accuracy por cabeza**; loguea vía **trackio** sólo si
  `config.tracking` (para que los tests corran sin red/cuenta). Guarda el **mejor checkpoint** por
  macro-accuracy de validación en `out_dir`.
- **Logueo de reproducibilidad**: seed, versiones de dependencias, hash de la config de datos,
  hiperparámetros completos.
- Al terminar: evalúa en **test** → macro-acc por cabeza, accuracy LPI, matriz de confusión y latencia
  (media, p99); vuelca `metrics.json` en `out_dir`.

### `configs/temporal_cnn_elint/train.yaml`

Hiperparámetros completos (datos + modelo + entrenamiento). Nada hardcodeado en el código.

## Reproducibilidad

- Seeds explícitos en torch/numpy/random.
- Hiperparámetros sólo en `train.yaml` versionado.
- `split_dataset` determinista por seed.
- Logueo de seed, deps, hash de datos e hiperparámetros (CLAUDE.md, prioridad alta).
- `trackio` opcional y guardado por flag → tests reproducibles sin dependencias externas.

## Dependencias

Ninguna nueva para el core (torch, numpy ya están). `trackio` se usa de forma opcional/guardada; no es
requisito para tests ni para construir/entrenar en local.

## Tests (`tests/temporal_cnn_elint/`)

- `test_model.py`: shapes de salida (`type_logits[B,8]`, `mode_logits[B,4]`); `predict` cumple
  `threat == mode_to_threat(mode_pred)` siempre; determinismo por seed; corre en CPU; nº de params en rango
  esperado.
- `test_metrics.py`: `macro_accuracy` y `confusion_matrix` con casos conocidos; `lpi_accuracy` filtra a los
  índices LPI correctamente; `profile_latency` devuelve floats positivos con `p99 ≥ media`.
- `test_train.py`: **smoke test** end-to-end — 1-2 épocas con `PDWConfig` minúsculo (pocos emisores/modos,
  `n_trains` bajo) corre, **reduce la train loss**, genera checkpoint + `metrics.json`, y es determinista
  por seed.

## Fuera de alcance (YAGNI)

- Ejecución/entrenamiento real en Colab (paso posterior, skill `/experiment-run`).
- Longitud de secuencia variable; deinterleaving multi-emisor.
- Atención/Transformer/GCN; convolución causal para streaming.
- Exportación ONNX/TensorRT y optimización de despliegue.
- Data augmentation.

## Decisiones clave (resumen)

1. **TCN dilatada, backbone único compartido + 2 cabezas** (tipo, modo) — fiel a "Temporal CNN", ligera.
2. **Amenaza derivada del modo** (`mode_to_threat`) → 3 salidas coherentes, sin redundancia en el loss.
3. **Padding "same" no causal** (ventana completa en inferencia).
4. **Macro-accuracy** como métrica por cabeza + **accuracy LPI** dedicada (contribución de Propuesta.md).
5. **Latencia (media + p99) con batch=1** como métrica ancla (<1 ms).
6. **trackio opcional/guardado**; hiperparámetros en YAML versionado; seeds explícitos.
