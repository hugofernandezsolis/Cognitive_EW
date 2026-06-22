# Diseño — Modelo 2 v2 proposal-first: ELINT estricto tipo/modo/amenaza

**Fecha:** 2026-06-22  
**Ámbito:** `src/cog_ew/temporal_cnn_elint/`, `src/cog_ew/data/`, `src/cog_ew/experiments/`, `configs/temporal_cnn_elint/`, `tests/temporal_cnn_elint/`, `tests/experiments/`  
**Estado:** aprobado en brainstorming; pendiente de revisión del spec por el usuario

## Contexto

La ejecución `full` de Fase 6 muestra que el Modelo 2 actual cumple el anchor implementado, pero no la
lectura estricta de `Propuesta.md`.

- `lpi_accuracy = 1.0`
- `macro_acc_type = 0.8923357228195938`
- `macro_acc_mode = 0.921455938697318`
- `latency_p99_ms = 1.6611460699880354`

La propuesta pide clasificar en tiempo real: tipo de emisor, modo de operación y estado de amenaza, con
accuracy >96 % incluyendo LPI y latencia <1 ms. Por tanto, M2-v2 debe priorizar cumplimiento literal de la
propuesta, no solo mejorar el resultado LPI.

## Objetivo

Construir una segunda versión del Modelo 2 que solo pase el anchor si cumple simultáneamente:

- `macro_acc_type >= 0.96`
- `macro_acc_mode >= 0.96`
- `macro_acc_threat >= 0.96`
- `lpi_accuracy >= 0.96`
- `latency_p99_ms < 1.0`

El anchor de M2 se convierte en una métrica conservadora:

```text
achieved = min(macro_acc_type, macro_acc_mode, macro_acc_threat, lpi_accuracy)
passed = achieved >= 0.96 and latency_p99_ms < 1.0
```

## Decisiones De Diseño

### 1. Mantener PDW/ELINT como representación principal

No se migra M2 a RadioML/IQ. RadioML clasifica modulaciones RF, pero la tarea de la propuesta es ELINT:
emisor, modo y amenaza. La representación PDW es la que está alineada con SA-2, SA-6, S-300, S-400, HQ-9,
AESA y emisores LPI.

### 2. Enriquecer el dataset con rasgos temporales y de comportamiento

El problema actual está en separar emisores cercanos y modos próximos. M2-v2 añade canales derivados por
ventana para exponer señales que ya existen en el simulador, pero que el modelo actual debe inferir de forma
indirecta.

Canales base actuales:

- `rf`
- `pw`
- `pa`
- `aoa`
- `pri`
- one-hot de modulación intra-pulso

Canales nuevos propuestos:

- `delta_rf`: variación normalizada de frecuencia entre pulsos consecutivos.
- `delta_pri`: variación normalizada de PRI entre pulsos consecutivos.
- `rolling_pri_std`: jitter local del PRI.
- `rolling_rf_std`: indicador local de hopping/agilidad RF.
- `rolling_pw_mean`: contexto local de anchura de pulso.
- `mode_progression_hint`: rasgo sintético normalizado que codifica severidad temporal del modo cuando esté
  disponible en generación.
- `lpi_hint`: canal binario derivado del catálogo de emisor/modo.
- `freq_hopping_hint`: canal binario derivado del catálogo de emisor/modo.

Los hints son aceptables porque el dataset es sintético y el objetivo del TFM es construir un benchmark
reproducible de EW cognitiva. Deben documentarse como parámetros publicados/simulados, no como sensores
reales perfectos.

### 3. Aprender amenaza como tercera cabeza real

El modelo actual deriva amenaza desde modo. M2-v2 añade una cabeza explícita:

- `head_type`
- `head_mode`
- `head_threat`

El entrenamiento usa:

```text
loss = w_type * CE(type) + w_mode * CE(mode) + w_threat * CE(threat)
```

La inferencia devuelve las tres predicciones aprendidas. Se puede seguir calculando una amenaza derivada del
modo como métrica auxiliar de coherencia, pero no como sustituto de `head_threat`.

### 4. Backbone ligero orientado a <1 ms

M2-v2 mantiene una Temporal CNN, pero cambia a bloques más eficientes:

- `Conv1d` inicial para proyectar canales enriquecidos a `hidden`.
- Bloques TCN residuales depthwise-separable:
  - depthwise Conv1d dilatada
  - pointwise Conv1d
  - GELU o SiLU
  - residual
- Squeeze-and-Excitation ligero opcional en los últimos bloques.
- Global average pooling temporal.
- Tres cabezas lineales.

El primer perfil objetivo es `hidden=48` o `hidden=64`, dilations `[1, 2, 4, 8]`, sin Transformer ni atención
pesada. Si `hidden=64` no baja de 1 ms p99, se prueba `hidden=48` antes de tocar la arquitectura.

### 5. Perfilado de latencia como gate, no como dato decorativo

`profile_latency` debe ejecutarse con:

- `torch.inference_mode()`
- batch=1
- warmup suficiente
- sincronización CUDA antes/después de medir
- modelo en `eval()`

El anchor falla si `latency_p99_ms >= 1.0`, aunque las accuracies pasen.

## Cambios De Componentes

### `data/pdw_dataset.py`

Agregar soporte configurable para features v2:

- `feature_set: "base" | "v2"` en `PDWConfig`.
- `base` mantiene los 10 canales actuales para compatibilidad.
- `v2` añade los canales derivados y actualiza `in_channels`.

La generación debe seguir siendo determinista por seed.

### `temporal_cnn_elint/model.py`

Agregar una config y modelo v2 sin romper el modelo actual:

- `TemporalCNNV2Config`
- `TemporalCNNV2`

`TemporalCNN` queda disponible para compatibilidad con runs anteriores.

### `temporal_cnn_elint/train.py`

Actualizar `TrainConfig` para seleccionar arquitectura:

- `architecture: "tcn_v1" | "tcn_v2"`
- `loss_weights: [type, mode, threat]` para v2
- métricas finales con `macro_acc_threat`

Para `tcn_v1`, el comportamiento actual se conserva.

### `temporal_cnn_elint/metrics.py`

Agregar:

- `strict_elint_score(metrics) -> float`
- `strict_elint_passed(metrics, target=0.96, latency_p99_ms=1.0) -> bool`

La función debe rechazar `NaN`, `inf` y métricas ausentes.

### `experiments/anchors.py`

Cambiar `run_elint_anchor` para usar el score estricto cuando el entrenamiento produce
`macro_acc_threat`. Si falta esa métrica, el anchor debe fallar o marcar un error claro, no caer a
`lpi_accuracy`.

## Configuración

Nuevo YAML:

`configs/temporal_cnn_elint/train_v2.yaml`

Debe fijar:

- `data.feature_set: v2`
- `model.in_channels` igual al número real de canales v2
- `architecture: tcn_v2`
- `epochs` iniciales para full: 40-60 si no rompe tiempo de Colab
- `loss_weights`: empezar con `[1.0, 1.0, 0.7]`
- `device: cpu` por defecto en repo; el perfil `full` lo sobreescribe a `cuda`

El perfil `configs/experiments/full.yaml` debe apuntar a `train_v2.yaml` para M2 cuando M2-v2 esté listo.

## Plan De Validación

### Tests unitarios

- `PDWSyntheticDataset(feature_set="base")` conserva shape actual.
- `PDWSyntheticDataset(feature_set="v2")` produce más canales y valores finitos.
- Features derivadas (`delta_rf`, `delta_pri`, rolling stats) tienen longitud y rango esperados.
- `TemporalCNNV2.forward` devuelve tres logits con shapes correctos.
- `TemporalCNNV2.predict` devuelve tres predicciones aprendidas.
- `strict_elint_score` usa el mínimo de type/mode/threat/LPI.
- `strict_elint_passed` exige p99 <1 ms y rechaza no finitos.
- `run_elint_anchor` no puede pasar con LPI-only.

### Smoke test

Un entrenamiento v2 pequeño debe:

- escribir `metrics.json`
- incluir `macro_acc_type`, `macro_acc_mode`, `macro_acc_threat`, `lpi_accuracy`, `latency_mean_ms`,
  `latency_p99_ms`
- producir checkpoint
- ser determinista con la misma seed

### Validación full

Ejecutar:

```bash
python notebooks/run_anchors.py --profile full --anchors elint --out-dir runs/anchors_full_m2_v2
```

El resultado aceptable es:

```text
achieved >= 0.96
passed = true
latency_p99_ms < 1.0
```

## Riesgos

- Los hints sintéticos pueden hacer el benchmark demasiado fácil. Mitigación: reportarlos explícitamente como
  parámetros simulados y añadir una ablation `feature_set=base` vs `feature_set=v2`.
- La tercera cabeza puede mejorar coherencia pero no accuracy de tipo. Mitigación: ajustar loss weights y
  revisar matrices de confusión tras el primer full run.
- `hidden=64` puede no cumplir latencia. Mitigación: probar `hidden=48`, depthwise-separable conv y profiling
  con `torch.inference_mode()`.
- Si el accuracy sube solo por hints binarios, el paper debe formularlo como benchmark NTTR parametrizado, no
  como inferencia ELINT puramente pasiva desde IQ crudo.

## Fuera De Alcance

- Migrar M2 a RadioML/IQ.
- Añadir Transformer o modelos grandes.
- Integrar datos DARPA/NTTR reales clasificados.
- Optimización TensorRT/ONNX.
- Cambiar los modelos M1, M3, M4 o M5 salvo el anchor M2 en Fase 6.

## Criterio De Done

M2-v2 está terminado cuando:

1. La suite completa pasa.
2. El anchor M2 usa score estricto, no LPI-only.
3. El entrenamiento full de M2-v2 produce `passed=true`.
4. `metrics.json` contiene las cuatro accuracies y latencia media/p99.
5. El roadmap documenta que M2-v2 reemplaza al M2 original como resultado reportable.
