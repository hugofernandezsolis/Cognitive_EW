# Diseño — Pipeline de datos PDW sintético (Modelo 2, ELINT)

**Fecha:** 2026-06-14
**Ámbito:** `src/cog_ew/data/` (nuevos módulos PDW) + funciones puras en `preprocessing.py` + config en `configs/temporal_cnn_elint/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 2 (Temporal CNN ELINT) procesa la **secuencia de pulsos
interceptados por el RWR** y clasifica en tiempo real, de forma **multi-tarea**: (1) **tipo de emisor**
(SA-20, HQ-9, S-400, AESA…), (2) **modo de operación** (búsqueda, TWS, tracking, guiado de misil) y
(3) **estado de amenaza**, con latencia <1 ms, **incluyendo radares LPI** (contribución: >96 % accuracy
incl. LPI vs. <65 % convencional).

Este spec cubre **solo el pipeline de datos PDW** (sub-proyecto 1). El modelo Temporal CNN multi-tarea
es un sub-proyecto posterior con su propio spec.

Decisiones de encuadre tomadas en brainstorming:
- **Fuente:** generador sintético etiquetado, parametrizado desde parámetros tipo SIADS (alineado con
  Propuesta.md: "NTTR SIADS params, SA-2 a S-400").
- **Turing Synthetic Radar Dataset fuera de alcance:** no está en Propuesta.md y su tarea es
  *deinterleaving* (clustering de emisores desconocidos por V-measure), no clasificación por taxonomía.
- **Pipeline separado del de IQ** (regla de CLAUDE.md: "mantener pipelines separados para cada
  representación"). No reutiliza el `Dataset` de RadioML; sí reutiliza `split_dataset`.

## Representación de datos

### Pulso (PDW)
Cada pulso aporta 5 magnitudes continuas + 1 descriptor categórico:
- **RF** — frecuencia portadora.
- **PW** — ancho de pulso.
- **PA** — amplitud de pulso (porta la modulación de barrido de antena).
- **AOA** — ángulo de llegada (rumbo al emisor).
- **PRI** — intervalo de repetición (derivado de la diferencia de TOA entre pulsos consecutivos).
- **Modulación intra-pulso** — categórica, `K = 5`: `{none/CW, LFM, Barker/phase-coded, FMCW,
  polifásico (Frank)}`. Discriminador clave de LPI.

### Tensor de entrada al modelo
Ventana de **N = 64 pulsos** de longitud fija, **channels-first** para `Conv1d`:
- 5 canales continuos (RF, PW, PA, AOA, PRI) normalizados.
- `K = 5` canales one-hot de modulación intra-pulso.
- **F = 10 canales × N = 64** → `Tensor[10, 64]`.

### Etiquetas
Cada ventana lleva 3 etiquetas enteras: `(type, mode, threat)`. El **estado de amenaza se deriva del
modo**: búsqueda→bajo(0), TWS/adquisición→medio(1), tracking→alto(2), guiado de misil→crítico(3).

## Componentes

### `pdw_library.py` — taxonomía y parámetros (config versionada)
- Dataclasses para especificar un emisor y sus modos: por **modo**, rangos de RF (banda), patrón de PRI
  (`fixed` | `stagger` | `jitter`) con rango, rango de PW, periodo de barrido, flags de agilidad
  (`freq_hopping: bool`, `lpi: bool`) y **tipo(s) de modulación intra-pulso**.
- `EmitterLibrary.from_yaml(path)` carga la librería desde `configs/temporal_cnn_elint/emitters.yaml`.
- Taxonomía inicial (~8-10 emisores): **SA-2, SA-6, S-300 (SA-20), S-400, HQ-9, AESA, LPI-FMCW,
  LPI-polifásico**. Los LPI declaran modulación FMCW/polifásica; los convencionales none/LFM/Barker.
- `MODES = (search, tws, track, missile_guidance)` y el mapeo `mode → threat`.
- Constantes con los **rangos físicos** por feature (para normalización reproducible) y los nombres de
  clase de `type`, `mode`, `threat`, `intra_pulse_mod`.

### `pdw_generator.py` — generación sintética (reproducible)
- `generate_pulse_train(emitter, mode, n_pulses, rng) -> PulseTrain`: produce TOA, RF, PW, PA, AOA y
  tipo de modulación por pulso, aplicando realismo configurable:
  - **PRI** según patrón (fijo / stagger con secuencia / jitter aleatorio acotado).
  - **Agilidad de RF** (hopping dentro de la banda) si el emisor lo declara.
  - **Amplitud modulada por barrido** (PA varía con el periodo de barrido de antena).
  - **Ruido de medida** gaussiano por campo (desv. configurable).
  - **Pulsos perdidos** (drop con probabilidad p) y **espurios** (inserción de pulsos aleatorios).
- Todo el azar pasa por un `numpy.random.Generator` sembrado → determinista.

### `preprocessing.py` — funciones puras PDW (sin I/O, lo crítico a testear)
- `toa_to_pri(toa)` — deriva PRI de TOA consecutivos: `PRI[i] = TOA[i] − TOA[i−1]`, con `PRI[0] =
  PRI[1]` (padding del primero) para conservar la longitud N.
- `normalize_pdw(features, ranges)` — estandariza las 5 continuas a partir de **rangos físicos
  conocidos** de la librería (no z-score dependiente de datos → reproducible).
- `one_hot_intra_pulse(codes, K)` — one-hot del descriptor categórico.
- `window_sequence(train, n)` — parte un tren en ventanas de N pulsos de longitud fija
  (con manejo del resto: descartar cola incompleta).

### `pdw_dataset.py` — config y Dataset
- `PDWConfig` (dataclass + `from_yaml`): emisores/modos a incluir, `window=64`, flags de realismo,
  `n_trains` por (emisor, modo), `normalize: bool`, `seed`.
- `PDWSyntheticDataset(torch.utils.data.Dataset)`: genera los trenes con el generador, los ventana y
  preprocesa **una vez en memoria** (subconjunto cabe en RAM, como el pipeline IQ); `__getitem__`
  devuelve `(pdw: Tensor[10, 64], type: int, mode: int, threat: int)` en CPU.
- Reutiliza `split_dataset(ds, fractions, seed)` de `loaders.py`.

## Flujo de datos

```
PDWConfig + EmitterLibrary(YAML)
  → pdw_generator (trenes por emisor/modo, seeded, con realismo)
  → toa_to_pri + normalize_pdw + one_hot_intra_pulse + window_sequence
  → PDWSyntheticDataset → DataLoader → modelo multi-tarea (device en el train loop)
```

## Reproducibilidad
- Toda la generación pasa por un `Generator` sembrado desde `PDWConfig.seed`.
- Normalización por **rangos físicos versionados** (no dependiente de datos).
- Librería de emisores en **YAML versionado** (sin hardcodear parámetros).
- `split_dataset` determinista por seed.

## Dependencias
Ninguna nueva (numpy, torch, pyyaml ya están). No requiere h5py ni kagglehub.

## Tests (`tests/data/`)
- `test_pdw_preprocessing.py`: `toa_to_pri` correcto; `normalize_pdw` mapea rangos físicos a escala
  esperada; `one_hot_intra_pulse` shape/valores; `window_sequence` longitud y descarte de cola.
- `test_pdw_library.py`: `from_yaml` parsea emisores/modos; mapeo `mode→threat` correcto; todos los
  emisores LPI declaran modulación LPI.
- `test_pdw_generator.py`: determinismo por seed (misma seed → mismo tren); PRI respeta el patrón
  (fijo ≈ constante, jitter dentro de cota); drop/espurios alteran el conteo de pulsos como se espera.
- `test_pdw_dataset.py`: `__getitem__` devuelve `Tensor[10,64]` float32 en CPU + 3 etiquetas válidas;
  filtrado por emisores/modos; `split_dataset` determinista.

## Fuera de alcance (YAGNI)
- El modelo Temporal CNN multi-tarea (sub-proyecto siguiente).
- Turing dataset / loaders de datos reales.
- Entremezclado multi-emisor y *deinterleaving*.
- Cadena IQ→PDW (GNU Radio / MATLAB Phased Array).
- Data augmentation; longitud de secuencia variable.

## Decisiones clave (resumen)
1. **Generador sintético etiquetado** (no Turing) — alineado con Propuesta.md.
2. **Multi-tarea**: 3 etiquetas (tipo, modo, amenaza); amenaza derivada del modo.
3. **Features por pulso** = 5 continuas + one-hot modulación intra-pulso (K=5) → 10 canales × 64.
4. **Modulación intra-pulso** como discriminador explícito de LPI.
5. **Pipeline PDW separado** del de IQ; reutiliza `split_dataset`.
6. **Normalización por rangos físicos** versionados (reproducible).
