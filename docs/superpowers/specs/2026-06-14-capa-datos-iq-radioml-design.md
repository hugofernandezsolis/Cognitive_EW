# Diseño — Capa de datos compartida (IQ / RML2018.01A)

**Fecha:** 2026-06-14
**Ámbito:** `src/cog_ew/data/` (`loaders.py`, `preprocessing.py`) + dependencias en `pyproject.toml`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto

La capa de datos es compartida por los 5 modelos del proyecto. Este spec cubre el primer
pipeline: **IQ samples** anclado en **RadioML 2018.01A** (RML2018), para alimentar la
clasificación (Temporal CNN ELINT) y la GAN. El pipeline de PDW queda fuera de alcance.

La ejecución es en **Google Colab** (default L4; T4 dev; A100 puntual). Esto condiciona dos
decisiones clave de diseño respecto a un planteamiento naïve de "Dataset lazy sobre HDF5":

1. **I/O aleatorio sobre Google Drive es lento.** Leer 2,5M filas una a una desde un `.h5`
   montado en Drive destruye el throughput por epoch.
2. **`h5py` + workers de `DataLoader` (fork) corrompe handles.** El patrón lazy obliga a abrir
   el fichero por worker; fuente clásica de bugs en Colab.

Ambos problemas desaparecen filtrando y cargando el subconjunto en RAM una sola vez.

## Dataset: RML2018.01A

- Fichero HDF5 (`GOLD_XYZ_OSC.0001_1024x2M.h5`), ~20GB.
- Tres datasets internos:
  - `X`: `(2_555_904, 1024, 2)` float — IQ (I, Q) de 1024 muestras.
  - `Y`: `(2_555_904, 24)` one-hot — 24 modulaciones.
  - `Z`: `(2_555_904, 1)` int — SNR en dB.
- Layout **contiguo**: 24 modulaciones × 26 SNRs (−20..+30 dB en pasos de 2) × 4096 frames.
  Permite leer bloques `(mod, snr)` como *slices contiguos* (eficiente incluso sobre Drive).
- 24 modulaciones (orden canónico DeepSig): OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK,
  16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC,
  AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK.

> El orden exacto de `MODULATIONS_2018` se valida contra la documentación del fichero durante la
> implementación (los datos de `Y` son one-hot, así que el índice de clase es estable; el nombre
> es metadato).

## Fuente de datos (Colab)

Dos vías soportadas, resueltas por `resolve_h5_path(config)`:

1. **Kaggle (default)** vía `kagglehub.dataset_download(config.kaggle_dataset)` — descarga y
   cachea el HDF5; devuelve ruta local. Dataset por defecto: `"pinxau1000/radioml2018"`.
2. **Ruta explícita** (`config.h5_path`) — p. ej. Drive montado (persistente entre sesiones).

Lógica de `resolve_h5_path`: si `h5_path` está definido y existe → úsalo; en caso contrario, si
`kaggle_dataset` está definido → descarga con `kagglehub` y localiza el `.h5` dentro del
directorio cacheado; si nada resuelve → `FileNotFoundError` con mensaje accionable.

## Componentes

### `preprocessing.py` — funciones puras (sin I/O)

Operan sobre arrays/tensores IQ. Son la lógica crítica a testear (CLAUDE.md).

- `normalize_power(iq)`: normaliza a **potencia media unitaria por ejemplo**
  (`iq / sqrt(mean(I² + Q²))`). Estándar en RF-ML; desacopla amplitud del SNR.
- `to_channels_first(iq)`: `(N, 2) → (2, N)` para `Conv1d`.
- `iq_to_complex(iq)` / `complex_to_iq(z)`: conversión con **round-trip exacto**.

Convención de tipos: entrada/salida `np.ndarray` float32 (las funciones se aplican antes de
tensorizar; mantener NumPy permite testearlas sin torch). Tensorización en el `Dataset`.

### `loaders.py` — acceso a RML2018.01A

- **`RadioMLConfig`** (dataclass, con `from_yaml(path)` usando PyYAML):
  - `h5_path: str | None` — ruta explícita (override Drive).
  - `kaggle_dataset: str | None = "pinxau1000/radioml2018"` — fuente de descarga.
  - `snr_range: tuple[int, int] | None` — filtro `[min, max]` dB inclusive (None = todos).
  - `modulations: tuple[str, ...] | None` — subconjunto por nombre (None = las 24).
  - `normalize: bool = True` — aplica `normalize_power`.
  - `seed: int = 0` — para splits deterministas.
- **`MODULATIONS_2018: tuple[str, ...]`** — los 24 nombres en orden canónico.
- **`RadioML2018Dataset(torch.utils.data.Dataset)`**:
  - Constructor recibe `RadioMLConfig`. Resuelve ruta, abre HDF5, calcula los rangos de índice
    contiguos de los `(mod, snr)` solicitados, **lee esos slices a RAM** (`np.ndarray`),
    cierra el fichero. Aplica `normalize_power` si procede.
  - `__len__` → nº de ejemplos del subconjunto.
  - `__getitem__(i)` → `(iq: Tensor[2, 1024] float32, label: int, snr: int)`. Tensor en **CPU**;
    el movimiento a device es responsabilidad del bucle de entrenamiento, no del Dataset.
- **`split_dataset(dataset, fractions, seed)`**: split train/val/test **determinista por seed**
  vía `torch.utils.data.random_split` con `torch.Generator().manual_seed(seed)`. `fractions`
  suma 1.0; devuelve los subsets en orden.

## Flujo de datos

```
RadioMLConfig
  → resolve_h5_path (kagglehub | ruta Drive)
  → leer slices contiguos filtrados (mod, snr) a RAM
  → normalize_power + to_channels_first
  → RadioML2018Dataset  → DataLoader  → modelo (device en el train loop)
```

## Reproducibilidad

- `seed` explícito en config; `split_dataset` determinista.
- Config como dataclass cargable desde YAML versionado (sin hiperparámetros hardcodeados).
- El loader no fija seeds globales (responsabilidad del runner de experimentos / `/experiment-run`).

## Dependencias (`pyproject.toml`)

- Runtime: `h5py` (leer HDF5), `pyyaml` (configs YAML), `kagglehub` (descarga + caché).
- Dev: `types-pyyaml` (mypy strict).

## Tests (`tests/data/`)

Sin depender del fichero real de 20GB; usar **fixture HDF5 sintético minúsculo** con el layout
contiguo `(mod, snr, frame)`.

- `test_preprocessing.py`:
  - `normalize_power` deja potencia media ≈ 1.0; idempotencia aproximada.
  - `to_channels_first` produce shape `(2, N)`.
  - round-trip `iq_to_complex`/`complex_to_iq` exacto (allclose).
- `test_loaders.py`:
  - fixture h5 sintético (p. ej. 3 mods × 3 snrs × 4 frames).
  - filtrado por `snr_range` y `modulations` selecciona los **slices correctos** (etiquetas y SNR
    coinciden con lo construido).
  - `__getitem__` devuelve shapes/tipos correctos y tensor en CPU.
  - `split_dataset` es **determinista** con la misma seed y cambia con seed distinta; fracciones
    respetan tamaños.

## Fuera de alcance (YAGNI)

- Pipeline PDW (Pulse Descriptor Words).
- Lazy-mode para el dataset completo (cuando el subconjunto no quepa en RAM).
- Preprocesado-a-disco (.npy/shards).
- Data augmentation.
- Loaders de señales sintéticas de la GAN.
- Mover tensores a device dentro del Dataset.

## Decisiones clave (resumen)

1. **Eager subset → RAM** en vez de lazy fila a fila (Drive I/O + seguridad de workers).
2. **Kaggle/`kagglehub`** como fuente por defecto (no hay mirror fiable en HF); Drive vía ruta.
3. **Preprocessing en NumPy**, tensorización y device fuera de las funciones puras.
4. **RML2018.01A** como dataset ancla (24 mods, IQ 1024×2).
