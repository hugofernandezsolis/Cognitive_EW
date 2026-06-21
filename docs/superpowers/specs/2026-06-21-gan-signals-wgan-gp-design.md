# Modelo 4 — GAN de señales de amenaza sintéticas · Sub-pieza A: núcleo cWGAN-GP

> Fecha: 2026-06-21 · Estado: aprobado · Fuente de verdad del *qué*: `Propuesta.md` (Modelo 4).

## Contexto y objetivo del Modelo 4

El Modelo 4 genera señales radar sintéticas realistas (AESA con waveform cognitiva, LPI, banda
ancha, pasivo) para proporcionar datos de entrenamiento ilimitados, **incluyendo sistemas nunca
catalogados**. Su valor se mide reevaluando el **Modelo 2** (Temporal CNN ELINT) con datos
aumentados.

**Ancla (Propuesta.md):** generar **>200 000** señales sintéticas con **50+ tipos** de radar
cognitivo futuro; mejorar la robustez del clasificador (Modelo 2) un **+22 %** frente a sistemas no
catalogados.

### Consumidor: representación PDW del Modelo 2

El Modelo 2 ingiere **ventanas PDW channels-first** de forma `(10, 64)` (10 features = canales,
64 pulsos = longitud; `window_sequence` transpone a `(features, window)`, confirmado empíricamente y
por la config ELINT `in_channels: 10, seq_len: 64`):

- 5 features continuas normalizadas a [0,1] (`normalize_pdw` contra `CONTINUOUS_RANGES`):
  `rf, pw, pa, aoa, pri` (canales 0–4).
- 5 columnas one-hot de `intra_pulse_mod` (`none, lfm, barker, fmcw, polyphase`) (canales 5–9).
- Etiquetas: `type` (emisor), `mode`, `threat`.

Fuente de datos reales para la GAN: `PDWSyntheticDataset` (generador PDW basado en reglas, ya
validado y mergeado). El catálogo por defecto (`configs/temporal_cnn_elint/emitters.yaml`) tiene
**8 emisores** (`n_emitters = 8`).

## Descomposición del Modelo 4

Tres sub-piezas, cada una con su ciclo spec → plan → implementación (espejo del Modelo 3 A/B/C):

- **A — Núcleo cWGAN-GP** (este documento): generador + crítico + entrenamiento WGAN-GP.
- **B — Muestreo, export masivo y validez:** >200 000 ventanas sobre 50+ embeddings (barrido /
  interpolación del espacio de tipos, incluyendo no catalogados) → `data/synthetic/` (HDF5) con
  etiquetas; métricas de validez física y diversidad.
- **C — Evaluación de robustez (+22 %):** hold-out de tipos como "no catalogados"; Modelo 2 baseline
  vs Modelo 2 + aumento sintético; Δaccuracy sobre los tipos retenidos.

## Decisiones de diseño fijadas

1. **Representación generada:** ventanas PDW channels-first `(10, 64)` directamente (10 features ×
   64 pulsos; camino más directo y medible al ancla +22 %; alimentan al Modelo 2 sin conversión).
2. **Condicionamiento:** **embedding de tipo continuo** (no etiqueta discreta), para poder generar
   tipos "nunca catalogados" interpolando/extrapolando en el espacio de embeddings.
3. **Arquitectura G/D:** **1D-CNN temporal (DCGAN)** sobre el eje de pulsos, para capturar estructura
   pulso-a-pulso (patrón PRI, modulación de scan en `pa`, frequency hopping). Alineada con el Temporal
   CNN del Modelo 2.
4. **Salida categórica:** **Gumbel-softmax straight-through** (temperatura `τ`) sobre las 5 columnas
   `intra_pulse_mod`, para que el crítico vea la misma forma casi-one-hot que los datos reales.
5. **Estabilización:** **WGAN-GP** (gradient penalty `λ=10`), referencia del estado del arte (cWGAN-GP
   para formas de onda LPI/LPD).

## Arquitectura

### Convención de ejes

Los datos del Modelo 2 ya son **channels-first** `(10, 64)` (10 features = canales, 64 pulsos =
longitud). La GAN opera nativamente en `(B, 10, 64)`, sin transposiciones ni adaptadores:
`PDWGenerator.sample()` devuelve `(B, 10, 64)` y `PDWCritic.forward()` acepta `(B, 10, 64)`, idénticos
a los lotes que consume el Modelo 2.

### `TypeEmbedding(n_emitters, e_dim)`

`nn.Embedding` que mapea id de emisor → vector continuo `e ∈ R^{e_dim}`. **`G` y `D` reciben `e` como
tensor continuo, no como id**, de modo que la sub-pieza B pueda alimentar embeddings
interpolados/novedosos directamente. La tabla se optimiza en el paso del generador y se guarda en el
checkpoint. Expone una utilidad de interpolación lineal entre dos ids (para B).

### `PDWGenerator.forward(z, e) → (B, 10, 64)`

- Entrada: `z` (ruido, `z_dim`) concatenado con `e` (`e_dim`).
- Proyección a semilla `(B, C, L0)` → pila de `ConvTranspose1d` con upsampling hasta longitud 64,
  canal de salida = 10.
- Cabeza de salida partida sobre los 10 canales:
  - canales 0–4 (continuas) → `sigmoid` → [0,1].
  - canales 5–9 (categóricas) → `gumbel_softmax(..., tau=τ, hard=True)` sobre el eje de canal
    (`dim=1`) por pulso → casi one-hot, diferenciable (straight-through).
- Salida `(B, 10, 64)` (channels-first, sin transponer).

### `PDWCritic.forward(x, e) → (B, 1)`

- `x` `(B, 10, 64)` → pila de `Conv1d` con downsampling → vector global (flatten / pooling).
- Se concatena con `e` proyectado → `Linear` → score escalar. **Sin sigmoid final** (Wasserstein).
- **LayerNorm** en las capas del crítico (no BatchNorm: invalidaría el gradient penalty).

## Entrenamiento (WGAN-GP)

- **Pérdida del crítico:** `E[D(fake, e)] − E[D(real, e)] + λ · GP`, con
  `GP = E[(‖∇_x̂ D(x̂, e)‖₂ − 1)²]`, donde `x̂` es la interpolación real/fake con el **mismo `e`** y
  `λ = lambda_gp` (10 por defecto). El gradient penalty se calcula solo respecto a `x̂` (no a `e`).
- **Pérdida del generador:** `−E[D(G(z, e), e)]`.
- `n_critic` pasos de crítico por cada paso de generador.
- Optimizadores Adam con `β = (0.0, 0.9)`; `lr` desde config.
- Embedding optimizado en el paso del generador (conditioning del lado generador); el crítico lee `e`
  como entrada de condicionamiento pero solo sus propios parámetros se actualizan en el paso de crítico.
- Temperatura Gumbel `τ` fija (config) en la sub-pieza A (YAGNI: sin annealing).

## Configuración (YAML)

`configs/gan_signals/wgan_gp.yaml`. **Todos los hiperparámetros viven en YAML**, nunca hardcodeados:
`z_dim, e_dim, channels, n_critic, lambda_gp, lr, gumbel_tau, batch_size, total_steps, seed, device,
out_dir` + referencia a la config PDW (`library_path`, `window`, `n_pulses`, `n_trains`, etc.). Carga
vía dataclass `WGANGPConfig.from_yaml` (patrón de `marl_formation`/`temporal_cnn_elint`).

**Valores por defecto sugeridos** (se concretan en el YAML; los tests usan configs pequeñas para ser
rápidos): `z_dim=64`, `e_dim=16`, `channels=64` (base de canales de las convoluciones),
`n_critic=5`, `lambda_gp=10`, `lr=1e-4`, `gumbel_tau=1.0`, `batch_size=64`, `total_steps=20000`. La
referencia PDW reutiliza una config de catálogo existente de `configs/temporal_cnn_elint/` o
`configs/gan_signals/`.

## Reproducibilidad y salidas

- `_set_seeds(seed)`: `random.seed`, `numpy.random.seed`, `torch.manual_seed`.
- `run_meta.json`: seed, hiperparámetros completos, `config_hash` (sha256 de la config), versiones de
  dependencias (python/torch/numpy). Mismo patrón que `marl_formation/train.py`.
- `metrics.json`: Wasserstein estimate final, gradient penalty, **latencia del generador (mean/p99)**
  (requisito de dominio: perfilado de inferencia).
- Checkpoint del generador + `TypeEmbedding` (`.pt`, `torch.save` de los `state_dict`; carga con
  `weights_only=True`).
- **Métricas de estabilidad logueadas por intervalo:** Wasserstein estimate (`D(real) − D(fake)`),
  gradient penalty, loss del crítico, loss del generador, y un **proxy de mode-collapse** (desviación
  típica media de las features del batch generado).
- No se exponen en logs ni artefactos parámetros de amenazas reales: el pipeline opera solo sobre el
  catálogo sintético `PDWSyntheticDataset`.

## Módulos

- `src/cog_ew/gan_signals/generator.py` — `PDWGenerator`, cabeza de salida (sigmoid + Gumbel-softmax),
  `TypeEmbedding`.
- `src/cog_ew/gan_signals/discriminator.py` — `PDWCritic`.
- `src/cog_ew/gan_signals/train.py` — `WGANGPConfig`, `gradient_penalty()`, bucle de entrenamiento,
  `_set_seeds`, `run_meta`/`metrics`, perfilado de latencia, checkpoint.

## Plan de tests (TDD)

1. `PDWGenerator.sample()` devuelve `(B, 10, 64)`.
2. Columnas continuas (0–4) en [0,1].
3. Columnas categóricas (5–9) suman ~1 por pulso y son casi one-hot (probabilidad máxima alta).
4. `PDWCritic` devuelve `(B, 1)` y admite valores negativos (sin sigmoid).
5. `gradient_penalty()` devuelve escalar finito, ≥ 0 y diferenciable.
6. Determinismo por seed: misma seed → mismo batch generado.
7. Un paso de crítico cambia los parámetros del crítico; un paso de generador cambia los parámetros del
   generador y del `TypeEmbedding`.
8. Las pérdidas de crítico y generador son finitas.
9. El condicionamiento importa: distinto `e` → salida distinta (sanity).
10. `TypeEmbedding` mapea ids a vectores e interpola linealmente entre dos ids.

## Fuera de alcance de la sub-pieza A

- Export masivo de >200 000 señales y barrido de 50+ tipos (sub-pieza B).
- Métricas de validez física y diversidad distribucional vs PDW real (sub-pieza B).
- Evaluación de robustez +22 % sobre el Modelo 2 (sub-pieza C).
- Annealing de temperatura Gumbel, schedulers de lr, EMA del generador (YAGNI; reconsiderar en Fase 6
  si el entrenamiento real lo requiere).
