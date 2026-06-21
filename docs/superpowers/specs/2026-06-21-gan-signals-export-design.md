# Modelo 4 · Sub-pieza B — Muestreo, export masivo y validez

> Fecha: 2026-06-21 · Estado: aprobado · Fuente de verdad del *qué*: `Propuesta.md` (Modelo 4).
> Depende de: sub-pieza A (núcleo cWGAN-GP), ya mergeada.

## Contexto y objetivo

La sub-pieza A entrena un cWGAN-GP que genera ventanas PDW channels-first `(10, 64)` condicionadas por
un embedding de tipo continuo. La sub-pieza B usa un checkpoint entrenado de A para **muestrear y
exportar masivamente** señales sintéticas a `data/synthetic/`, incluyendo **tipos de radar no
catalogados** (obtenidos por interpolación en el espacio de embeddings), y para **medir su validez y
diversidad**.

**Ancla (Propuesta.md):** generar **>200 000** señales sintéticas con **50+ tipos** de radar cognitivo
futuro (incluyendo sistemas nunca catalogados). La mejora **+22 %** de robustez del Modelo 2 se mide en
la sub-pieza C; B produce el dataset y su informe de calidad.

### Decisiones de diseño fijadas

1. **Estrategia de novedad:** **interpolación entre pares de emisores conocidos**. Los 8 emisores del
   catálogo ocupan puntos del espacio de embeddings; los tipos "no catalogados" se generan
   interpolando (y extrapolando fuera de [0,1]) entre pares. Determinista, interpretable y reproducible.
2. **Métricas de validez:** **estructural + distribucional por-feature + diversidad** (sin dependencias
   nuevas; interpretable por feature, apto para publicación).

## Entradas y artefactos previos

- Checkpoint de A: `best.pt = {"generator", "embedding", "critic"}` (state_dicts). B usa solo
  `generator` y `embedding`.
- `PDWGenerator.sample(e) -> (B,10,64)` y `TypeEmbedding.interpolate(id_a, id_b, alpha) -> (e_dim,)`,
  ya existentes.
- Catálogo por defecto `configs/temporal_cnn_elint/emitters.yaml` → **8 emisores** (`n_emitters = 8`).
- Convención de ejes: channels-first `(10, 64)`; canales 0–4 continuos en [0,1]
  (`rf, pw, pa, aoa, pri`), 5–9 one-hot `intra_pulse_mod`.
- `h5py >= 3.16` ya es dependencia; el proyecto usa esquema HDF5 estilo RadioML.

## Módulos (una responsabilidad por fichero)

### `src/cog_ew/gan_signals/sampler.py` — producción de ventanas

- `SyntheticType` (dataclass frozen): `type_id: int`, `source_a: int`, `source_b: int`,
  `alpha: float`, `is_known: bool`.
- `load_generator(checkpoint, *, z_dim, e_dim, channels, n_emitters, device) -> tuple[PDWGenerator, TypeEmbedding]`:
  construye ambos módulos, carga sus `state_dict` desde el checkpoint con
  `torch.load(..., weights_only=True)`, y los pone en `.eval()` (la BatchNorm del generador usa running
  stats → distribución de inferencia estable y determinista).
- `build_type_catalog(n_known, *, alphas, extrapolate) -> list[SyntheticType]`: emite los `n_known`
  tipos conocidos (`source_a == source_b == i`, `alpha = 0.0`, `is_known = True`) más, para cada par
  `(a, b)` con `a < b`, un tipo por cada valor de `alphas` (`is_known = False`). Si `extrapolate` está
  activo, añade también alphas fuera de [0,1]. Con `n_known = 8` y al menos 2 alphas, produce ≥50 tipos.
  `type_id` es el índice secuencial. Determinista.
- `resolve_embedding(embedding, stype) -> torch.Tensor`: para un tipo conocido, `embedding` del id;
  para uno novedoso, `embedding.interpolate(source_a, source_b, alpha)`. Devuelve `(e_dim,)`.
- `sample_type(generator, embedding, stype, n, device) -> torch.Tensor`: bajo `torch.no_grad()`,
  resuelve el embedding, lo repite `n` veces, y devuelve `generator.sample(e_batch)` → `(n, 10, 64)`.

### `src/cog_ew/gan_signals/validity.py` — métricas

- `structural_validity(windows) -> dict[str, float]`: `continuous_in_range_frac` (fracción de valores de
  los canales 0–4 dentro de [0,1]) y `categorical_onehot_frac` (fracción de pulsos cuyos canales 5–9
  suman 1 y son one-hot). Sanity ≈ 1.0 por construcción del generador.
- `distributional_realism(generated, real) -> dict[str, Any]`: `wasserstein1_per_feature` (lista de 5,
  distancia de Wasserstein-1 entre las distribuciones marginales gen vs real de cada feature continua),
  `wasserstein1_mean`, y `categorical_tv_distance` (distancia de variación total entre la distribución
  agregada de `intra_pulse_mod` generada y real). `generated`/`real` son tensores `(N,10,64)`.
- `diversity(windows, type_ids) -> dict[str, float]`: `mean_intersample_std` (media de la desviación
  típica entre muestras, proxy de mode-collapse), `n_distinct_categorical_patterns` (nº de secuencias
  categóricas únicas), `n_types`, `coverage` (fracción de tipos con ≥1 muestra).

### `src/cog_ew/gan_signals/export.py` — orquestación

- `ExportConfig` (dataclass + `from_yaml`): `checkpoint: str`, `z_dim, e_dim, channels, n_emitters`,
  `alphas: tuple[float, ...]`, `extrapolate: bool`, `samples_per_type: int`, `out_path: str`,
  `library_path: str`, `n_real_compare: int`, `seed: int`, `device: str`.
- `export_synthetic(config) -> dict[str, Any]`: `_set_seeds`; carga gen+emb; construye el catálogo;
  pre-dimensiona el HDF5 (`N = n_types * samples_per_type`) y **rellena por-tipo** los slices
  (sin acumular todas las ventanas en RAM); calcula validez contra un lote de `n_real_compare` ventanas
  reales de `PDWSyntheticDataset`; escribe `run_meta.json` y `metrics.json` junto al `.h5`.

## Esquema HDF5 (`data/synthetic/<name>.h5`)

Datasets:
- `X`: `(N, 10, 64)` float32 — las ventanas.
- `type_id`: `(N,)` int64.
- `source_a`: `(N,)` int64.
- `source_b`: `(N,)` int64.
- `alpha`: `(N,)` float32.
- `is_known`: `(N,)` bool.

Atributos del fichero: `n_types`, `samples_per_type`, `checkpoint_hash` (sha256 del fichero de
checkpoint), `seed`.

La provenance (`source_a/source_b/alpha/is_known`) deja a la sub-pieza C libertad total para definir
etiquetas (catalogado vs no catalogado) sin reexportar.

## Reproducibilidad y salidas

- `_set_seeds(seed)`: `random`, `numpy`, `torch` (mismo patrón que A).
- `run_meta.json`: `seed`, hiperparámetros (`asdict(config)`), `config_hash` (sha256), `checkpoint_hash`,
  versiones de dependencias (python/torch/numpy/h5py).
- `metrics.json`: las claves de `structural_validity`, `distributional_realism` y `diversity`, más
  `n_windows` y `n_types`.
- `weights_only=True` en toda carga de checkpoint.
- No se exponen parámetros de amenazas reales: el pipeline opera solo sobre el catálogo sintético y un
  checkpoint entrenado sobre él.

## Configuración (YAML)

`configs/gan_signals/export.yaml`. Todos los hiperparámetros viven en YAML, nunca hardcodeados:
`checkpoint`, `z_dim`, `e_dim`, `channels`, `n_emitters`, `alphas`, `extrapolate`, `samples_per_type`,
`out_path`, `library_path`, `n_real_compare`, `seed`, `device`.

**Valores por defecto sugeridos** (se concretan en el YAML; los tests usan configs pequeñas):
`z_dim=64`, `e_dim=16`, `channels=64`, `n_emitters=8`, `alphas=[0.25, 0.5, 0.75]`, `extrapolate=false`,
`samples_per_type=2500`, `out_path=data/synthetic/wgan_gp.h5`,
`library_path=configs/temporal_cnn_elint/emitters.yaml`, `n_real_compare=4000`, `seed=0`, `device=cpu`.
Con `n_emitters=8` y 3 alphas: 8 conocidos + 28 pares × 3 alphas = **92 tipos** (≥50);
92 × 2500 = **230 000** ventanas (≥200 000).

## Plan de tests (TDD)

1. `build_type_catalog(8, alphas=(0.25,0.5,0.75), extrapolate=False)` produce ≥50 tipos; los 8 primeros
   son conocidos (`is_known`, `alpha==0`, `source_a==source_b`); los novedosos tienen `(a<b)` y un
   `alpha` de la lista; `type_id` secuencial; determinista (dos llamadas iguales).
2. `resolve_embedding` para un tipo conocido devuelve el `embedding` del id; para uno novedoso, la
   interpolación; forma `(e_dim,)`.
3. `sample_type` devuelve `(n, 10, 64)` y pasa `structural_validity` ≈ 1.0.
4. `structural_validity` sobre salida del generador da `continuous_in_range_frac == 1.0` y
   `categorical_onehot_frac == 1.0`.
5. `distributional_realism(gen, real)` devuelve 5 valores Wasserstein-1 finitos ≥ 0, su media, y una
   `categorical_tv_distance` en [0,1].
6. `diversity`: muestras variadas → `mean_intersample_std > 0`; muestras idénticas (mode-collapse) →
   `mean_intersample_std == 0`; `coverage == 1.0` cuando todos los tipos están representados.
7. **Integración:** `export_synthetic` con config diminuta (pocos tipos, `samples_per_type` pequeño,
   `out_path` en `tmp_path`) escribe el `.h5` con el esquema completo, `run_meta.json` y `metrics.json`;
   el `.h5` se relee con `h5py` (formas correctas, `N == n_types*samples_per_type`); reproducible por
   seed (dos exports con la misma seed dan `X` idéntico).

## Fuera de alcance de la sub-pieza B

- Evaluación de robustez +22 % sobre el Modelo 2 (sub-pieza C); cualquier reentrenamiento de M2.
- Un loader de `data/synthetic/` para M2 (lo añade C según lo que necesite).
- Logging por intervalo del entrenamiento de A (backlog de A/Fase 6).
- Annealing/optimización de la GAN (sub-pieza A / Fase 6).
