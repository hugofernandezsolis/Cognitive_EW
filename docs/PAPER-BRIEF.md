# Brief para el agente de análisis de resultados y redacción del artículo (Fase 6 · B+C)

> Este documento es la **instrucción de tarea** para un agente que: (1) analiza los resultados de la
> ejecución real de los experimentos y (2) redacta el artículo científico. No describe *cómo* se
> construyó el código (eso está en los specs/plans), sino *qué* hay que producir y desde dónde.
>
> **Fuente de verdad del proyecto:** [`Propuesta.md`](../Propuesta.md) — **inmutable, no editar bajo
> ningún concepto.** Define los 5 modelos, la contribución innovadora, las anclas Q1 y las revistas
> objetivo. Si algo aquí contradice a `Propuesta.md`, gana `Propuesta.md`.

## 1. Qué hay que hacer

1. **Analizar los resultados** de la ejecución real (perfil `full`, GPU) del arnés de anclas: leer las
   cifras, compararlas con los umbrales Q1, y producir tablas/figuras y la narrativa de resultados.
2. **Redactar el artículo** completo en **inglés**, en **LaTeX** con el template de **IEEE TNNLS**,
   listo para someter (o tan cerca como sea posible con los datos disponibles).

## 2. Revista objetivo y formato

- **Revista:** *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)* (JCR Q1, CS-AI).
- **Idioma:** inglés.
- **Formato:** LaTeX con la clase `IEEEtran`. Template descargado en
  [`../IEEE-Transactions-LaTeX2e-templates-and-instructions/`](../IEEE-Transactions-LaTeX2e-templates-and-instructions/):
  - `IEEEtran.cls` — la clase.
  - `bare_jrnl_new_sample4.tex` — **esqueleto de partida** (artículo de journal). Copiar y rellenar.
  - `New_IEEEtran_how-to.pdf` / `.tex` — instrucciones de uso del template.
  - ⚠️ **Este directorio está actualmente fuera de control de versiones** (no está en git). Si trabajas
    desde un clon limpio del repo, no estará presente: pídelo al usuario o que se commitee primero.
- Coloca el artículo en `paper/` (créalo): `paper/main.tex`, `paper/refs.bib`, `paper/figs/`.

## 3. Dónde están los resultados (¡ojo: aún no en el repo!)

Los artefactos de resultados **no están versionados** — `runs/` está en `.gitignore`. La ejecución real
se hace en Colab/GPU con:

```bash
python notebooks/run_anchors.py --profile full --anchors all --out-dir runs/anchors_full
```

El usuario debe **entregarte** estos artefactos (commitearlos en p. ej. `results/`, o pasártelos):

- `runs/anchors_full/anchors_report.json` — **reporte agregado**, la fuente primaria de cifras del paper.
  Por ancla: `target`, `achieved`, `baseline`, `passed`, `run_dir`; más `profile_name`, `seed`,
  `config_hash`, `dependencies` (python/torch/numpy).
- Por cada ancla, dentro de su `run_dir`:
  - `metrics.json` — métricas detalladas (latencias media/p99, matrices de confusión de M2,
    historiales de win_rate, suppressed_fraction, `global` de M4, etc.).
  - `run_meta.json` — seed, hiperparámetros completos, `config_hash`, hashes de datos, versiones de
    dependencias → **base de la sección de reproducibilidad**.

> Si solo recibes el `anchors_report.json` (no los `metrics.json`), puedes escribir Resultados pero te
> faltarán latencias, matrices de confusión y curvas para las figuras: pídelos.

**`quick` vs `full`:** las cifras publicables salen **solo del perfil `full`**. El perfil `quick` es
validación del arnés (CPU, entrenamientos diminutos) y **no** alcanza los umbrales (de hecho produce
`relative_improvement = ∞` con baseline 0); no uses sus números en el paper salvo para describir, si
acaso, la metodología de validación del pipeline.

## 4. Las cinco contribuciones y sus anclas Q1

Umbrales en [`Propuesta.md`](../Propuesta.md) y resumidos en la tabla "done publicable" de
[`docs/ROADMAP.md`](ROADMAP.md). Mapa ancla → métrica → código que la produce:

| Ancla (clave JSON) | Modelo | Métrica reportada | Umbral Q1 | Producida por |
|---|---|---|---|---|
| `jamming` | M1 Deep RL jamming | `cognitive.win_rate` vs `baseline.win_rate` (librería) | ≥ 0.92 (baseline ≈ 0.58) | `deep_rl_jamming.compare.compare` |
| `elint` | M2 Temporal CNN ELINT | `lpi_accuracy` (y `macro_acc_type`/`_mode`) | ≥ 0.96 (vs < 0.65) | `temporal_cnn_elint.train.train` |
| `marl` | M3 MARL formación | `relative_improvement.suppressed_fraction` (QMIX coord. vs IQL indep.) | ≥ 0.45 | `marl_formation.compare.compare_policies` |
| `gan` | M4 GAN señales | `relative_improvement` (M2 aumentado vs baseline, emisores retenidos) | ≥ 0.22 | `gan_signals.robustness.run_robustness_experiment` |
| (volumen M4) | M4 GAN señales | nº de señales sintéticas exportadas (50+ tipos) | > 200 000 | `gan_signals.export.export_synthetic` |
| M5 Baseline | `ew_library` | — (referencia de comparación de M1 y M3, sin ancla propia) | — | `ew_library.library` |

**Latencia** es métrica de primera clase (ver `CLAUDE.md` y la tabla "done publicable"): M1 inferencia
< 5 ms, M2 < 1 ms (media + p99). Esos valores están en los `metrics.json` (`latency_mean_ms`,
`latency_p99_ms`) — repórtalos.

## 5. Material para *related work* y motivación

- [`docs/research/estado-del-arte.md`](research/estado-del-arte.md) — estado del arte ya redactado
  (huecos del SOTA por modelo, justificación de cada contribución).
- [`docs/articles/`](articles/) — **corpus de PDFs de referencia** (uno por técnica) con
  [`README.md`](articles/README.md) que mapea cada PDF a su modelo y referencia. Úsalo para construir
  `refs.bib` y la sección de trabajo relacionado. Incluye además refs paywall a recuperar manualmente.
- Specs de diseño en [`docs/superpowers/specs/`](superpowers/specs/) — fundamentan las decisiones de
  arquitectura de cada modelo (útiles para la sección de método).

## 6. Estructura sugerida del artículo (TNNLS, un solo paper integrado)

Supuesto de alcance: **un único artículo** que presenta el sistema de GE cognitiva integrado (los 5
modelos como componentes de una contribución coherente), no cinco papers separados. Confirmar con el
usuario antes de fijar el outline (ver §8).

1. **Abstract** — sistema, los 5 componentes, las cifras ancla principales, validado en escenario NTTR.
2. **Introduction** — relevancia de la GE cognitiva, huecos del SOTA, contribución.
3. **Related Work** — desde `estado-del-arte.md` + `docs/articles/`.
4. **System Architecture / Methods** — los 5 modelos (RL jamming, Temporal CNN ELINT, MARL formación,
   GAN señales, baseline) y cómo encajan; remitir a specs para detalle.
5. **Experimental Setup** — dataset/escenario, perfil `full`, hardware (Colab GPU), y **reproducibilidad**
   (seeds, `config_hash`, versiones — desde `run_meta.json`).
6. **Results** — una subsección por ancla con la métrica `achieved` vs `target` vs `baseline`, latencias,
   matrices de confusión (M2), y la comparación M1/M3 contra el baseline M5. Tablas + figuras.
7. **Discussion** — interpretación, limitaciones.
8. **Conclusion**.

## 7. Restricciones duras (no negociables)

- **`Propuesta.md` es inmutable.** No editarla, reescribirla ni reformatearla.
- **No exponer en el paper parámetros de amenazas reales ni datos EW sensibles.** Todos los parámetros
  del proyecto son **sintéticos/publicados** (catálogos del propio repo); el artículo debe dejarlo claro
  y no incluir nada que sugiera datos clasificados.
- **No sobre-afirmar.** Las afirmaciones de resultados deben respaldarse con los `achieved` reales del
  `full`. Si una ancla **no** alcanza su umbral en la corrida real, repórtalo con honestidad (resultado
  parcial / trabajo futuro), no lo maquilles.
- Las cifras del paper salen del perfil **`full`**, nunca del `quick`.
- Citar el corpus de `docs/articles/` correctamente; recuperar las refs paywall si se necesitan a texto
  completo.

## 8. Decisiones de alcance — confirmar con el usuario antes de redactar

- **Un paper integrado vs varios** (asumido: uno integrado). Si el plan es un paper por modelo, el
  outline y el alcance cambian.
- **¿Se commitean los resultados** (`anchors_report.json` + `metrics.json` + `run_meta.json`) en un
  `results/` versionado, o se entregan a mano? Sin ellos no hay análisis posible.
- **¿Se commitea el template IEEE** y el directorio `paper/` en el repo?
- Orden de autores, afiliaciones y datos de sometimiento (los pone el usuario).
