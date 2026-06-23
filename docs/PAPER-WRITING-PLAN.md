# Paper Brainstorming and Writing Plan

## 1. Decisiones confirmadas

- **Tipo de artículo:** un único paper integrado, no una serie de papers por modelo.
- **Revista objetivo:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS).
- **Idioma:** inglés.
- **Formato:** LaTeX con `IEEEtran`, usando el template de journal disponible en `IEEE-Transactions-LaTeX2e-templates-and-instructions/`.
- **Título exacto:** “Cognitive Electronic Warfare in the NTTR Electronic Combat Range using Deep RL: Adaptive Jamming, Electronic Protection, and Deception Operations in the World's Densest Simulated Threat Environment”.
- **Autoría inicial:** Hugo Fernández Solís — Projener.ai — hugofernandezsolis@gmail.com.
- **Tratamiento de NTTR:** escenario sintético inspirado en NTTR, no recreación operacional real. Redacción recomendada: “an NTTR-inspired simulated electronic combat range”.
- **Nivel de acabado:** paper académico completo listo para enviar, con limitaciones y future work explícitos.
- **Materiales a versionar:** `paper/`, `paper/figs/`, `paper/refs.bib`, resultados aceptados en `results/`, y template IEEE necesario para compilar.
- **Fuente de verdad del proyecto:** `Propuesta.md`, inmutable.
- **Reporte aceptado:** el reporte full entregado en `runs (1)/runs/anchors_full/anchors_report.json` se toma como referencia actual para resultados.

## 2. Brainstorming editorial

### Narrativa central

El paper debe presentar un sistema integrado de Cognitive Electronic Warfare en un campo de combate electrónico sintético inspirado en NTTR. La contribución no es un único modelo aislado, sino una arquitectura completa con cinco piezas coordinadas:

1. Deep RL para jamming adaptativo.
2. Temporal CNN ELINT v2 para clasificación de emisor, modo y amenaza.
3. MARL para coordinación EW en formación.
4. GAN para generación de señales de amenaza sintéticas y mejora de robustez.
5. Librería EW convencional como baseline reproducible.

La narrativa debe ser ambiciosa pero defendible: el sistema se evalúa sobre datos y entornos sintéticos del repositorio, no sobre datos clasificados ni parámetros operacionales reales. La frase “NTTR-inspired simulated electronic combat range” permite conservar el valor narrativo del título sin sobre-afirmar.

### Mensaje de resultados

El reporte full aceptado muestra una historia fuerte:

| Modelo | Resultado | Lectura editorial |
|---|---:|---|
| M1 Deep RL jamming | `achieved=1.0` vs target `0.92`; baseline `0.705` | El agente supera claramente al baseline convencional. |
| M2 ELINT v2 | accuracy estricta mínima `0.98445`; LPI `1.0`; p99 `1.3045 ms` | La clasificación cumple; la reducción de latencia p99 queda como mejora futura. |
| M3 MARL | `achieved=3.18848` vs target `0.45`; baseline `0.23875` | La coordinación supera ampliamente la actuación independiente. |
| M4 GAN | `achieved=0.99528` vs target `0.22`; baseline held-out `0.0` | La aumentación sintética mejora de forma sustancial la robustez frente a emisores retenidos. |

El abstract y results deben enfatizar que M1, M3 y M4 pasan sus anclas, y que M2 alcanza accuracy estricta pero no el p99 de latencia. Esa limitación debe aparecer como future work, no esconderse.

### Posicionamiento para TNNLS

El paper debe conectar con TNNLS mediante:

- aprendizaje profundo y deep RL aplicados a señales EW;
- evaluación multi-modelo con anclas cuantitativas;
- reproducibilidad mediante `config_hash`, seeds, versiones y artefactos;
- integración de discriminative learning, reinforcement learning, multi-agent learning y generative modeling.

La parte “electronic warfare” debe escribirse con prudencia: todos los escenarios, emisores y señales son sintéticos o proceden de parámetros publicados del repositorio.

## 3. Estructura propuesta del artículo

1. **Abstract**
   - Sistema integrado.
   - Escenario NTTR-inspired.
   - Cinco componentes.
   - Resultados principales: M1 100% win rate, M2 98.45% strict type/mode/threat/LPI accuracy floor with p99 latency limitation, M3 3.188 relative improvement, M4 99.53% held-out improvement.

2. **Introduction**
   - Problema: EW cognitiva en entornos densos y cambiantes.
   - Limitaciones de librerías fijas y pipelines aislados.
   - Contribución integrada.
   - Resumen de resultados.

3. **Related Work**
   - DRL jamming y anti-jamming.
   - Radar/ELINT classification.
   - MARL cooperative jamming/coordination.
   - GAN/cWGAN-GP para señales radar.
   - Conventional EW response libraries.

4. **System Architecture**
   - Vista global M1-M5.
   - Flujo: synthetic PDW/IQ data → ELINT classification → jamming/coordination decisions → synthetic augmentation → baseline comparisons.
   - Separar claramente training, evaluation y baseline.

5. **Methods**
   - M1: D3QN adaptive jamming.
   - M2: TemporalCNNV2 with strict type/mode/threat/LPI evaluation.
   - M3: QMIX coordinated formation vs IQL independent baseline.
   - M4: cWGAN-GP synthetic PDW generator and robustness experiment.
   - M5: deterministic EW response library.

6. **Experimental Setup**
   - Full profile GPU execution.
   - Accepted report path and reproducibility metadata.
   - Synthetic NTTR-inspired electronic combat range.
   - Seeds, dependencies, config hashes.
   - No classified or sensitive EW data.

7. **Results**
   - Main anchor table.
   - M1 comparison against baseline.
   - M2 strict metrics and confusion/latency discussion.
   - M3 coordinated vs independent comparison.
   - M4 export validity and robustness improvement.

8. **Discussion**
   - Integrated cognitive EW implications.
   - Why M2 accuracy is publishable while p99 latency is future work.
   - Robustness and synthetic data interpretation.
   - Threats to validity: synthetic environment, no real operational NTTR data, no classified waveforms.

9. **Conclusion**
   - Integrated system contribution.
   - Main empirical outcomes.
   - Next steps: M2 latency optimization, richer scenario validation, broader synthetic threat catalog.

## 4. Figuras y tablas previstas

### Tablas

- **Table I:** Summary of system components M1-M5.
- **Table II:** Full anchor results from accepted report.
- **Table III:** M2 strict metrics and latency.
- **Table IV:** M4 synthetic validity and robustness improvement.
- **Table V:** Reproducibility metadata: seed, config hash, Python/Torch/NumPy versions.

### Figuras

- **Fig. 1:** Integrated Cognitive EW architecture.
- **Fig. 2:** NTTR-inspired simulated electronic combat range workflow.
- **Fig. 3:** M2 confusion matrix by emitter type.
- **Fig. 4:** M1/M3 baseline comparison bars.
- **Fig. 5:** M4 robustness baseline vs augmented classifier.

Figures should be generated from versioned result files in `results/anchors_full_accepted/`, not from ad hoc screenshots.

## 5. Versioned file layout to create

```text
paper/
  main.tex
  refs.bib
  figs/
    architecture.pdf
    nttr_workflow.pdf
    m2_confusion_type.pdf
    anchor_results.pdf
    gan_robustness.pdf
  ieee/
    IEEEtran.cls
results/
  anchors_full_accepted/
    anchors_report.json
    elint/
      metrics.json
      run_meta.json
    gan/
      metrics.json
      run_meta.json
      robustness/
        metrics.json
        run_meta.json
      wgan_gp/
        metrics.json
        run_meta.json
    jamming/
      metrics.json
      run_meta.json
    marl/
      qmix/
        metrics.json
        run_meta.json
      iql/
        metrics.json
        run_meta.json
scripts/
  paper/
    build_figures.py
    extract_results_tables.py
```

Large binary files should not be copied unless needed for reproducibility of figures. The initial version should copy JSON metrics and metadata. Checkpoints and HDF5 files can remain out of git unless the user explicitly requests a heavyweight artifact package.

## 6. Implementation and writing plan

### Phase 1 — Stage accepted results

1. Create `results/anchors_full_accepted/`.
2. Copy JSON files from the accepted full run.
3. Exclude `best.pt`, `synthetic.h5` and `Zone.Identifier` files from the versioned results.
4. Add a small `results/anchors_full_accepted/README.md` explaining provenance and that these are synthetic Colab/GPU results.
5. Verify JSON files can be parsed.

### Phase 2 — Set up paper skeleton

1. Create `paper/`.
2. Copy `IEEEtran.cls` into `paper/ieee/` or reference the existing template path if that is cleaner.
3. Create `paper/main.tex` from the IEEE journal sample.
4. Create `paper/refs.bib` with references from `docs/articles/README.md` and `docs/research/estado-del-arte.md`.
5. Add author, affiliation, email and exact title.
6. Set the framing to “NTTR-inspired simulated electronic combat range”.

### Phase 3 — Generate tables and figures

1. Create `scripts/paper/extract_results_tables.py`.
2. Create `scripts/paper/build_figures.py`.
3. Generate machine-readable tables from JSON results.
4. Generate figures into `paper/figs/`.
5. Make scripts deterministic and rerunnable from the repo root.

### Phase 4 — Draft the manuscript

1. Write Abstract and Introduction.
2. Write Related Work using local research docs and PDFs.
3. Write System Architecture and Methods from specs.
4. Write Experimental Setup from `run_meta.json`.
5. Write Results from accepted JSON artifacts.
6. Write Discussion with M2 latency as future work.
7. Write Conclusion.
8. Ensure every numerical claim maps to a result file.

### Phase 5 — Reproducibility and safety review

1. Check that no text implies classified data or operational NTTR fidelity.
2. Check that `Propuesta.md` remains untouched.
3. Check that quick-profile numbers are not used as paper results.
4. Check that M2 p99 latency is reported honestly as a limitation.
5. Check that M4 uses the finite robustness score semantics.

### Phase 6 — Build and polish

1. Compile LaTeX locally with `latexmk` or `pdflatex` if available.
2. Fix references, overfull boxes and missing figures.
3. Confirm all figures render in black-and-white.
4. Run a final grep for placeholders such as `TODO`, `TBD`, `??`, `citation needed`.
5. Commit paper, results JSON, scripts and figures.

## 7. Open points before drafting

- Confirm whether to version only JSON results or also large artifacts (`best.pt`, `synthetic.h5`).
- Confirm whether the exact long title should also appear as the running title, or whether to use a shorter running head.
- Confirm whether `Projener.ai` should be written exactly with that capitalization in the IEEE author block.

## 8. Recommended next action

Proceed with Phase 1 and Phase 2 first: stage accepted JSON results, create `paper/`, copy the IEEE class, and build a compilable `main.tex` skeleton. Once the skeleton compiles, draft section by section.
