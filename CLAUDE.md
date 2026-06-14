# Cognitive Electronic Warfare System — CLAUDE.md

## Contexto del proyecto

TFM sobre **Guerra Electrónica Cognitiva** validada en el entorno NTTR (Nevada Test and Training Range), con los ranges PBECR y TPECR. El objetivo es publicar en revista Q1 (IEEE TNNLS, Knowledge-Based Systems, Information Sciences, Expert Systems with Applications).

El sistema combate electrónico cognitivo basado en Deep RL que debe superar al adversario en >92% de enfrentamientos de espectro con latencias de respuesta <5ms.

## Modelos a implementar

| # | Modelo | Objetivo |
|---|--------|----------|
| 1 | **Deep RL** (jamming adaptativo) | Agente que genera en tiempo real la combinación óptima de técnicas de jamming (noise, DRFM, cross-eye, VGPO, RGPO) adaptándose a cambios de waveform y ECCM adversario. Latencia <5ms. |
| 2 | **Temporal CNN** (ELINT/clasificación) | Clasifica en tiempo real el tipo de emisor (SA-20, S-400, radar AESA, etc.), modo de operación y estado de amenaza desde secuencias de pulsos RWR. Latencia <1ms. |
| 3 | **Multi-Agent RL** (coordinación en formación) | MARL donde cada aeronave es un agente EW que coordina emisiones para maximizar supresión del IADS (distribución de tareas, gestión de potencia, deception coordinada). |
| 4 | **GAN** (señales sintéticas) | Genera señales radar sintéticas realistas (radar cognitivo, LPI, banda ancha, pasivo) para data augmentation de los modelos de clasificación y jamming. |
| 5 | **Librería EW convencional** (baseline) | Contramedidas pre-programadas por tipo de amenaza identificada. Referencia de comparación para demostrar mejora de los modelos RL/CNN. |

## Datasets

- **DeepSig RadioML** — clasificación RF con modulaciones reales a diferentes SNR
- **DARPA SC2** (Spectrum Collaboration Challenge) — inteligencia espectral
- **DARPA RFMLS** (RF Machine Learning Systems) — señales RF para ML
- **GNU Radio Signal Data** — señales generadas/capturadas
- **NTTR SIADS params** — parámetros publicados de simuladores SA-2 a S-400

Los datasets no son cerrados; se pueden usar alternativas o datasets sintéticos generados por el modelo GAN del proyecto.

## Stack tecnológico

- **Lenguaje**: Python 3.11+
- **Gestión de dependencias**: `uv` + `pyproject.toml`
- **Deep learning**: PyTorch
- **Procesamiento de señales**: NumPy, SciPy, GNU Radio (gnuradio), torchaudio
- **RL**: Stable-Baselines3, RLlib, o implementación propia en PyTorch
- **Entrenamiento**: Google Colab / Kaggle (notebooks deben ser autocontenidos y exportables)
- **Tracking de experimentos**: Weights & Biases (wandb) o MLflow
- **Linting**: ruff
- **Tipos**: mypy (strict donde sea práctico)
- **Tests**: pytest

## Convenciones de código

### General
- Python moderno: type hints en todas las funciones públicas, dataclasses o Pydantic para configs
- No añadir comentarios que expliquen *qué* hace el código — solo el *por qué* cuando no es obvio
- Sin abstracciones prematuras; no diseñar para requisitos hipotéticos futuros

### Estructura de proyecto
```
cog_ew/
├── configs/                        # Configs YAML de experimentos por modelo
│   ├── deep_rl_jamming/
│   ├── temporal_cnn_elint/
│   ├── marl_formation/
│   ├── gan_signals/
│   └── ew_library/
├── data/                           # Datasets compartidos por todos los modelos
│   ├── raw/                        # Datos descargados (RadioML, DARPA, etc.)
│   ├── processed/                  # Datos preprocesados listos para entrenar
│   └── synthetic/                  # Señales generadas por la GAN
├── notebooks/                      # Notebooks Colab/Kaggle
├── src/cog_ew/
│   ├── data/                       # Utilidades compartidas
│   │   ├── loaders.py              # Dataset loaders para todos los modelos
│   │   └── preprocessing.py        # Preprocesado de señales (IQ, PDW)
│   ├── deep_rl_jamming/            # Modelo 1: Deep RL jamming adaptativo
│   │   ├── agent.py
│   │   ├── env.py                  # Simulación ciclo radar (PRI, ECCM)
│   │   └── train.py
│   ├── temporal_cnn_elint/         # Modelo 2: Temporal CNN clasificación ELINT
│   │   ├── model.py
│   │   └── train.py
│   ├── marl_formation/             # Modelo 3: MARL coordinación en formación
│   │   ├── agents.py
│   │   ├── env.py                  # Entorno multi-agente IADS
│   │   └── train.py
│   ├── gan_signals/                # Modelo 4: GAN señales sintéticas
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── train.py
│   └── ew_library/                 # Modelo 5: Baseline convencional
│       └── library.py
├── tests/                          # Espejo de src/, un directorio por modelo
│   ├── data/
│   ├── deep_rl_jamming/
│   ├── temporal_cnn_elint/
│   ├── marl_formation/
│   ├── gan_signals/
│   └── ew_library/
├── pyproject.toml
└── README.md
```

### Reproducibilidad (prioridad alta)
- **Seeds explícitos** en todos los experimentos: `torch.manual_seed`, `numpy.random.seed`, `random.seed`
- **Configs versionadas** en YAML (nunca hardcodear hiperparámetros en el código)
- **Loguear siempre**: seed, versión de dependencias, hash del dataset, hiperparámetros completos
- Usar `wandb`, `mlflow` o `trackio` (skill `huggingface-skills:huggingface-trackio`) para tracking; cada experimento debe ser reproducible desde su config
- Para lanzar entrenamientos de forma reproducible, seguir el skill `/experiment-run` (verifica config, seed, datos y registra metadatos)
- Los notebooks de Colab/Kaggle deben incluir celda de setup completa (instalación, seed, config)

### Calidad de código
- `ruff check` y `ruff format` antes de cualquier commit
- `mypy` para módulos en `src/` (al menos las interfaces públicas)
- Tests en `pytest` para lógica crítica: preprocesado de señales, métricas EW, lógica del entorno RL
- No mockear lo que no es necesario mockear; tests de integración donde sea práctico

## Consideraciones de dominio EW

- Las métricas clave son: **J/S ratio** (Jamming-to-Signal), **burnthrough range**, **probability of intercept (POI)**, **false alarm rate**
- Los entornos RL simulan ciclos radar (PRI, frecuencia, modo ECCM) — modelar con fidelidad suficiente para que los resultados sean publicables
- La latencia es crítica: los modelos de inferencia deben perfilar tiempo de ejecución; documentar latencia media y p99
- Los datos de señales son series temporales de pulsos (IQ samples, PDW — Pulse Descriptor Words); mantener pipelines separados para cada representación

## Lo que Claude debe evitar

- No commitear sin que se solicite explícitamente
- No exponer en logs ni artefactos parámetros de amenazas reales o datos sensibles de EW

## Flujo de trabajo habitual

1. Nuevas features/modelos: usar `superpowers:brainstorming` antes de implementar
2. Implementar lógica crítica con `superpowers:test-driven-development` (preprocesado de señales, métricas EW, reward del entorno RL)
3. Bugs o comportamiento inesperado (no converge, NaN, etc.): usar `superpowers:systematic-debugging`
4. Antes de declarar algo completo: usar `superpowers:verification-before-completion`
5. Al terminar una rama: usar `superpowers:finishing-a-development-branch`

## Plugins y skills recomendados (cuándo usarlos)

### Núcleo — uso habitual
- **`context7`** (MCP): consultar SIEMPRE que se use la API de PyTorch, torchvision, Stable-Baselines3 o cualquier librería externa. Evita alucinaciones de API.
- **`huggingface-skills:huggingface-datasets`**: explorar y cargar datasets RF (RadioML y derivados).
- **`huggingface-skills:huggingface-trackio`**: tracking ligero de experimentos (alternativa a W&B/MLflow).
- **`huggingface-skills:hf-mem`**: estimar memoria de un modelo antes de entrenar — crítico por la VRAM limitada (GTX 1060, 6GB).
- **`huggingface-skills:trl-training` / `huggingface-llm-trainer`**: solo si se entrena en jobs cloud de HF.

### Calidad de código — al cerrar cada pieza
- **`/code-review`**: revisar el diff antes de commits importantes de un modelo.
- **`pr-review-toolkit`** (`pr-test-analyzer`): verificar cobertura de tests de la lógica de señales/métricas.
- **`/security-review`** (security-guidance) + subagente `security-reviewer`: comprobar que no se exponen datos sensibles de EW.
- **`code-simplifier`**: pasar tras prototipar rápido, antes de dejar el código publicable.

### Flujo git / mantenimiento — bajo demanda
- **`commit-commands`** (`/commit`, `/commit-push-pr`) + **`github`** (MCP): commits y PRs (requiere `GITHUB_PERSONAL_ACCESS_TOKEN`).
- **`remember`**: guardar estado al final de sesiones largas de implementación.
- **`claude-md-management`**: actualizar este fichero cuando el proyecto crezca.

### No aplican a este proyecto
- `playwright` (sin frontend web), `microsoft-docs` (sin Azure/.NET), `clangd-lsp` (proyecto Python puro), `data-engineering`/Airflow (salvo que se monten pipelines ETL recurrentes).
