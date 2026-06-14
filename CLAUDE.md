# Cognitive Electronic Warfare System — CLAUDE.md

## Idea base del proyecto

> **Fuente de verdad:** la idea base a implementar está en **[`Propuesta.md`](Propuesta.md)** — contexto y relevancia (GE cognitiva en el NTTR), los 5 modelos/técnicas, el dataset/benchmark, las revistas Q1 objetivo y la contribución innovadora. **Ese documento manda; los modelos deben alinearse con él.**
>
> ⚠️ **`Propuesta.md` NO debe modificarse bajo ningún concepto.** Es un documento de solo lectura: no editarlo, reescribirlo ni reformatearlo. Si surge una discrepancia, se adapta el código o este CLAUDE.md, nunca `Propuesta.md`.

Este CLAUDE.md cubre **el *cómo*** (stack, estructura, convenciones, reproducibilidad, calidad de código y flujo de trabajo), no el *qué*. Si algo de aquí contradice a `Propuesta.md`, gana `Propuesta.md`.

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

- No commitear sin que se solicite explícitamente, **salvo que una skill en ejecución lo indique como paso de su flujo** (p. ej. `superpowers:brainstorming` commitea el spec de diseño)
- No exponer en logs ni artefactos parámetros de amenazas reales o datos sensibles de EW

## Flujo de trabajo habitual

1. Usar `superpowers` por defecto para código o diseño del peoyecto
2. Nuevas features/modelos: usar `superpowers:brainstorming` antes de implementar
3. Implementar lógica crítica con `superpowers:test-driven-development` (preprocesado de señales, métricas EW, reward del entorno RL)
4. Bugs o comportamiento inesperado (no converge, NaN, etc.): usar `superpowers:systematic-debugging`
5. Antes de declarar algo completo: usar `superpowers:verification-before-completion`
6. Al terminar una rama: usar `superpowers:finishing-a-development-branch`

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
