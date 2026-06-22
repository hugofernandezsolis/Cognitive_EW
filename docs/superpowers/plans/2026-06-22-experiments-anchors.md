# Fase 6 · Sub-pieza A — Arnés unificado de experimentos + reporte de anclas Q1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un paquete `src/cog_ew/experiments/` que orquesta los pipelines entrenables de M1–M4 y agrega sus métricas ancla Q1 en un único `anchors_report.json` reproducible, testeable en CPU (perfil `quick`) y listo para Colab GPU (perfil `full`).

**Architecture:** `anchors.py` define `AnchorResult` y cuatro funciones runner (una por modelo) que ejecutan el pipeline real del modelo en un subdirectorio de salida y devuelven el resultado del ancla. `report.py` define `ExperimentProfile` (selección de configs + knobs de duración por perfil), el dict `ANCHOR_RUNNERS` y `run_anchors`, que ejecuta los runners seleccionados y escribe el reporte agregado. Un notebook fino `notebooks/run_anchors.py` es la entrada CLI para Colab. Los perfiles viven en `configs/experiments/{quick,full}.yaml`.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, h5py, pytest; herramientas vía `.venv/bin/<tool>`.

## Global Constraints

- **`Propuesta.md` NO se modifica** bajo ningún concepto.
- **No exponer en logs ni artefactos parámetros de amenazas reales** — solo catálogos sintéticos del proyecto.
- **`torch.load` siempre con `weights_only=True`.**
- **No commitear `.claude/`.** `changes.md` es scratch — no commitear. Los PDF de `docs/articles/` no se añaden a `.gitignore`.
- **Reproducibilidad:** seeds explícitos; hiperparámetros versionados en YAML; cada pipeline ya escribe su `run_meta.json`/`metrics.json`.
- **Calidad:** `ruff check`, `ruff format`, `mypy` (interfaces públicas) antes de cada commit; type hints en funciones públicas; sin comentarios que expliquen el *qué*.
- **Targets de anclas Q1 (invariantes, no configurables):** `jamming` `win_rate ≥ 0.92`; `elint` `lpi_accuracy ≥ 0.96`; `marl` `relative_improvement.suppressed_fraction ≥ 0.45`; `gan` `relative_improvement ≥ 0.22`.
- **Regla `passed`:** `math.isfinite(achieved) and achieved >= target` (el guard de finitud evita que `inf` apruebe de forma vacua).
- Mensajes de commit terminan con:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2`

## File Structure

- `src/cog_ew/experiments/__init__.py` — paquete nuevo (vacío).
- `src/cog_ew/experiments/anchors.py` — `AnchorResult`, `_TARGETS`, `_passed`, los 4 runners.
- `src/cog_ew/experiments/report.py` — `ExperimentProfile` (+`from_yaml`), `ANCHOR_RUNNERS`, `run_anchors`.
- `configs/experiments/quick.yaml` — perfil CPU diminuto.
- `configs/experiments/full.yaml` — perfil Colab GPU (duraciones reales vía `null`).
- `notebooks/run_anchors.py` — entrada CLI fina (sustituye a `colab_train_models.py`, que se elimina).
- `tests/experiments/__init__.py`, `tests/experiments/test_anchors.py`, `tests/experiments/test_report.py`, `tests/experiments/test_run_anchors_cli.py`, `tests/experiments/test_gpu_readiness.py`.

**Evitar importación circular:** `anchors.py` usa `from __future__ import annotations` y anota `profile: "ExperimentProfile"` con `if TYPE_CHECKING: from cog_ew.experiments.report import ExperimentProfile`. `report.py` importa los runners desde `anchors.py` en tiempo de ejecución. Los runners acceden a los atributos del profile por duck-typing, sin importar la clase en runtime.

---

### Task 1: GPU-readiness — `torch.cuda.manual_seed_all` en todos los `_set_seeds`

Añade la siembra CUDA a cada `_set_seeds` de los pipelines entrenables para reproducibilidad en runs GPU. Cambio mínimo y aislado, prerrequisito de los runners.

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/train.py` (`_set_seeds`)
- Modify: `src/cog_ew/temporal_cnn_elint/train.py` (`_set_seeds`)
- Modify: `src/cog_ew/marl_formation/train.py` (`_set_seeds`)
- Modify: `src/cog_ew/gan_signals/train.py` (`_set_seeds`)
- Modify: `src/cog_ew/gan_signals/export.py` (`_set_seeds`)
- Modify: `src/cog_ew/gan_signals/robustness.py` (`_set_seeds`)
- Test: `tests/experiments/test_gpu_readiness.py`

**Interfaces:**
- Consumes: nada.
- Produces: `_set_seeds(seed: int) -> None` en cada módulo llama a `torch.cuda.manual_seed_all(seed)` además de random/numpy/torch.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/__init__.py` (empty) and `tests/experiments/test_gpu_readiness.py`:

```python
import importlib

import pytest

MODULES = [
    "cog_ew.deep_rl_jamming.train",
    "cog_ew.temporal_cnn_elint.train",
    "cog_ew.marl_formation.train",
    "cog_ew.gan_signals.train",
    "cog_ew.gan_signals.export",
    "cog_ew.gan_signals.robustness",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_set_seeds_seeds_cuda(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    calls: list[int] = []
    monkeypatch.setattr(module.torch.cuda, "manual_seed_all", lambda s: calls.append(s))
    module._set_seeds(123)
    assert calls == [123]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_gpu_readiness.py -v`
Expected: FAIL — `manual_seed_all` no se llama (`calls == []`) en cada módulo.

- [ ] **Step 3: Add the CUDA seeding line in each module**

In each of the six `_set_seeds`, append after `torch.manual_seed(seed)`:

```python
    torch.cuda.manual_seed_all(seed)
```

So each becomes:

```python
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_gpu_readiness.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`
Expected: sin errores.

```bash
git add src/cog_ew/deep_rl_jamming/train.py src/cog_ew/temporal_cnn_elint/train.py \
  src/cog_ew/marl_formation/train.py src/cog_ew/gan_signals/train.py \
  src/cog_ew/gan_signals/export.py src/cog_ew/gan_signals/robustness.py \
  tests/experiments/__init__.py tests/experiments/test_gpu_readiness.py
git commit -m "$(cat <<'EOF'
feat(experiments): GPU-readiness — cuda.manual_seed_all en _set_seeds

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 2: `AnchorResult`, targets y guard `_passed`

Andamiaje de `anchors.py`: el dataclass de resultado, los targets invariantes y el guard de finitud.

**Files:**
- Create: `src/cog_ew/experiments/__init__.py` (empty)
- Create: `src/cog_ew/experiments/anchors.py`
- Test: `tests/experiments/test_anchors.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `@dataclass(frozen=True) AnchorResult(name: str, target: float, achieved: float, baseline: float | None, passed: bool, run_dir: str)`
  - `_TARGETS: dict[str, float]` = `{"jamming": 0.92, "elint": 0.96, "marl": 0.45, "gan": 0.22}`
  - `_passed(achieved: float, target: float) -> bool` = `math.isfinite(achieved) and achieved >= target`

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/test_anchors.py`:

```python
import math

from cog_ew.experiments.anchors import _TARGETS, _passed, AnchorResult


def test_targets_are_the_q1_anchors():
    assert _TARGETS == {"jamming": 0.92, "elint": 0.96, "marl": 0.45, "gan": 0.22}


def test_passed_requires_finite_and_ge_target():
    assert _passed(0.93, 0.92) is True
    assert _passed(0.92, 0.92) is True
    assert _passed(0.91, 0.92) is False
    assert _passed(math.inf, 0.45) is False
    assert _passed(math.nan, 0.22) is False


def test_anchor_result_is_frozen():
    r = AnchorResult("elint", 0.96, 0.5, None, False, "/tmp/run")
    assert r.name == "elint"
    assert r.baseline is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py -v`
Expected: FAIL — `ModuleNotFoundError: cog_ew.experiments.anchors`.

- [ ] **Step 3: Write minimal implementation**

Create `src/cog_ew/experiments/__init__.py` (empty file).

Create `src/cog_ew/experiments/anchors.py`:

```python
"""Runners de las anclas Q1: ejecutan el pipeline de cada modelo y devuelven su AnchorResult."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from cog_ew.experiments.report import ExperimentProfile

_TARGETS: dict[str, float] = {"jamming": 0.92, "elint": 0.96, "marl": 0.45, "gan": 0.22}


@dataclass(frozen=True)
class AnchorResult:
    name: str
    target: float
    achieved: float
    baseline: float | None
    passed: bool
    run_dir: str


def _passed(achieved: float, target: float) -> bool:
    return math.isfinite(achieved) and achieved >= target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/__init__.py src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py
git commit -m "$(cat <<'EOF'
feat(experiments): AnchorResult, targets Q1 y guard de finitud

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 3: `ExperimentProfile` + `from_yaml` + perfiles `quick`/`full`

El profile selecciona las configs reales por modelo y los knobs de duración por perfil. Los knobs de tamaño son `int | None`: `None` = usar el valor del YAML del modelo (perfil `full`); un int = override diminuto (perfil `quick`). `device`, `seed` y los episodios de comparación (que no viven en ningún YAML de modelo) son siempre obligatorios.

**Files:**
- Create: `src/cog_ew/experiments/report.py`
- Create: `configs/experiments/quick.yaml`
- Create: `configs/experiments/full.yaml`
- Test: `tests/experiments/test_report.py`

**Interfaces:**
- Consumes: nada en runtime (las claves de los YAML).
- Produces:
  - `@dataclass(frozen=True) ExperimentProfile` con campos:
    `name: str`, `device: str`, `seed: int`,
    `jamming_config: str`, `jamming_responses_config: str`, `jamming_total_steps: int | None`, `jamming_eval_episodes: int | None`, `jamming_compare_episodes: int`,
    `elint_config: str`, `elint_epochs: int | None`,
    `marl_qmix_config: str`, `marl_iql_config: str`, `marl_total_episodes: int | None`, `marl_eval_episodes: int | None`, `marl_compare_episodes: int`,
    `gan_config: str`, `export_config: str`, `robustness_config: str`, `gan_total_steps: int | None`, `export_samples_per_type: int | None`, `robustness_epochs: int | None`
  - `ExperimentProfile.from_yaml(path: str | Path) -> ExperimentProfile`

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/test_report.py`:

```python
from cog_ew.experiments.report import ExperimentProfile


def test_quick_profile_loads_from_yaml():
    profile = ExperimentProfile.from_yaml("configs/experiments/quick.yaml")
    assert profile.name == "quick"
    assert profile.device == "cpu"
    assert profile.jamming_total_steps is not None and profile.jamming_total_steps > 0
    assert profile.elint_epochs is not None
    assert profile.marl_compare_episodes > 0
    assert profile.jamming_config == "configs/deep_rl_jamming/train.yaml"


def test_full_profile_uses_null_for_yaml_durations():
    profile = ExperimentProfile.from_yaml("configs/experiments/full.yaml")
    assert profile.name == "full"
    assert profile.device == "cuda"
    assert profile.jamming_total_steps is None
    assert profile.elint_epochs is None
    assert profile.gan_total_steps is None
    assert profile.jamming_compare_episodes > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: cog_ew.experiments.report`.

- [ ] **Step 3: Write minimal implementation**

Create `src/cog_ew/experiments/report.py` (solo el profile por ahora; el resto se añade en Task 8):

```python
"""Perfiles de experimento y agregación del reporte de anclas Q1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    device: str
    seed: int
    jamming_config: str
    jamming_responses_config: str
    jamming_total_steps: int | None
    jamming_eval_episodes: int | None
    jamming_compare_episodes: int
    elint_config: str
    elint_epochs: int | None
    marl_qmix_config: str
    marl_iql_config: str
    marl_total_episodes: int | None
    marl_eval_episodes: int | None
    marl_compare_episodes: int
    gan_config: str
    export_config: str
    robustness_config: str
    gan_total_steps: int | None
    export_samples_per_type: int | None
    robustness_epochs: int | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentProfile:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)
```

Create `configs/experiments/quick.yaml`:

```yaml
name: quick
device: cpu
seed: 0
jamming_config: configs/deep_rl_jamming/train.yaml
jamming_responses_config: configs/ew_library/responses.yaml
jamming_total_steps: 200
jamming_eval_episodes: 5
jamming_compare_episodes: 5
elint_config: configs/temporal_cnn_elint/train.yaml
elint_epochs: 2
marl_qmix_config: configs/marl_formation/qmix.yaml
marl_iql_config: configs/marl_formation/iql.yaml
marl_total_episodes: 20
marl_eval_episodes: 5
marl_compare_episodes: 5
gan_config: configs/gan_signals/wgan_gp.yaml
export_config: configs/gan_signals/export.yaml
robustness_config: configs/gan_signals/robustness.yaml
gan_total_steps: 30
export_samples_per_type: 8
robustness_epochs: 2
```

Create `configs/experiments/full.yaml`:

```yaml
name: full
device: cuda
seed: 0
jamming_config: configs/deep_rl_jamming/train.yaml
jamming_responses_config: configs/ew_library/responses.yaml
jamming_total_steps: null
jamming_eval_episodes: null
jamming_compare_episodes: 200
elint_config: configs/temporal_cnn_elint/train.yaml
elint_epochs: null
marl_qmix_config: configs/marl_formation/qmix.yaml
marl_iql_config: configs/marl_formation/iql.yaml
marl_total_episodes: null
marl_eval_episodes: null
marl_compare_episodes: 200
gan_config: configs/gan_signals/wgan_gp.yaml
export_config: configs/gan_signals/export.yaml
robustness_config: configs/gan_signals/robustness.yaml
gan_total_steps: null
export_samples_per_type: null
robustness_epochs: null
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/report.py configs/experiments/quick.yaml configs/experiments/full.yaml tests/experiments/test_report.py
git commit -m "$(cat <<'EOF'
feat(experiments): ExperimentProfile + perfiles quick/full

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 4: `run_elint_anchor`

El runner más simple: entrena M2 y lee `lpi_accuracy`. Sin recarga de checkpoint ni baseline.

**Files:**
- Modify: `src/cog_ew/experiments/anchors.py`
- Test: `tests/experiments/test_anchors.py`

**Interfaces:**
- Consumes: `ExperimentProfile` (atributos `elint_config`, `elint_epochs`, `device`, `seed`); `cog_ew.temporal_cnn_elint.train.TrainConfig.from_yaml`, `.train(config) -> {"test": {"lpi_accuracy": float, ...}}`.
- Produces: `run_elint_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult` (name `"elint"`, target `0.96`, baseline `None`, run_dir `<out_dir>/elint`).

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_anchors.py`:

```python
from pathlib import Path

from cog_ew.experiments.anchors import run_elint_anchor
from cog_ew.experiments.report import ExperimentProfile

QUICK = "configs/experiments/quick.yaml"


def test_run_elint_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_elint_anchor(profile, tmp_path)
    assert result.name == "elint"
    assert result.target == 0.96
    assert result.baseline is None
    assert 0.0 <= result.achieved <= 1.0
    assert Path(result.run_dir).exists()
    assert (Path(result.run_dir) / "metrics.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_elint_anchor_quick -v`
Expected: FAIL — `ImportError: cannot import name 'run_elint_anchor'`.

- [ ] **Step 3: Write minimal implementation**

In `src/cog_ew/experiments/anchors.py`, add the runtime imports at the top of the module (after the existing imports, NOT under `TYPE_CHECKING`) and the helper + runner:

```python
from dataclasses import replace
from pathlib import Path

from cog_ew.temporal_cnn_elint.train import TrainConfig as ElintTrainConfig
from cog_ew.temporal_cnn_elint.train import train as train_elint


def _overrides(**kwargs: object) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if value is not None}


def run_elint_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "elint"
    config = ElintTrainConfig.from_yaml(profile.elint_config)
    config = replace(
        config,
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir),
        **_overrides(epochs=profile.elint_epochs),
    )
    result = train_elint(config)
    achieved = float(result["test"]["lpi_accuracy"])
    return AnchorResult(
        name="elint",
        target=_TARGETS["elint"],
        achieved=achieved,
        baseline=None,
        passed=_passed(achieved, _TARGETS["elint"]),
        run_dir=str(run_dir),
    )
```

Note: `Path` is now imported at runtime, so remove it from the `TYPE_CHECKING` block (keep only `from cog_ew.experiments.report import ExperimentProfile` under `TYPE_CHECKING`). Because `from __future__ import annotations` is active, `profile: ExperimentProfile` stays a string annotation — no circular import.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_elint_anchor_quick -v`
Expected: PASS (puede tardar varios segundos por el entrenamiento diminuto).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py
git commit -m "$(cat <<'EOF'
feat(experiments): run_elint_anchor (ancla M2 lpi_accuracy)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 5: `run_jamming_anchor`

Entrena M1 (D3QN), recarga el checkpoint en un agente fresco, lo envuelve como política cognitiva y lo compara contra la `BaselinePolicy` rule-based.

**Files:**
- Modify: `src/cog_ew/experiments/anchors.py`
- Test: `tests/experiments/test_anchors.py`

**Interfaces:**
- Consumes:
  - `cog_ew.deep_rl_jamming.train.TrainConfig.from_yaml`, `.train(config)` (escribe `<out_dir>/best.pt` = `agent.online_net.state_dict()`); campos `env: RadarEnvConfig`, `agent: D3QNConfig`, `total_steps`, `eval_episodes`, `device`, `seed`, `out_dir`.
  - `cog_ew.deep_rl_jamming.env.RadarJammingEnv(config.env)`; `.observation_space.shape`, `.action_space.n`.
  - `cog_ew.deep_rl_jamming.agent.D3QNAgent(obs_dim, n_actions, config.agent, device, rng)`; `.online_net.load_state_dict(...)`.
  - `cog_ew.deep_rl_jamming.compare.AgentPolicy(agent)`, `.BaselinePolicy(library, env)`, `.compare(env, cognitive, baseline, episodes, seed) -> {"cognitive": {"win_rate"}, "baseline": {"win_rate"}, ...}`.
  - `cog_ew.ew_library.library.EWResponseLibrary.from_yaml(path)`.
- Produces: `run_jamming_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult` (name `"jamming"`, target `0.92`, baseline = win_rate de la librería, run_dir `<out_dir>/jamming`).

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_anchors.py`:

```python
from cog_ew.experiments.anchors import run_jamming_anchor


def test_run_jamming_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_jamming_anchor(profile, tmp_path)
    assert result.name == "jamming"
    assert result.target == 0.92
    assert 0.0 <= result.achieved <= 1.0
    assert result.baseline is not None and 0.0 <= result.baseline <= 1.0
    assert (Path(result.run_dir) / "best.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_jamming_anchor_quick -v`
Expected: FAIL — `ImportError: cannot import name 'run_jamming_anchor'`.

- [ ] **Step 3: Write minimal implementation**

Add to the runtime imports of `src/cog_ew/experiments/anchors.py`:

```python
import numpy as np
import torch
from gymnasium.spaces import Discrete

from cog_ew.deep_rl_jamming.agent import D3QNAgent
from cog_ew.deep_rl_jamming.compare import AgentPolicy, BaselinePolicy, compare
from cog_ew.deep_rl_jamming.env import RadarJammingEnv
from cog_ew.deep_rl_jamming.train import TrainConfig as JammingTrainConfig
from cog_ew.deep_rl_jamming.train import train as train_jamming
from cog_ew.ew_library.library import EWResponseLibrary
```

Add the runner:

```python
def run_jamming_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "jamming"
    config = JammingTrainConfig.from_yaml(profile.jamming_config)
    config = replace(
        config,
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir),
        **_overrides(
            total_steps=profile.jamming_total_steps,
            eval_episodes=profile.jamming_eval_episodes,
        ),
    )
    train_jamming(config)

    env = RadarJammingEnv(config.env)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    obs_dim = int(np.prod(obs_shape))
    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    rng = np.random.default_rng(profile.seed)
    agent = D3QNAgent(obs_dim, n_actions, config.agent, profile.device, rng)
    state_dict = torch.load(run_dir / "best.pt", map_location=profile.device, weights_only=True)
    agent.online_net.load_state_dict(state_dict)

    library = EWResponseLibrary.from_yaml(profile.jamming_responses_config)
    cognitive = AgentPolicy(agent)
    baseline = BaselinePolicy(library, env)
    result = compare(env, cognitive, baseline, profile.jamming_compare_episodes, profile.seed)

    achieved = float(result["cognitive"]["win_rate"])
    baseline_wr = float(result["baseline"]["win_rate"])
    return AnchorResult(
        name="jamming",
        target=_TARGETS["jamming"],
        achieved=achieved,
        baseline=baseline_wr,
        passed=_passed(achieved, _TARGETS["jamming"]),
        run_dir=str(run_dir),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_jamming_anchor_quick -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py
git commit -m "$(cat <<'EOF'
feat(experiments): run_jamming_anchor (cognitivo D3QN vs librería)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 6: `run_marl_anchor`

Entrena un QMIX y un IQL diminutos, recarga ambos como `AgentPolicy` y los compara para obtener `relative_improvement.suppressed_fraction`.

**Files:**
- Modify: `src/cog_ew/experiments/anchors.py`
- Test: `tests/experiments/test_anchors.py`

**Interfaces:**
- Consumes:
  - `cog_ew.marl_formation.train.TrainConfig.from_yaml`, `.train(config)` (escribe `<out_dir>/best.pt` = `learner.agent.state_dict()`); campos `env: IADSEnvConfig`, `agent: QMIXConfig` (con `.hidden`), `total_episodes`, `eval_episodes`, `device`, `seed`, `out_dir`, `regime`.
  - `cog_ew.marl_formation.env.IADSFormationEnv(config.env)`; atributos `.obs_dim`, `.action_dim`, `.n_agents`.
  - `cog_ew.marl_formation.compare.AgentPolicy.from_checkpoint(path, *, obs_dim, action_dim, hidden, n_agents, device)`, `.compare_policies(env, *, coordinated, independent, episodes, seed) -> {"independent": {"suppressed_fraction"}, "relative_improvement": {"suppressed_fraction"}}`.
- Produces: `run_marl_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult` (name `"marl"`, target `0.45`, achieved = `relative_improvement.suppressed_fraction` (puede ser `inf`), baseline = `independent.suppressed_fraction`, run_dir `<out_dir>/marl`).

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_anchors.py`:

```python
import math

from cog_ew.experiments.anchors import run_marl_anchor


def test_run_marl_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_marl_anchor(profile, tmp_path)
    assert result.name == "marl"
    assert result.target == 0.45
    assert result.baseline is not None
    assert math.isfinite(result.achieved) or math.isinf(result.achieved)
    if math.isinf(result.achieved):
        assert result.passed is False
    assert (Path(result.run_dir) / "qmix" / "best.pt").exists()
    assert (Path(result.run_dir) / "iql" / "best.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_marl_anchor_quick -v`
Expected: FAIL — `ImportError: cannot import name 'run_marl_anchor'`.

- [ ] **Step 3: Write minimal implementation**

Add to the runtime imports of `src/cog_ew/experiments/anchors.py`:

```python
from cog_ew.marl_formation.compare import AgentPolicy as MarlAgentPolicy
from cog_ew.marl_formation.compare import compare_policies
from cog_ew.marl_formation.env import IADSFormationEnv
from cog_ew.marl_formation.train import TrainConfig as MarlTrainConfig
from cog_ew.marl_formation.train import train as train_marl
```

Add the runner:

```python
def run_marl_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "marl"
    size = _overrides(
        total_episodes=profile.marl_total_episodes,
        eval_episodes=profile.marl_eval_episodes,
    )
    qmix_dir = run_dir / "qmix"
    iql_dir = run_dir / "iql"
    qmix_cfg = replace(
        MarlTrainConfig.from_yaml(profile.marl_qmix_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(qmix_dir),
        **size,
    )
    iql_cfg = replace(
        MarlTrainConfig.from_yaml(profile.marl_iql_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(iql_dir),
        **size,
    )
    train_marl(qmix_cfg)
    train_marl(iql_cfg)

    env = IADSFormationEnv(qmix_cfg.env)
    coordinated = MarlAgentPolicy.from_checkpoint(
        qmix_dir / "best.pt",
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=qmix_cfg.agent.hidden,
        n_agents=env.n_agents,
        device=profile.device,
    )
    independent = MarlAgentPolicy.from_checkpoint(
        iql_dir / "best.pt",
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=iql_cfg.agent.hidden,
        n_agents=env.n_agents,
        device=profile.device,
    )
    result = compare_policies(
        env,
        coordinated=coordinated,
        independent=independent,
        episodes=profile.marl_compare_episodes,
        seed=profile.seed,
    )
    achieved = float(result["relative_improvement"]["suppressed_fraction"])
    baseline = float(result["independent"]["suppressed_fraction"])
    return AnchorResult(
        name="marl",
        target=_TARGETS["marl"],
        achieved=achieved,
        baseline=baseline,
        passed=_passed(achieved, _TARGETS["marl"]),
        run_dir=str(run_dir),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_marl_anchor_quick -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py
git commit -m "$(cat <<'EOF'
feat(experiments): run_marl_anchor (QMIX coordinado vs IQL independiente)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 7: `run_gan_anchor`

Encadena el pipeline M4: entrena la GAN → exporta el HDF5 sintético desde ese checkpoint → ejecuta el experimento de robustez sobre ese HDF5. El runner cablea `export.checkpoint` y `robustness.synthetic_path` a los artefactos producidos en el run dir.

**Files:**
- Modify: `src/cog_ew/experiments/anchors.py`
- Test: `tests/experiments/test_anchors.py`

**Interfaces:**
- Consumes:
  - `cog_ew.gan_signals.train.WGANGPConfig.from_yaml`, `.train(config)` (escribe `<out_dir>/best.pt`); campos `total_steps`, `device`, `seed`, `out_dir`.
  - `cog_ew.gan_signals.export.ExportConfig.from_yaml`, `.export_synthetic(config)`; campos `checkpoint`, `samples_per_type`, `out_path`, `device`, `seed`.
  - `cog_ew.gan_signals.robustness.RobustnessConfig.from_yaml`, `.run_robustness_experiment(config) -> {"baseline": float, "relative_improvement": float, ...}`; campos `synthetic_path`, `epochs`, `device`, `seed`, `out_dir`.
- Produces: `run_gan_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult` (name `"gan"`, target `0.22`, achieved = `relative_improvement` (puede ser `inf`), baseline = `baseline`, run_dir `<out_dir>/gan`).

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_anchors.py`:

```python
from cog_ew.experiments.anchors import run_gan_anchor


def test_run_gan_anchor_quick(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    result = run_gan_anchor(profile, tmp_path)
    assert result.name == "gan"
    assert result.target == 0.22
    assert result.baseline is not None
    assert math.isfinite(result.achieved) or math.isinf(result.achieved)
    if math.isinf(result.achieved):
        assert result.passed is False
    assert (Path(result.run_dir) / "synthetic.h5").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_gan_anchor_quick -v`
Expected: FAIL — `ImportError: cannot import name 'run_gan_anchor'`.

- [ ] **Step 3: Write minimal implementation**

Add to the runtime imports of `src/cog_ew/experiments/anchors.py`:

```python
from cog_ew.gan_signals.export import ExportConfig
from cog_ew.gan_signals.export import export_synthetic
from cog_ew.gan_signals.robustness import RobustnessConfig
from cog_ew.gan_signals.robustness import run_robustness_experiment
from cog_ew.gan_signals.train import WGANGPConfig
from cog_ew.gan_signals.train import train as train_gan
```

Add the runner:

```python
def run_gan_anchor(profile: ExperimentProfile, out_dir: Path) -> AnchorResult:
    run_dir = Path(out_dir) / "gan"
    gan_dir = run_dir / "wgan_gp"
    synth_path = run_dir / "synthetic.h5"

    gan_cfg = replace(
        WGANGPConfig.from_yaml(profile.gan_config),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(gan_dir),
        **_overrides(total_steps=profile.gan_total_steps),
    )
    train_gan(gan_cfg)

    export_cfg = replace(
        ExportConfig.from_yaml(profile.export_config),
        checkpoint=str(gan_dir / "best.pt"),
        out_path=str(synth_path),
        device=profile.device,
        seed=profile.seed,
        **_overrides(samples_per_type=profile.export_samples_per_type),
    )
    export_synthetic(export_cfg)

    rob_cfg = replace(
        RobustnessConfig.from_yaml(profile.robustness_config),
        synthetic_path=str(synth_path),
        device=profile.device,
        seed=profile.seed,
        out_dir=str(run_dir / "robustness"),
        **_overrides(epochs=profile.robustness_epochs),
    )
    result = run_robustness_experiment(rob_cfg)

    achieved = float(result["relative_improvement"])
    baseline = float(result["baseline"])
    return AnchorResult(
        name="gan",
        target=_TARGETS["gan"],
        achieved=achieved,
        baseline=baseline,
        passed=_passed(achieved, _TARGETS["gan"]),
        run_dir=str(run_dir),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_anchors.py::test_run_gan_anchor_quick -v`
Expected: PASS (es el test más pesado: GAN + export + 2 entrenamientos de M2).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py
git commit -m "$(cat <<'EOF'
feat(experiments): run_gan_anchor (GAN -> export -> robustez +22%)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 8: `ANCHOR_RUNNERS` + `run_anchors` + `anchors_report.json`

Cierra `report.py`: el registro de runners y la agregación que escribe el reporte con metadatos de reproducibilidad.

**Files:**
- Modify: `src/cog_ew/experiments/report.py`
- Test: `tests/experiments/test_report.py`

**Interfaces:**
- Consumes: `cog_ew.experiments.anchors` (`AnchorResult`, los 4 runners); `ExperimentProfile`.
- Produces:
  - `ANCHOR_RUNNERS: dict[str, Callable[[ExperimentProfile, Path], AnchorResult]]` con claves `jamming`, `elint`, `marl`, `gan`.
  - `run_anchors(names: tuple[str, ...], profile: ExperimentProfile, out_dir: str | Path) -> dict[str, Any]` — ejecuta los runners seleccionados, escribe `<out_dir>/anchors_report.json` y devuelve el dict del reporte con claves: `profile_name`, `seed`, `config_hash`, `dependencies`, `anchors` (por ancla: `target`, `achieved`, `baseline`, `passed`, `run_dir`).

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_report.py`:

```python
import json
from pathlib import Path

from cog_ew.experiments.report import ANCHOR_RUNNERS, run_anchors

QUICK = "configs/experiments/quick.yaml"


def test_anchor_runners_cover_all_four():
    assert set(ANCHOR_RUNNERS) == {"jamming", "elint", "marl", "gan"}


def test_run_anchors_single_writes_report(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    report = run_anchors(("elint",), profile, tmp_path)
    assert report["profile_name"] == "quick"
    assert report["seed"] == 0
    assert "config_hash" in report
    assert "torch" in report["dependencies"]
    elint = report["anchors"]["elint"]
    assert elint["target"] == 0.96
    assert elint["passed"] == (elint["achieved"] >= elint["target"])
    on_disk = json.loads((tmp_path / "anchors_report.json").read_text())
    assert on_disk == report


def test_run_anchors_elint_reproducible_by_seed(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    a = run_anchors(("elint",), profile, tmp_path / "a")
    b = run_anchors(("elint",), profile, tmp_path / "b")
    assert a["anchors"]["elint"]["achieved"] == b["anchors"]["elint"]["achieved"]


def test_run_anchors_all_aggregates_four(tmp_path):
    profile = ExperimentProfile.from_yaml(QUICK)
    report = run_anchors(("jamming", "elint", "marl", "gan"), profile, tmp_path)
    assert set(report["anchors"]) == {"jamming", "elint", "marl", "gan"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'ANCHOR_RUNNERS'`.

- [ ] **Step 3: Write minimal implementation**

In `src/cog_ew/experiments/report.py`, extend the imports and add the registry + `run_anchors`:

```python
import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from cog_ew.experiments.anchors import (
    AnchorResult,
    run_elint_anchor,
    run_gan_anchor,
    run_jamming_anchor,
    run_marl_anchor,
)

ANCHOR_RUNNERS: dict[str, Callable[[ExperimentProfile, Path], AnchorResult]] = {
    "jamming": run_jamming_anchor,
    "elint": run_elint_anchor,
    "marl": run_marl_anchor,
    "gan": run_gan_anchor,
}


def _config_hash(profile: ExperimentProfile) -> str:
    blob = json.dumps(asdict(profile), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def run_anchors(
    names: tuple[str, ...], profile: ExperimentProfile, out_dir: str | Path
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors: dict[str, Any] = {}
    for name in names:
        result = ANCHOR_RUNNERS[name](profile, out_dir)
        anchors[name] = {
            "target": result.target,
            "achieved": result.achieved,
            "baseline": result.baseline,
            "passed": result.passed,
            "run_dir": result.run_dir,
        }
    report: dict[str, Any] = {
        "profile_name": profile.name,
        "seed": profile.seed,
        "config_hash": _config_hash(profile),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "anchors": anchors,
    }
    (out_dir / "anchors_report.json").write_text(json.dumps(report, indent=2))
    return report
```

Note: keep the `from __future__ import annotations` at the top of `report.py` so the forward reference to `ExperimentProfile` in `ANCHOR_RUNNERS`/`_config_hash` resolves as a string (the dataclass is defined above the registry, but the annotations stay lazy regardless).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_report.py -v`
Expected: PASS (4 nuevos + 2 de Task 3 = 6 passed). El test `all_aggregates_four` corre el pipeline completo y es lento.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/cog_ew tests/experiments && .venv/bin/ruff format src/cog_ew tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add src/cog_ew/experiments/report.py tests/experiments/test_report.py
git commit -m "$(cat <<'EOF'
feat(experiments): run_anchors + anchors_report.json reproducible

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 9: `notebooks/run_anchors.py` (CLI) y eliminación de `colab_train_models.py`

Entrada fina para Colab: `--profile {quick,full}`, `--anchors {all|lista}`, `--out-dir`. Sustituye al runner parcial antiguo, que se elimina para no dejar dos caminos divergentes.

**Files:**
- Create: `notebooks/run_anchors.py`
- Delete: `notebooks/colab_train_models.py`
- Test: `tests/experiments/test_run_anchors_cli.py`

**Interfaces:**
- Consumes: `cog_ew.experiments.report.run_anchors`, `ExperimentProfile`, `ANCHOR_RUNNERS`.
- Produces:
  - `parse_args(argv: list[str] | None = None) -> argparse.Namespace` con `profile: str`, `anchors: str`, `out_dir: str`.
  - `resolve_anchors(spec: str) -> tuple[str, ...]` (`"all"` → las 4 claves en orden `jamming, elint, marl, gan`; lista separada por comas en otro caso).
  - `main(argv: list[str] | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/test_run_anchors_cli.py`:

```python
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_anchors_cli", Path("notebooks/run_anchors.py")
)
assert _SPEC is not None and _SPEC.loader is not None
run_anchors_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_anchors_cli)


def test_parse_args_defaults():
    args = run_anchors_cli.parse_args(["--profile", "quick"])
    assert args.profile == "quick"
    assert args.anchors == "all"


def test_resolve_anchors_all():
    assert run_anchors_cli.resolve_anchors("all") == ("jamming", "elint", "marl", "gan")


def test_resolve_anchors_list():
    assert run_anchors_cli.resolve_anchors("elint,marl") == ("elint", "marl")


def test_main_dispatches_to_run_anchors(monkeypatch, tmp_path):
    captured = {}

    def fake_run_anchors(names, profile, out_dir):
        captured["names"] = names
        captured["profile_name"] = profile.name
        captured["out_dir"] = out_dir
        return {"anchors": {}}

    monkeypatch.setattr(run_anchors_cli, "run_anchors", fake_run_anchors)
    run_anchors_cli.main(
        ["--profile", "quick", "--anchors", "elint", "--out-dir", str(tmp_path)]
    )
    assert captured["names"] == ("elint",)
    assert captured["profile_name"] == "quick"
    assert captured["out_dir"] == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_run_anchors_cli.py -v`
Expected: FAIL — `FileNotFoundError`/`spec` no carga (el archivo no existe aún).

- [ ] **Step 3: Write minimal implementation**

Create `notebooks/run_anchors.py`:

```python
"""Entrada CLI para correr las anclas Q1 (Fase 6) en Colab o local.

Uso:
    python notebooks/run_anchors.py --profile quick --anchors all --out-dir runs/anchors
    python notebooks/run_anchors.py --profile full --anchors elint,marl
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from cog_ew.experiments.report import ANCHOR_RUNNERS, ExperimentProfile, run_anchors

_PROFILES = {
    "quick": "configs/experiments/quick.yaml",
    "full": "configs/experiments/full.yaml",
}
_ORDER = ("jamming", "elint", "marl", "gan")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre las anclas Q1 del proyecto Cognitive EW.")
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--anchors", default="all", help="'all' o lista separada por comas")
    parser.add_argument("--out-dir", default="runs/anchors")
    return parser.parse_args(argv)


def resolve_anchors(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return _ORDER
    names = tuple(name.strip() for name in spec.split(",") if name.strip())
    unknown = [name for name in names if name not in ANCHOR_RUNNERS]
    if unknown:
        raise ValueError(f"anclas desconocidas: {unknown}")
    return names


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    profile = ExperimentProfile.from_yaml(_PROFILES[args.profile])
    names = resolve_anchors(args.anchors)
    report = run_anchors(names, profile, args.out_dir)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
```

Delete the old runner:

```bash
git rm notebooks/colab_train_models.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/experiments/test_run_anchors_cli.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check notebooks/run_anchors.py tests/experiments && .venv/bin/ruff format notebooks/run_anchors.py tests/experiments && .venv/bin/mypy src/cog_ew`

```bash
git add notebooks/run_anchors.py tests/experiments/test_run_anchors_cli.py
git commit -m "$(cat <<'EOF'
feat(experiments): notebook run_anchors.py; elimina colab_train_models.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KYpCP9CbP5cJY8PuZi6ja2
EOF
)"
```

---

### Task 10: Verificación final de la suite

- [ ] **Step 1: Run the full experiments suite**

Run: `.venv/bin/pytest tests/experiments/ -v`
Expected: todos PASS. (Los tests `quick` que entrenan de verdad son lentos pero deterministas.)

- [ ] **Step 2: Run the whole test suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: toda la suite del repo PASS.

- [ ] **Step 3: Final lint + type-check**

Run: `.venv/bin/ruff check . && .venv/bin/mypy src/cog_ew`
Expected: sin errores.

No commit aquí salvo que la verificación obligue a un fix (en cuyo caso, commit del fix con el trailer requerido).

---

## Self-Review

**1. Spec coverage:**
- Paquete `experiments/` con `anchors.py` (`AnchorResult` + 4 runners) y `report.py` (`ExperimentProfile.from_yaml`, `ANCHOR_RUNNERS`, `run_anchors`) → Tasks 2–8. ✓
- `passed = math.isfinite(achieved) and achieved >= target` → Task 2 `_passed`, verificado con `inf`/`nan`. ✓
- Perfiles `configs/experiments/{quick,full}.yaml` (quick CPU/diminuto, full cuda) → Task 3. ✓
- `anchors_report.json` con `profile_name`, `seed`, `config_hash`, dependencias y anclas → Task 8. ✓
- GPU-readiness (`torch.cuda.manual_seed_all`) → Task 1 (cubre los 5 del spec + export por consistencia). ✓
- `notebooks/run_anchors.py` sustituye/elimina `colab_train_models.py` → Task 9. ✓
- Baseline `marl` = QMIX vs IQL → Task 6. ✓
- 7 tests del spec → test_elint (4·1), test_jamming (4·1+5), test_marl (4·3), test_gan (4·4), run_anchors single+metadatos (8·5), aggregate-4 (8·6), reproducible elint (8·7). ✓
- `weights_only=True` en recarga de checkpoints → Task 5 (jamming) usa `weights_only=True`; M3/M4 ya lo cumplen. ✓

**2. Placeholder scan:** Sin TBD/TODO; todo el código está completo en cada step. ✓

**3. Type consistency:** `AnchorResult` (6 campos) idéntico en Tasks 2/4/5/6/7/8. `_overrides`/`_passed`/`_TARGETS` definidos en Task 2/4 y reusados con la misma firma. `ExperimentProfile` campos consumidos por los runners coinciden con los definidos en Task 3. `run_anchors(names, profile, out_dir)` firma idéntica en Task 8 y Task 9. Claves de retorno (`relative_improvement.suppressed_fraction`, `independent.suppressed_fraction`, `cognitive.win_rate`, `test.lpi_accuracy`, `relative_improvement`, `baseline`) verificadas contra el código real. ✓

**Nota de desviación menor:** Task 1 añade `cuda.manual_seed_all` también a `gan_signals/export.py` (que tiene su propio `_set_seeds`), no solo a los 5 módulos enumerados en el spec — estrictamente más consistente, sin coste.
