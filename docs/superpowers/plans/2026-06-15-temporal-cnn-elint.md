# Temporal CNN multi-tarea ELINT (Modelo 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar el slice entrenable del Modelo 2: una Temporal CNN dilatada multi-tarea (tipo de emisor + modo; amenaza derivada del modo) con métricas (macro-accuracy, accuracy LPI, matriz de confusión), perfilado de latencia (media/p99) y bucle de entrenamiento reproducible que consume `PDWSyntheticDataset`.

**Architecture:** Backbone TCN dilatado (Conv1d "same" + bloques residuales con dilations cíclicas) compartido por dos cabezas lineales (tipo, modo). La amenaza se obtiene en inferencia con un buffer constante `mode_to_threat`. Funciones puras de métricas y latencia, testables sin entrenamiento. Un `train.py` con config YAML anidada (datos + modelo + entrenamiento), seeds explícitos, checkpoint del mejor modelo por macro-accuracy de validación y volcado de `metrics.json`.

**Tech Stack:** Python 3.11, PyTorch 2.6, NumPy 2.4, PyYAML, pytest. `trackio` opcional/guardado (no es dependencia de tests). ruff line-length 100, mypy --strict. Usar `.venv/bin/<tool>`.

---

## File Structure

- `src/cog_ew/data/pdw_library.py` — añadir `EmitterLibrary.lpi_indices()`.
- `src/cog_ew/temporal_cnn_elint/model.py` — `TemporalCNNConfig`, `_TCNBlock`, `TemporalCNN`.
- `src/cog_ew/temporal_cnn_elint/metrics.py` — `macro_accuracy`, `confusion_matrix`, `lpi_accuracy`, `profile_latency`.
- `src/cog_ew/temporal_cnn_elint/train.py` — `TrainConfig`, `train`, helpers internos.
- `configs/temporal_cnn_elint/train.yaml` — hiperparámetros (datos + modelo + entrenamiento).
- `tests/temporal_cnn_elint/` — `test_model.py`, `test_metrics.py`, `test_train.py`.
- `tests/data/test_pdw_library.py` — añadir test de `lpi_indices`.

Convenciones (CLAUDE.md): type hints en API pública; sin comentarios de *qué*; `ruff format` + `ruff check` + `mypy --strict` limpios sobre los ficheros tocados (incl. tests) en cada commit.

---

## Task 1: `EmitterLibrary.lpi_indices()` + `split_dataset` genérico

**Files:**
- Modify: `src/cog_ew/data/pdw_library.py`
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_pdw_library.py`

### Parte A — `split_dataset` genérico (typing, sin cambio de comportamiento)

`split_dataset` está anotado concretamente para un dataset de 3-tupla; `train.py` (Modelo 2) le pasará
`PDWSyntheticDataset`, que emite 4-tuplas → mypy --strict daría error de tipos. Se generaliza con un
`TypeVar` (retrocompatible; el comportamiento ya está cubierto por los tests existentes de `test_loaders.py`
y `test_pdw_dataset.py`).

- [ ] **Step A1: Editar `src/cog_ew/data/loaders.py`**

Añadir el import del `TypeVar` en la zona de imports de la stdlib (junto a `from collections.abc import
Sequence`):

```python
from typing import TypeVar
```

Y, justo antes de la definición de `split_dataset`, declarar el typevar y reescribir la firma:

```python
_SampleT = TypeVar("_SampleT")


def split_dataset(
    dataset: Dataset[_SampleT],
    fractions: Sequence[float],
    seed: int,
) -> list[Subset[_SampleT]]:
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(fractions), generator=generator)
```

(El cuerpo no cambia; sólo la firma pasa de `tuple[torch.Tensor, int, int]` a `_SampleT`.)

- [ ] **Step A2: Verificar tests + tipos de la capa de datos**

Run: `.venv/bin/pytest tests/data/test_loaders.py tests/data/test_pdw_dataset.py -q`
Expected: PASS (sin cambios de comportamiento).
Run: `.venv/bin/mypy src/cog_ew/data/loaders.py`
Expected: `Success: no issues found`.

### Parte B — `EmitterLibrary.lpi_indices()`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/data/test_pdw_library.py`:

```python
def test_lpi_indices_returns_lpi_emitter_positions():
    lib = EmitterLibrary.from_yaml(CONFIG)
    # LPI-FMCW y LPI-polyphase son los dos últimos de la taxonomía
    assert lib.lpi_indices() == (6, 7)
```

- [ ] **Step 2: Ejecutar para verificar que falla**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py::test_lpi_indices_returns_lpi_emitter_positions -v`
Expected: FAIL con `AttributeError: 'EmitterLibrary' object has no attribute 'lpi_indices'`.

- [ ] **Step 3: Implementación**

Añadir el método dentro de la clase `EmitterLibrary` en `src/cog_ew/data/pdw_library.py`, justo después de `emitter_names`:

```python
    def lpi_indices(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, emitter in enumerate(self.emitters)
            if any(mode_spec.lpi for mode_spec in emitter.modes.values())
        )
```

- [ ] **Step 4: Ejecutar para verificar que pasa**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py -v`
Expected: PASS (todos los de la librería).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_library.py src/cog_ew/data/loaders.py tests/data/test_pdw_library.py
.venv/bin/ruff check src/cog_ew/data/pdw_library.py src/cog_ew/data/loaders.py tests/data/test_pdw_library.py
.venv/bin/mypy src/cog_ew/data/pdw_library.py src/cog_ew/data/loaders.py
git add src/cog_ew/data/pdw_library.py src/cog_ew/data/loaders.py tests/data/test_pdw_library.py
git commit -m "feat(data): lpi_indices + split_dataset genérico (TypeVar)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `TemporalCNNConfig` + `TemporalCNN.forward`

**Files:**
- Create: `src/cog_ew/temporal_cnn_elint/model.py`
- Test: `tests/temporal_cnn_elint/test_model.py`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/temporal_cnn_elint/test_model.py`:

```python
import torch

from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def test_forward_output_shapes():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(4, 10, 64)

    type_logits, mode_logits = model(x)

    assert type_logits.shape == (4, 8)
    assert mode_logits.shape == (4, 4)


def test_forward_deterministic_in_eval():
    torch.manual_seed(0)
    model = TemporalCNN(TemporalCNNConfig())
    model.eval()
    x = torch.randn(2, 10, 64)

    a = model(x)[0]
    b = model(x)[0]

    assert torch.allclose(a, b)


def test_param_count_is_lightweight():
    model = TemporalCNN(TemporalCNNConfig())
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 300_000
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_model.py -v`
Expected: FAIL con `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Implementación**

Crear `src/cog_ew/temporal_cnn_elint/model.py`:

```python
"""Temporal CNN para clasificación en tiempo real de señales de amenaza ELINT (<1ms latencia)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn

from cog_ew.data.pdw_library import MODES, mode_to_threat


@dataclass
class TemporalCNNConfig:
    in_channels: int = 10
    seq_len: int = 64
    hidden: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    n_types: int = 8
    n_modes: int = 4
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> TemporalCNNConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if "dilations" in raw:
            raw["dilations"] = tuple(raw["dilations"])
        return cls(**raw)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation  # kernel=3 → "same" con pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class TemporalCNN(nn.Module):
    threat_from_mode: torch.Tensor

    def __init__(self, config: TemporalCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.hidden, 3, padding=1),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _TCNBlock(config.hidden, d, config.dropout) for d in config.dilations
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_type = nn.Linear(config.hidden, config.n_types)
        self.head_mode = nn.Linear(config.hidden, config.n_modes)
        threat = torch.tensor([mode_to_threat(m) for m in MODES], dtype=torch.int64)
        self.register_buffer("threat_from_mode", threat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        feat = self.pool(h).squeeze(-1)
        return self.head_type(feat), self.head_mode(feat)
```

> Nota mypy: la anotación de clase `threat_from_mode: torch.Tensor` informa a mypy del tipo del buffer
> registrado con `register_buffer`. El import `field` se usará en Task 3 (default mutable del buffer no
> aplica aquí); si `ruff` marca `field` como no usado en este commit, elimínalo y reañádelo en Task 3.

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/model.py
git add src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
git commit -m "feat(elint): TemporalCNN dilatada (backbone + cabezas tipo/modo)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Si en Step 5 `ruff check` marca `from dataclasses import ... field` como no usado, edita el import a
> `from dataclasses import dataclass` y vuelve a correr lint antes del commit.

---

## Task 3: `TemporalCNN.predict` (amenaza derivada del modo)

**Files:**
- Modify: `src/cog_ew/temporal_cnn_elint/model.py`
- Test: `tests/temporal_cnn_elint/test_model.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/temporal_cnn_elint/test_model.py`:

```python
from cog_ew.data.pdw_library import mode_to_threat


def test_predict_shapes_and_dtypes():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(5, 10, 64)

    type_pred, mode_pred, threat_pred = model.predict(x)

    assert type_pred.shape == (5,)
    assert mode_pred.shape == (5,)
    assert threat_pred.shape == (5,)


def test_predict_threat_is_consistent_with_mode():
    model = TemporalCNN(TemporalCNNConfig())
    x = torch.randn(16, 10, 64)

    _, mode_pred, threat_pred = model.predict(x)

    from cog_ew.data.pdw_library import MODES

    expected = torch.tensor([mode_to_threat(MODES[m]) for m in mode_pred.tolist()])
    assert torch.equal(threat_pred.cpu(), expected)
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_model.py -k predict -v`
Expected: FAIL con `AttributeError: 'TemporalCNN' object has no attribute 'predict'`.

- [ ] **Step 3: Implementación**

Añadir el método `predict` a la clase `TemporalCNN` en `model.py`, justo después de `forward`:

```python
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_logits, mode_logits = self.forward(x)
        type_pred = type_logits.argmax(dim=-1)
        mode_pred = mode_logits.argmax(dim=-1)
        threat_pred = self.threat_from_mode[mode_pred]
        return type_pred, mode_pred, threat_pred
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_model.py -v`
Expected: PASS (5 tests en total).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/model.py
git add src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
git commit -m "feat(elint): TemporalCNN.predict con amenaza derivada del modo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Métricas — `macro_accuracy` + `confusion_matrix`

**Files:**
- Create: `src/cog_ew/temporal_cnn_elint/metrics.py`
- Test: `tests/temporal_cnn_elint/test_metrics.py`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/temporal_cnn_elint/test_metrics.py`:

```python
import torch

from cog_ew.temporal_cnn_elint.metrics import confusion_matrix, macro_accuracy


def test_macro_accuracy_perfect():
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3])

    assert macro_accuracy(preds, targets, 4) == 1.0


def test_macro_accuracy_balances_classes():
    # clase 0 perfecta (2/2), clase 1 fallada (0/2) → macro = 0.5
    preds = torch.tensor([0, 0, 0, 0])
    targets = torch.tensor([0, 0, 1, 1])

    assert macro_accuracy(preds, targets, 2) == 0.5


def test_macro_accuracy_ignores_unsupported_classes():
    # sólo aparece la clase 0 (perfecta); las clases 1 y 2 no tienen soporte
    preds = torch.tensor([0, 0])
    targets = torch.tensor([0, 0])

    assert macro_accuracy(preds, targets, 3) == 1.0


def test_confusion_matrix_counts():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])

    cm = confusion_matrix(preds, targets, 3)

    assert cm.shape == (3, 3)
    assert cm[2, 1].item() == 1  # un real-2 predicho como 1
    assert cm[2, 2].item() == 1
    assert cm[0, 0].item() == 1
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementación**

Crear `src/cog_ew/temporal_cnn_elint/metrics.py`:

```python
"""Métricas y perfilado de latencia para el clasificador ELINT."""

from __future__ import annotations

import torch


def macro_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    recalls: list[float] = []
    for c in range(num_classes):
        mask = targets == c
        support = int(mask.sum().item())
        if support == 0:
            continue
        correct = int((preds[mask] == c).sum().item())
        recalls.append(correct / support)
    return sum(recalls) / len(recalls) if recalls else 0.0


def confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py -v`
Expected: PASS (4 tests). En este punto el fichero de test sólo contiene los de
`macro_accuracy`/`confusion_matrix`; los de `lpi`/`latency` se añaden en Task 5.

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/metrics.py
git add src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
git commit -m "feat(elint): métricas macro_accuracy y confusion_matrix

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Nota: `time`, `numpy` y `nn` no se importan aún (se añaden en Task 5, donde se usan). Mantener el import
> mínimo (`import torch`) deja este commit limpio de lint.

---

## Task 5: Métricas — `lpi_accuracy` + `profile_latency`

**Files:**
- Modify: `src/cog_ew/temporal_cnn_elint/metrics.py`
- Test: `tests/temporal_cnn_elint/test_metrics.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/temporal_cnn_elint/test_metrics.py`:

```python
from cog_ew.temporal_cnn_elint.metrics import lpi_accuracy, profile_latency
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig


def test_lpi_accuracy_filters_to_lpi_classes():
    # tipos reales: [6, 7, 0, 1]; LPI = {6, 7}
    type_preds = torch.tensor([6, 0, 0, 1])
    type_targets = torch.tensor([6, 7, 0, 1])

    # sólo se evalúan los reales 6 y 7: 6→6 acierta, 7→0 falla → 0.5
    assert lpi_accuracy(type_preds, type_targets, (6, 7)) == 0.5


def test_lpi_accuracy_no_lpi_samples_returns_zero():
    type_preds = torch.tensor([0, 1])
    type_targets = torch.tensor([0, 1])

    assert lpi_accuracy(type_preds, type_targets, (6, 7)) == 0.0


def test_profile_latency_returns_positive_mean_and_p99():
    model = TemporalCNN(TemporalCNNConfig())
    sample = torch.randn(1, 10, 64)

    mean_ms, p99_ms = profile_latency(
        model, sample, n_warmup=2, n_iter=10, device="cpu"
    )

    assert mean_ms > 0.0
    assert p99_ms > 0.0
    assert p99_ms >= mean_ms
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py -k "lpi or latency" -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementación**

Primero, ampliar los imports al inicio de `metrics.py` (reemplazar la línea `import torch` por este bloque):

```python
import time

import numpy as np
import torch
from torch import nn
```

Luego añadir al final de `src/cog_ew/temporal_cnn_elint/metrics.py`:

```python
def lpi_accuracy(
    type_preds: torch.Tensor,
    type_targets: torch.Tensor,
    lpi_indices: tuple[int, ...],
) -> float:
    lpi = torch.tensor(list(lpi_indices), dtype=type_targets.dtype)
    mask = torch.isin(type_targets, lpi)
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((type_preds[mask] == type_targets[mask]).float().mean().item())


def profile_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    n_warmup: int = 10,
    n_iter: int = 100,
    device: str = "cpu",
) -> tuple[float, float]:
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    sample = sample.to(dev)
    is_cuda = dev.type == "cuda"
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)
        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(n_iter):
            start = time.perf_counter()
            model(sample)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(np.percentile(arr, 99))
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py -v`
Expected: PASS (7 tests en total).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/metrics.py
git add src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
git commit -m "feat(elint): lpi_accuracy y profile_latency (media/p99, batch=1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `TrainConfig` + `from_yaml` + `train.yaml`

**Files:**
- Create: `src/cog_ew/temporal_cnn_elint/train.py`
- Create: `configs/temporal_cnn_elint/train.yaml`
- Test: `tests/temporal_cnn_elint/test_train.py`

- [ ] **Step 1: Crear el fichero de configuración**

Crear `configs/temporal_cnn_elint/train.yaml`:

```yaml
data:
  library_path: configs/temporal_cnn_elint/emitters.yaml
  emitters: null
  modes: null
  window: 64
  n_pulses: 256
  n_trains: 16
  normalize: true
  noise_std: 0.02
  drop_prob: 0.02
  spurious_prob: 0.01
  seed: 0
model:
  in_channels: 10
  seq_len: 64
  hidden: 64
  dilations: [1, 2, 4, 8]
  n_types: 8
  n_modes: 4
  dropout: 0.1
splits: [0.7, 0.15, 0.15]
batch_size: 64
epochs: 30
lr: 0.001
weight_decay: 0.0001
loss_weights: [1.0, 1.0]
device: cpu
seed: 0
out_dir: runs/temporal_cnn_elint
tracking: false
```

- [ ] **Step 2: Escribir el test que falla**

Crear `tests/temporal_cnn_elint/test_train.py`:

```python
from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig
from cog_ew.temporal_cnn_elint.train import TrainConfig

CONFIG = "configs/temporal_cnn_elint/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)

    assert isinstance(config.data, PDWConfig)
    assert isinstance(config.model, TemporalCNNConfig)
    assert config.data.window == 64
    assert config.model.dilations == (1, 2, 4, 8)
    assert config.splits == (0.7, 0.15, 0.15)
    assert config.loss_weights == (1.0, 1.0)
    assert config.tracking is False
```

- [ ] **Step 3: Ejecutar para verificar que falla**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_train.py -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 4: Implementación**

Crear `src/cog_ew/temporal_cnn_elint/train.py`:

```python
"""Script de entrenamiento de la Temporal CNN para clasificación ELINT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig


@dataclass
class TrainConfig:
    data: PDWConfig
    model: TemporalCNNConfig
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_weights: tuple[float, float] = (1.0, 1.0)
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/temporal_cnn_elint"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        data_raw = raw.pop("data")
        for key in ("emitters", "modes"):
            if data_raw.get(key) is not None:
                data_raw[key] = tuple(data_raw[key])
        model_raw = raw.pop("model")
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        for key in ("splits", "loss_weights"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(
            data=PDWConfig(**data_raw), model=TemporalCNNConfig(**model_raw), **raw
        )
```

- [ ] **Step 5: Ejecutar para verificar que pasa**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_train.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/train.py
git add src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py configs/temporal_cnn_elint/train.yaml
git commit -m "feat(elint): TrainConfig anidada (datos+modelo) desde YAML

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Bucle `train` + evaluación + checkpoint + `metrics.json`

**Files:**
- Modify: `src/cog_ew/temporal_cnn_elint/train.py`
- Test: `tests/temporal_cnn_elint/test_train.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/temporal_cnn_elint/test_train.py`:

```python
from cog_ew.temporal_cnn_elint.train import train


def _tiny_config(out_dir):
    data = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        emitters=("SA-2", "LPI-FMCW"),
        modes=("search", "track"),
        window=64,
        n_pulses=192,
        n_trains=6,
        seed=0,
    )
    model = TemporalCNNConfig()
    return TrainConfig(
        data=data,
        model=model,
        splits=(0.6, 0.2, 0.2),
        batch_size=16,
        epochs=3,
        lr=1e-3,
        device="cpu",
        seed=0,
        out_dir=str(out_dir),
        tracking=False,
    )


def test_train_smoke_reduces_loss_and_writes_metrics(tmp_path):
    result = train(_tiny_config(tmp_path))

    history = result["train_loss_history"]
    assert history[-1] < history[0]
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    test_metrics = result["test"]
    assert "macro_acc_type" in test_metrics
    assert "macro_acc_mode" in test_metrics
    assert "lpi_accuracy" in test_metrics
    assert test_metrics["latency_mean_ms"] > 0.0


def test_train_is_deterministic(tmp_path):
    a = train(_tiny_config(tmp_path / "a"))
    b = train(_tiny_config(tmp_path / "b"))
    assert a["train_loss_history"] == b["train_loss_history"]
```

- [ ] **Step 2: Ejecutar para verificar que falla**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_train.py -k "smoke or deterministic" -v`
Expected: FAIL con `ImportError: cannot import name 'train'`.

- [ ] **Step 3: Implementación**

Actualizar imports al inicio de `src/cog_ew/temporal_cnn_elint/train.py` (reemplazar el bloque de imports
existente por este):

```python
"""Script de entrenamiento de la Temporal CNN para clasificación ELINT."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.temporal_cnn_elint.metrics import (
    lpi_accuracy,
    macro_accuracy,
    profile_latency,
)
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig
```

Añadir al final de `train.py` (después de la clase `TrainConfig`):

```python
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_preds(
    model: TemporalCNN, loader: DataLoader[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_preds, type_targets, mode_preds, mode_targets = [], [], [], []
    model.eval()
    for x, y_type, y_mode, _ in loader:
        tp, mp, _ = model.predict(x.to(device))
        type_preds.append(tp.cpu())
        mode_preds.append(mp.cpu())
        type_targets.append(y_type)
        mode_targets.append(y_mode)
    return (
        torch.cat(type_preds),
        torch.cat(type_targets),
        torch.cat(mode_preds),
        torch.cat(mode_targets),
    )


def _init_tracking(config: TrainConfig) -> Any:
    import trackio

    return trackio.init(project="temporal_cnn_elint", config=vars(config))


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    device = torch.device(config.device)

    dataset = PDWSyntheticDataset(config.data)
    train_ds, val_ds, test_ds = split_dataset(dataset, config.splits, config.seed)
    train_loader: DataLoader[Any] = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )
    val_loader: DataLoader[Any] = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader: DataLoader[Any] = DataLoader(test_ds, batch_size=config.batch_size)

    model = TemporalCNN(config.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    w_type, w_mode = config.loss_weights

    run = _init_tracking(config) if config.tracking else None
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_val = -1.0
    history: list[float] = []

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for x, y_type, y_mode, _ in train_loader:
            x = x.to(device)
            y_type = y_type.to(device)
            y_mode = y_mode.to(device)
            type_logits, mode_logits = model(x)
            loss = w_type * F.cross_entropy(type_logits, y_type) + w_mode * F.cross_entropy(
                mode_logits, y_mode
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        history.append(train_loss)

        tp, tt, mp, mt = _collect_preds(model, val_loader, device)
        val_type = macro_accuracy(tp, tt, config.model.n_types)
        val_mode = macro_accuracy(mp, mt, config.model.n_modes)
        val_score = 0.5 * (val_type + val_mode)
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_macro_acc_type": val_type,
                    "val_macro_acc_mode": val_mode,
                }
            )
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, weights_only=True))
    library = EmitterLibrary.from_yaml(config.data.library_path)
    tp, tt, mp, mt = _collect_preds(model, test_loader, device)
    test_metrics = {
        "macro_acc_type": macro_accuracy(tp, tt, config.model.n_types),
        "macro_acc_mode": macro_accuracy(mp, mt, config.model.n_modes),
        "lpi_accuracy": lpi_accuracy(tp, tt, library.lpi_indices()),
    }
    sample = next(iter(test_loader))[0][:1].to(device)
    mean_ms, p99_ms = profile_latency(
        model, sample, n_warmup=5, n_iter=50, device=config.device
    )
    test_metrics["latency_mean_ms"] = mean_ms
    test_metrics["latency_p99_ms"] = p99_ms

    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if run is not None:
        run.finish()
    return {"test": test_metrics, "train_loss_history": history}
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/temporal_cnn_elint/test_train.py -v`
Expected: PASS (3 tests). Si `test_train_smoke...` falla en `history[-1] < history[0]`, NO relajes el
assert: revisa que las semillas se fijan antes de construir el dataset y el modelo, y que `lr=1e-3`.

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/train.py
git add src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py
git commit -m "feat(elint): bucle de entrenamiento multi-tarea + eval + latencia

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Nota seguridad/mypy: usar **siempre** `torch.load(best_path, weights_only=True)` (el checkpoint es un
> `state_dict` de tensores; `weights_only=True` evita deserializar objetos arbitrarios). `load_state_dict`
> acepta el `Any` que devuelve `torch.load`, así que no hace falta cast.

---

## Task 8: Verificación final

**Files:** ninguno nuevo (sólo correcciones si hace falta)

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/pytest -q`
Expected: PASS (datos IQ + datos PDW + Temporal CNN).

- [ ] **Step 2: Lint + formato + tipos del paquete**

```bash
.venv/bin/ruff check src/cog_ew/temporal_cnn_elint/ tests/temporal_cnn_elint/
.venv/bin/ruff format --check src/cog_ew/temporal_cnn_elint/ tests/temporal_cnn_elint/
.venv/bin/mypy src/cog_ew/temporal_cnn_elint/
```
Expected: sin errores / "already formatted" / `Success: no issues found`.

- [ ] **Step 3: Commit de correcciones (si las hubo)**

```bash
git add -A
git commit -m "style(elint): ruff format y correcciones de lint/tipos

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
> Si no hubo cambios, omitir.

---

## Self-Review (cobertura del spec)

- **TCN dilatada, backbone único + 2 cabezas (tipo, modo)** → Task 2. ✅
- **Amenaza derivada del modo** (`predict` + buffer `threat_from_mode`) → Task 3. ✅
- **Padding "same" no causal** → Task 2 (`pad = dilation`, stem `padding=1`). ✅
- **macro_accuracy + confusion_matrix** → Task 4. ✅
- **lpi_accuracy + profile_latency (media/p99, batch=1)** → Task 5; `lpi_indices` → Task 1B. ✅
- **`split_dataset` reutilizable con el dataset PDW (4-tupla)** → Task 1A (genérico con `TypeVar`). ✅
- **TrainConfig anidada + YAML versionado** → Task 6. ✅
- **Bucle: Adam, loss multi-tarea, seeds, checkpoint mejor val, eval test, metrics.json** → Task 7. ✅
- **trackio opcional/guardado por flag** → Task 7 (`_init_tracking` sólo si `config.tracking`). ✅
- **Reproducibilidad (seeds torch/numpy/random, splits deterministas)** → Tasks 6-7. ✅
- **Tests: shapes, consistencia amenaza, métricas conocidas, latencia, smoke entrenable determinista** → Tasks 2-7. ✅
- **Sin dependencias nuevas** → no hay tarea de deps; `trackio` no se importa en tests. ✅

Fuera de alcance (coherente con el spec): ejecución real en Colab, longitud variable, deinterleaving,
atención/Transformer, ONNX/TensorRT, data augmentation.
