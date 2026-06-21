# Modelo 4 · Sub-pieza C — Evaluación de robustez (+22%) · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir la mejora de robustez del Modelo 2 frente a emisores no catalogados cuando su entrenamiento se aumenta con las señales sintéticas del cWGAN-GP (la métrica ancla +22% del Modelo 4).

**Architecture:** Un loader del HDF5 sintético (`data/synthetic_loader.py`) que entrega el mismo 4-tuple que `PDWSyntheticDataset` con `mode=-1`, y un harness (`gan_signals/robustness.py`) que entrena dos clasificadores `TemporalCNN` con la misma seed (baseline = solo reales catalogados; aumentado = + sintéticos de los emisores retenidos, con la cabeza de modo enmascarada vía `ignore_index=-1`) y compara su `macro_acc_type` sobre las señales reales de los emisores retenidos.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, h5py, PyYAML, pytest.

## Global Constraints

- **Convención de ejes:** ventanas PDW channels-first `(10, 64)`; etiquetas de tipo = índices globales de emisor (consistentes entre `PDWSyntheticDataset.type` y el `source_a` del HDF5).
- **Consume:** `TemporalCNN` / `TemporalCNNConfig` (`cog_ew.temporal_cnn_elint.model`); `macro_accuracy`, `profile_latency` (`cog_ew.temporal_cnn_elint.metrics`); `PDWConfig` / `PDWSyntheticDataset` (`cog_ew.data.pdw_dataset`); `EmitterLibrary` (`cog_ew.data.pdw_library`); `split_dataset` (`cog_ew.data.loaders`). HDF5 de B: datasets `X (N,10,64) f32`, `source_a (N) i64`, `is_known (N) bool`.
- **Supervisión:** los sintéticos solo supervisan la cabeza `type` (su `mode = -1`); la pérdida de modo usa solo las filas con `mode >= 0` (un batch todo-sintético aporta `0` a la pérdida de modo, sin gradiente a `head_mode`).
- **Reproducibilidad:** `_set_seeds` (random/numpy/torch) **antes de cada** entrenamiento (baseline y aumentado parten de la misma init y RNG; la única diferencia es la data sintética); hiperparámetros solo en YAML (`configs/gan_signals/robustness.yaml`); `run_meta.json` con seed + hiperparámetros + `config_hash` + `synthetic_hash` (sha256 del HDF5) + versiones de dependencias.
- **`relative_improvement`** `= (augmented − baseline) / baseline`, con guard `baseline == 0 → float("inf")` (igual que el harness del Modelo 3).
- **Seguridad:** no se exponen parámetros de amenazas reales (solo catálogo sintético + datos derivados); cualquier `torch.load` usa `weights_only=True`.
- **Refinamiento vs spec:** `RobustnessConfig` no tiene un `library_path` de nivel superior redundante; la librería de emisores se carga de `config.pdw.library_path` (única fuente de verdad).
- **Calidad:** `ruff format` + `ruff check` + `mypy` limpios antes de cada commit; type hints en funciones públicas; comentarios solo el *por qué* no obvio.
- **Commits:** terminan con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `SyntheticPDWDataset`

**Files:**
- Create: `src/cog_ew/data/synthetic_loader.py`
- Test: `tests/data/test_synthetic_loader.py`

**Interfaces:**
- Produces:
  - `SyntheticPDWDataset(hdf5_path, *, emitters: tuple[int, ...] | None = None, known_only: bool = True)` — `torch.utils.data.Dataset`. Lee `X`/`source_a`/`is_known`; filtra por `is_known` (si `known_only`) y por `source_a ∈ emitters` (si `emitters`). `__getitem__(i) -> (x: torch.Tensor (10,64), type: int = source_a, mode: int = -1, threat: int = -1)`.

- [ ] **Step 1: Write the failing test**

```python
import h5py
import numpy as np
import torch

from cog_ew.data.synthetic_loader import SyntheticPDWDataset


def _make_synth(path, source_a, is_known):
    n = len(source_a)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(n, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.asarray(source_a, dtype=np.int64))
        fh.create_dataset("is_known", data=np.asarray(is_known, dtype=bool))
    return str(path)


def test_synthetic_dataset_returns_four_tuple(tmp_path):
    path = _make_synth(tmp_path / "s.h5", [6, 7, 6], [True, True, True])
    ds = SyntheticPDWDataset(path)
    x, type_id, mode, threat = ds[0]
    assert x.shape == (10, 64)
    assert type_id == 6
    assert mode == -1 and threat == -1
    assert len(ds) == 3


def test_synthetic_dataset_filters_emitters_and_known(tmp_path):
    path = _make_synth(
        tmp_path / "s.h5",
        source_a=[6, 7, 6, 0],
        is_known=[True, True, False, True],
    )
    ds = SyntheticPDWDataset(path, emitters=(6, 7), known_only=True)
    types = sorted(ds[i][1] for i in range(len(ds)))
    assert types == [6, 7]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_synthetic_loader.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'SyntheticPDWDataset'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Loader del HDF5 sintético de la GAN para aumentar el clasificador ELINT (Modelo 4, sub-pieza C)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticPDWDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        emitters: tuple[int, ...] | None = None,
        known_only: bool = True,
    ) -> None:
        with h5py.File(hdf5_path, "r") as fh:
            x = np.asarray(fh["X"], dtype=np.float32)
            source_a = np.asarray(fh["source_a"], dtype=np.int64)
            is_known = np.asarray(fh["is_known"], dtype=bool)
        mask = np.ones(x.shape[0], dtype=bool)
        if known_only:
            mask &= is_known
        if emitters is not None:
            mask &= np.isin(source_a, np.asarray(emitters, dtype=np.int64))
        self._x = torch.from_numpy(np.ascontiguousarray(x[mask]))
        self._type = torch.from_numpy(np.ascontiguousarray(source_a[mask]))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        return self._x[index], int(self._type[index]), -1, -1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_synthetic_loader.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/data/synthetic_loader.py tests/data/test_synthetic_loader.py && .venv/bin/ruff check src/cog_ew/data/ tests/data/ && .venv/bin/mypy src/cog_ew/data/synthetic_loader.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/data/synthetic_loader.py tests/data/test_synthetic_loader.py
git commit -m "$(cat <<'EOF'
feat(data): SyntheticPDWDataset — loader del HDF5 sintético (mode=-1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `RobustnessConfig` + YAML

**Files:**
- Create: `src/cog_ew/gan_signals/robustness.py`
- Create: `configs/gan_signals/robustness.yaml`
- Test: `tests/gan_signals/test_robustness.py`

**Interfaces:**
- Consumes: `TemporalCNNConfig` (`cog_ew.temporal_cnn_elint.model`); `PDWConfig` (`cog_ew.data.pdw_dataset`).
- Produces:
  - `RobustnessConfig` (dataclass): `synthetic_path: str`, `held_out: tuple[str, ...]`, `model: TemporalCNNConfig`, `pdw: PDWConfig`, `augment_held_out_only: bool = True`, `epochs: int = 30`, `batch_size: int = 64`, `lr: float = 1e-3`, `weight_decay: float = 1e-4`, `seed: int = 0`, `device: str = "cpu"`, `out_dir: str = "runs/gan_signals/robustness"`.
  - `RobustnessConfig.from_yaml(path) -> RobustnessConfig` (parsea `model`, `pdw` y convierte `held_out`/tuplas).

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig
from cog_ew.gan_signals.robustness import RobustnessConfig


def test_robustness_config_from_yaml():
    config = RobustnessConfig.from_yaml("configs/gan_signals/robustness.yaml")
    assert isinstance(config.model, TemporalCNNConfig)
    assert isinstance(config.pdw, PDWConfig)
    assert config.held_out == ("LPI-FMCW", "LPI-polyphase")
    assert config.augment_held_out_only is True
    assert config.synthetic_path.endswith(".h5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -q`
Expected: FAIL — `ImportError: cannot import name 'RobustnessConfig'`.

- [ ] **Step 3: Write the config file**

Create `configs/gan_signals/robustness.yaml`:

```yaml
synthetic_path: data/synthetic/wgan_gp.h5
held_out: [LPI-FMCW, LPI-polyphase]
augment_held_out_only: true
model:
  in_channels: 10
  seq_len: 64
  hidden: 64
  dilations: [1, 2, 4, 8]
  n_types: 8
  n_modes: 4
  dropout: 0.1
pdw:
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
epochs: 30
batch_size: 64
lr: 0.001
weight_decay: 0.0001
seed: 0
device: cpu
out_dir: runs/gan_signals/robustness
```

- [ ] **Step 4: Write minimal implementation**

```python
"""Experimento de robustez +22%: aumento del Modelo 2 con señales sintéticas (Modelo 4, sub-pieza C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig


@dataclass
class RobustnessConfig:
    synthetic_path: str
    held_out: tuple[str, ...]
    model: TemporalCNNConfig
    pdw: PDWConfig
    augment_held_out_only: bool = True
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/gan_signals/robustness"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobustnessConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        model_raw = raw.pop("model")
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        pdw_raw = raw.pop("pdw")
        for key in ("emitters", "modes"):
            if pdw_raw.get(key) is not None:
                pdw_raw[key] = tuple(pdw_raw[key])
        raw["held_out"] = tuple(raw["held_out"])
        return cls(
            model=TemporalCNNConfig(**model_raw),
            pdw=PDWConfig(**pdw_raw),
            **raw,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/robustness.py tests/gan_signals/test_robustness.py && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/robustness.py`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/cog_ew/gan_signals/robustness.py configs/gan_signals/robustness.yaml tests/gan_signals/test_robustness.py
git commit -m "$(cat <<'EOF'
feat(gan): RobustnessConfig + robustness.yaml

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `_classifier_loss` + `evaluate_type_accuracy`

**Files:**
- Modify: `src/cog_ew/gan_signals/robustness.py`
- Test: `tests/gan_signals/test_robustness.py`

**Interfaces:**
- Consumes: `TemporalCNN` (`cog_ew.temporal_cnn_elint.model`); `macro_accuracy` (`cog_ew.temporal_cnn_elint.metrics`).
- Produces:
  - `_classifier_loss(type_logits, mode_logits, y_type, y_mode) -> torch.Tensor` — CE de tipo + CE de modo SOLO sobre filas con `y_mode >= 0` (si no hay ninguna, el término de modo es `0`).
  - `evaluate_type_accuracy(model, dataset, n_types, device) -> float` — `macro_accuracy` de la cabeza de tipo sobre `dataset`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig
from cog_ew.gan_signals.robustness import _classifier_loss, evaluate_type_accuracy
from cog_ew.data.synthetic_loader import SyntheticPDWDataset


def _tiny_model() -> TemporalCNN:
    return TemporalCNN(TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0))


def test_synthetic_only_batch_does_not_touch_mode_head():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.rand(4, 10, 64)
    y_type = torch.tensor([6, 7, 6, 7])
    y_mode = torch.full((4,), -1)
    type_logits, mode_logits = model(x)
    loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head_mode.weight.grad is None


def test_mixed_batch_loss_is_finite_and_trains_mode_head():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.rand(4, 10, 64)
    y_type = torch.tensor([0, 1, 2, 3])
    y_mode = torch.tensor([0, -1, 2, -1])
    type_logits, mode_logits = model(x)
    loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head_mode.weight.grad is not None


def test_evaluate_type_accuracy_in_unit_range(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(12, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.full(12, 3, dtype=np.int64))
        fh.create_dataset("is_known", data=np.ones(12, dtype=bool))
    ds = SyntheticPDWDataset(path)
    acc = evaluate_type_accuracy(_tiny_model(), ds, n_types=8, device="cpu")
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -k "batch or evaluate" -q`
Expected: FAIL — `ImportError: cannot import name '_classifier_loss'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/robustness.py` (extend imports):

```python
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cog_ew.temporal_cnn_elint.metrics import macro_accuracy
from cog_ew.temporal_cnn_elint.model import TemporalCNN
```

```python
def _classifier_loss(
    type_logits: torch.Tensor,
    mode_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_mode: torch.Tensor,
) -> torch.Tensor:
    type_loss = F.cross_entropy(type_logits, y_type)
    valid = y_mode >= 0
    if bool(valid.any()):
        mode_loss = F.cross_entropy(mode_logits[valid], y_mode[valid])
    else:
        mode_loss = type_logits.new_zeros(())
    return type_loss + mode_loss


@torch.no_grad()
def evaluate_type_accuracy(
    model: TemporalCNN,
    dataset: Dataset[Any],
    n_types: int,
    device: str,
) -> float:
    dev = torch.device(device)
    model.eval()
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=256)
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y_type, _, _ in loader:
        type_pred, _, _ = model.predict(x.to(dev))
        preds.append(type_pred.cpu())
        targets.append(y_type)
    return macro_accuracy(torch.cat(preds), torch.cat(targets), n_types)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -q`
Expected: PASS (all robustness tests so far).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/robustness.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/robustness.py tests/gan_signals/test_robustness.py
git commit -m "$(cat <<'EOF'
feat(gan): _classifier_loss (modo enmascarado) + evaluate_type_accuracy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `_fit_classifier`

**Files:**
- Modify: `src/cog_ew/gan_signals/robustness.py`
- Test: `tests/gan_signals/test_robustness.py`

**Interfaces:**
- Consumes: `_classifier_loss`, `evaluate_type_accuracy` (Task 3); `TemporalCNN`/`TemporalCNNConfig`.
- Produces:
  - `_fit_classifier(model_config, train_ds, val_ds, *, epochs, batch_size, lr, weight_decay, device) -> TemporalCNN` — entrena con `_classifier_loss`, selecciona por mejor `evaluate_type_accuracy` en `val_ds`, devuelve el modelo con los mejores pesos.

- [ ] **Step 1: Write the failing test**

```python
def test_fit_classifier_returns_trained_model(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(40, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.random.randint(0, 8, 40).astype(np.int64))
        fh.create_dataset("is_known", data=np.ones(40, dtype=bool))
    ds = SyntheticPDWDataset(path)
    from cog_ew.gan_signals.robustness import _fit_classifier

    model = _fit_classifier(
        TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0),
        ds,
        ds,
        epochs=2,
        batch_size=16,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )
    assert isinstance(model, TemporalCNN)
    acc = evaluate_type_accuracy(model, ds, n_types=8, device="cpu")
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -k fit -q`
Expected: FAIL — `ImportError: cannot import name '_fit_classifier'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/robustness.py` (extend imports with `from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig`):

```python
def _fit_classifier(
    model_config: TemporalCNNConfig,
    train_ds: Dataset[Any],
    val_ds: Dataset[Any],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> TemporalCNN:
    dev = torch.device(device)
    model = TemporalCNN(model_config).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_acc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        for x, y_type, y_mode, _ in train_loader:
            x = x.to(dev)
            y_type = y_type.to(dev)
            y_mode = y_mode.to(dev)
            type_logits, mode_logits = model(x)
            loss = _classifier_loss(type_logits, mode_logits, y_type, y_mode)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        acc = evaluate_type_accuracy(model, val_ds, model_config.n_types, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -q`
Expected: PASS (all robustness tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/robustness.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/robustness.py tests/gan_signals/test_robustness.py
git commit -m "$(cat <<'EOF'
feat(gan): _fit_classifier — entrena M2 con selección por val (modo enmascarado)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `run_robustness_experiment` (integración)

**Files:**
- Modify: `src/cog_ew/gan_signals/robustness.py`
- Test: `tests/gan_signals/test_robustness.py`

**Interfaces:**
- Consumes: `RobustnessConfig` (Task 2), `_fit_classifier`, `evaluate_type_accuracy` (Tasks 3–4), `SyntheticPDWDataset` (Task 1); `PDWSyntheticDataset`/`PDWConfig`, `EmitterLibrary`, `split_dataset`, `profile_latency`.
- Produces:
  - `run_robustness_experiment(config: RobustnessConfig) -> dict[str, Any]` — entrena baseline y aumentado con la misma seed, evalúa sobre los retenidos reales, escribe `run_meta.json`/`metrics.json`, y devuelve `{"baseline", "augmented", "delta", "relative_improvement", "global"}` (con `"global"` = `{"baseline", "augmented"}`).

- [ ] **Step 1: Write the failing test**

```python
import json
from dataclasses import replace

import h5py
import numpy as np

from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.robustness import RobustnessConfig, run_robustness_experiment


def _tiny_robustness_config(tmp_path) -> RobustnessConfig:
    synth = tmp_path / "s.h5"
    n = 24
    with h5py.File(synth, "w") as fh:
        fh.create_dataset("X", data=np.random.rand(n, 10, 64).astype(np.float32))
        fh.create_dataset("source_a", data=np.repeat([6, 7], n // 2).astype(np.int64))
        fh.create_dataset("is_known", data=np.ones(n, dtype=bool))
    pdw = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml", n_trains=2, n_pulses=256, window=64
    )
    return RobustnessConfig(
        synthetic_path=str(synth),
        held_out=("LPI-FMCW", "LPI-polyphase"),
        model=TemporalCNNConfig(hidden=8, dilations=(1,), dropout=0.0),
        pdw=pdw,
        epochs=1,
        batch_size=16,
        seed=0,
        device="cpu",
        out_dir=str(tmp_path / "run"),
    )


def test_run_robustness_experiment_reports_delta(tmp_path):
    config = _tiny_robustness_config(tmp_path)
    result = run_robustness_experiment(config)
    assert set(result) == {"baseline", "augmented", "delta", "relative_improvement", "global"}
    assert result["delta"] == result["augmented"] - result["baseline"]
    base = result["baseline"]
    expected_rel = (result["augmented"] - base) / base if base > 0 else float("inf")
    assert result["relative_improvement"] == expected_rel
    assert set(result["global"]) == {"baseline", "augmented"}
    out = tmp_path / "run"
    assert (out / "run_meta.json").is_file()
    assert json.loads((out / "metrics.json").read_text())["delta"] == result["delta"]


def test_run_robustness_experiment_is_reproducible(tmp_path):
    r1 = run_robustness_experiment(_tiny_robustness_config(tmp_path / "a"))
    r2 = run_robustness_experiment(_tiny_robustness_config(tmp_path / "b"))
    assert r1["baseline"] == r2["baseline"]
    assert r1["augmented"] == r2["augmented"]
```

(Add `from cog_ew.temporal_cnn_elint.model import TemporalCNNConfig` to the test imports if not already present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -k run_robustness -q`
Expected: FAIL — `ImportError: cannot import name 'run_robustness_experiment'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/robustness.py` (extend imports):

```python
import hashlib
import json
import platform
import random
from dataclasses import asdict, replace

import h5py
import numpy as np

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.data.synthetic_loader import SyntheticPDWDataset
from cog_ew.temporal_cnn_elint.metrics import profile_latency
from torch.utils.data import ConcatDataset
```

```python
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_robustness_experiment(config: RobustnessConfig) -> dict[str, Any]:
    library = EmitterLibrary.from_yaml(config.pdw.library_path)
    names = library.emitter_names()
    held_out_ids = tuple(names.index(n) for n in config.held_out)
    catalogued = tuple(n for n in names if n not in config.held_out)

    cat_ds = PDWSyntheticDataset(replace(config.pdw, emitters=catalogued))
    train_ds, val_ds = split_dataset(cat_ds, (0.85, 0.15), config.seed)
    held_ds = PDWSyntheticDataset(replace(config.pdw, emitters=config.held_out))

    aug_emitters = held_out_ids if config.augment_held_out_only else None
    synth_ds = SyntheticPDWDataset(config.synthetic_path, emitters=aug_emitters, known_only=True)

    n_types = config.model.n_types
    fit_kwargs = dict(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
    )

    _set_seeds(config.seed)
    base_model = _fit_classifier(config.model, train_ds, val_ds, **fit_kwargs)
    base_acc = evaluate_type_accuracy(base_model, held_ds, n_types, config.device)

    _set_seeds(config.seed)
    aug_model = _fit_classifier(
        config.model, ConcatDataset([train_ds, synth_ds]), val_ds, **fit_kwargs
    )
    aug_acc = evaluate_type_accuracy(aug_model, held_ds, n_types, config.device)

    global_eval = ConcatDataset([val_ds, held_ds])
    delta = aug_acc - base_acc
    rel = delta / base_acc if base_acc > 0 else float("inf")
    metrics: dict[str, Any] = {
        "baseline": base_acc,
        "augmented": aug_acc,
        "delta": delta,
        "relative_improvement": rel,
        "global": {
            "baseline": evaluate_type_accuracy(base_model, global_eval, n_types, config.device),
            "augmented": evaluate_type_accuracy(aug_model, global_eval, n_types, config.device),
        },
    }

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = next(iter(DataLoader(held_ds, batch_size=1)))[0]
    mean_ms, p99_ms = profile_latency(aug_model, sample, n_warmup=5, n_iter=50, device=config.device)
    on_disk = {**metrics, "latency_mean_ms": mean_ms, "latency_p99_ms": p99_ms}
    (out_dir / "metrics.json").write_text(json.dumps(on_disk, indent=2))

    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    run_meta = {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "synthetic_hash": _file_sha256(config.synthetic_path),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    return metrics
```

> Nota: el dict devuelto tiene exactamente las 5 claves (`baseline`, `augmented`, `delta`,
> `relative_improvement`, `global`); la latencia (`latency_mean_ms`/`latency_p99_ms`) se añade solo a
> la copia escrita en `metrics.json` vía `on_disk = {**metrics, ...}`, no al dict devuelto.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_robustness.py -q`
Expected: PASS (all robustness tests).

- [ ] **Step 5: Full suite + lint + type-check**

Run: `.venv/bin/ruff format src/cog_ew/ tests/ && .venv/bin/ruff check src/cog_ew/ tests/ && .venv/bin/mypy src/cog_ew/gan_signals/ src/cog_ew/data/synthetic_loader.py && .venv/bin/python -m pytest -q`
Expected: all clean; full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/robustness.py tests/gan_signals/test_robustness.py
git commit -m "$(cat <<'EOF'
feat(gan): run_robustness_experiment — baseline vs aumentado (ancla +22%)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Experimento leave-emitters-out (catalogados train/val, retenidos test, sintético de aumento) → Task 5. ✅
- Supervisión solo de tipo para sintéticos (`mode=-1` + máscara) → Task 1 (mode=-1), Task 3 (`_classifier_loss`). ✅
- Resolución nombre→id global → Task 5. ✅
- Métrica ancla `macro_acc_type` sobre retenidos + delta + relative_improvement (guard base==0) → Task 5. ✅
- `global` context = `{baseline, augmented}` → Task 5. ✅
- Misma seed baseline/aumentado → Task 5 (`_set_seeds` antes de cada fit). ✅
- `SyntheticPDWDataset` 4-tuple con `mode=-1`, filtros → Task 1. ✅
- Config YAML + reproducibilidad (`run_meta`, `synthetic_hash`, deps) → Task 2, Task 5. ✅
- Seguridad (sin parámetros reales; `weights_only` donde haya carga) → ningún `torch.load` en C; N/A salvo nota. ✅
- Fuera de alcance (Colab real, relabel de modo, tipos interpolados) → no incluidos. ✅

**Placeholder scan:** sin TBD/TODO; todo el código está completo. La nota de Task 3-step-4 (latencia fuera del dict devuelto) está resuelta con código explícito. ✅

**Type consistency:** `SyntheticPDWDataset(hdf5_path, *, emitters, known_only)`; `RobustnessConfig(.from_yaml)`; `_classifier_loss(type_logits, mode_logits, y_type, y_mode)`; `evaluate_type_accuracy(model, dataset, n_types, device)`; `_fit_classifier(model_config, train_ds, val_ds, *, epochs, batch_size, lr, weight_decay, device)`; `run_robustness_experiment(config)`. Consistentes entre tareas. ✅

---

## Execution Handoff

Tras guardar el plan, ofrecer al usuario la elección de ejecución (Subagent-Driven recomendado vs Inline).
