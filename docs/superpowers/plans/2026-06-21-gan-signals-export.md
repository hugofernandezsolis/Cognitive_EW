# Modelo 4 · Sub-pieza B — Muestreo, export masivo y validez · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Muestrear, exportar masivamente a `data/synthetic/` (HDF5) y medir la validez de señales PDW sintéticas generadas por el cWGAN-GP de la sub-pieza A, incluyendo tipos de radar no catalogados por interpolación de embeddings.

**Architecture:** Tres módulos en `src/cog_ew/gan_signals/`: `sampler.py` (catálogo de tipos por interpolación + carga del generador desde checkpoint + muestreo por tipo), `validity.py` (validez estructural, realismo distribucional por-feature con Wasserstein-1/scipy, y diversidad/mode-collapse), y `export.py` (orquesta: rellena un HDF5 pre-dimensionado por slices de tipo, calcula métricas contra un lote real, escribe `run_meta.json`/`metrics.json`).

**Tech Stack:** Python 3.11+, PyTorch, NumPy, h5py, scipy (Wasserstein-1), PyYAML, pytest. Herramientas vía `.venv/bin/<tool>`; tests vía `.venv/bin/python -m pytest`.

## Global Constraints

- **Convención de ejes:** ventanas PDW channels-first `(N, 10, 64)` (10 features = canales, 64 pulsos). Canales 0–4 continuos en [0,1] (`rf, pw, pa, aoa, pri`); canales 5–9 one-hot `intra_pulse_mod`.
- **Consume de la sub-pieza A:** `PDWGenerator(z_dim, e_dim, channels)` y `TypeEmbedding(n_emitters, e_dim)` de `cog_ew.gan_signals.generator`; `TypeEmbedding.interpolate(id_a, id_b, alpha) -> (e_dim,)`; `PDWGenerator.sample(e) -> (B,10,64)`. Checkpoint de A: dict con claves `"generator"`, `"embedding"`, `"critic"` (state_dicts).
- **Datos reales para validez:** `PDWSyntheticDataset(PDWConfig)` de `cog_ew.data.pdw_dataset`; items `(x (10,64), type, mode, threat)`. Catálogo `configs/temporal_cnn_elint/emitters.yaml` → `n_emitters = 8`.
- **Seguridad:** `torch.load(..., weights_only=True)` SIEMPRE; no exponer parámetros de amenazas reales (solo catálogo sintético + checkpoint entrenado sobre él).
- **Inferencia:** tras cargar, poner `generator.eval()` y `embedding.eval()` (BatchNorm usa running stats → muestreo determinista y estable).
- **Reproducibilidad:** `_set_seeds` (`random`, `numpy`, `torch`); hiperparámetros SOLO en YAML (`configs/gan_signals/export.yaml`), nunca hardcodeados; `run_meta.json` con seed + hiperparámetros + `config_hash` + `checkpoint_hash` + versiones de dependencias.
- **Escala:** el HDF5 se pre-dimensiona a `N = n_types * samples_per_type` y se rellena por slices de tipo (no acumular ~230k ventanas en RAM). Las métricas se calculan sobre un buffer acotado (`n_real_compare`), no sobre las N completas.
- **mypy:** el proyecto tiene `ignore_missing_imports = true` — importar `scipy`/`h5py` directamente, sin comentarios de ignore.
- **Calidad:** `ruff format` + `ruff check` + `mypy` limpios antes de cada commit; type hints en funciones públicas; comentarios solo el *por qué* no obvio.
- **Commits:** terminan con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `SyntheticType` + `build_type_catalog`

**Files:**
- Create: `src/cog_ew/gan_signals/sampler.py`
- Test: `tests/gan_signals/test_sampler.py`

**Interfaces:**
- Produces:
  - `SyntheticType` (frozen dataclass): `type_id: int`, `source_a: int`, `source_b: int`, `alpha: float`, `is_known: bool`.
  - `build_type_catalog(n_known: int, *, alphas: tuple[float, ...], extrapolate: bool = False) -> list[SyntheticType]`. Primero los `n_known` tipos conocidos (`source_a == source_b == i`, `alpha = 0.0`, `is_known = True`); luego, para cada par `(a, b)` con `a < b`, un tipo por cada alpha (`is_known = False`); si `extrapolate`, añade alphas `-0.25` y `1.25`. `type_id` secuencial.

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.gan_signals.sampler import SyntheticType, build_type_catalog


def test_catalog_has_known_first_then_interpolated():
    catalog = build_type_catalog(8, alphas=(0.25, 0.5, 0.75), extrapolate=False)
    assert len(catalog) >= 50
    known = catalog[:8]
    assert all(t.is_known and t.alpha == 0.0 and t.source_a == t.source_b for t in known)
    assert [t.source_a for t in known] == list(range(8))
    novel = catalog[8:]
    assert all(not t.is_known and t.source_a < t.source_b for t in novel)
    assert all(t.alpha in (0.25, 0.5, 0.75) for t in novel)
    assert [t.type_id for t in catalog] == list(range(len(catalog)))


def test_catalog_is_deterministic():
    a = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    b = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    assert a == b


def test_catalog_extrapolate_adds_out_of_range_alphas():
    catalog = build_type_catalog(8, alphas=(0.5,), extrapolate=True)
    novel_alphas = {t.alpha for t in catalog if not t.is_known}
    assert -0.25 in novel_alphas and 1.25 in novel_alphas
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_sampler.py -q`
Expected: FAIL — `ImportError: cannot import name 'SyntheticType'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Muestreo y catálogo de tipos para el export de señales PDW sintéticas (Modelo 4, sub-pieza B)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticType:
    type_id: int
    source_a: int
    source_b: int
    alpha: float
    is_known: bool


def build_type_catalog(
    n_known: int, *, alphas: tuple[float, ...], extrapolate: bool = False
) -> list[SyntheticType]:
    types: list[SyntheticType] = []
    next_id = 0
    for i in range(n_known):
        types.append(SyntheticType(next_id, i, i, 0.0, True))
        next_id += 1
    novel_alphas = list(alphas) + ([-0.25, 1.25] if extrapolate else [])
    for a in range(n_known):
        for b in range(a + 1, n_known):
            for alpha in novel_alphas:
                types.append(SyntheticType(next_id, a, b, float(alpha), False))
                next_id += 1
    return types
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_sampler.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/sampler.py tests/gan_signals/test_sampler.py && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/sampler.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/sampler.py tests/gan_signals/test_sampler.py
git commit -m "$(cat <<'EOF'
feat(gan): catálogo de tipos sintéticos (conocidos + interpolación de pares)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `load_generator` + `resolve_embedding` + `sample_type`

**Files:**
- Modify: `src/cog_ew/gan_signals/sampler.py`
- Test: `tests/gan_signals/test_sampler.py`

**Interfaces:**
- Consumes: `PDWGenerator`, `TypeEmbedding` (`cog_ew.gan_signals.generator`); `SyntheticType` (Task 1).
- Produces:
  - `load_generator(checkpoint, *, z_dim, e_dim, channels, n_emitters, device="cpu") -> tuple[PDWGenerator, TypeEmbedding]` — carga state_dicts `"generator"`/`"embedding"` con `weights_only=True`, ambos en `.eval()`.
  - `resolve_embedding(embedding, stype) -> torch.Tensor` — `(e_dim,)`; lookup si conocido, `interpolate` si novedoso.
  - `sample_type(generator, embedding, stype, n, device="cpu") -> torch.Tensor` — `(n, 10, 64)`, bajo `no_grad`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding
from cog_ew.gan_signals.sampler import (
    SyntheticType,
    load_generator,
    resolve_embedding,
    sample_type,
)


def _save_ckpt(tmp_path):
    gen = PDWGenerator(z_dim=8, e_dim=4, channels=8)
    emb = TypeEmbedding(n_emitters=8, e_dim=4)
    path = tmp_path / "best.pt"
    torch.save(
        {"generator": gen.state_dict(), "embedding": emb.state_dict(), "critic": {}}, path
    )
    return path


def test_load_generator_roundtrips(tmp_path):
    path = _save_ckpt(tmp_path)
    gen, emb = load_generator(
        path, z_dim=8, e_dim=4, channels=8, n_emitters=8, device="cpu"
    )
    assert isinstance(gen, PDWGenerator) and isinstance(emb, TypeEmbedding)
    assert not gen.training and not emb.training


def test_resolve_embedding_known_and_novel():
    emb = TypeEmbedding(n_emitters=8, e_dim=4)
    known = resolve_embedding(emb, SyntheticType(0, 3, 3, 0.0, True))
    novel = resolve_embedding(emb, SyntheticType(1, 0, 1, 0.5, False))
    assert known.shape == (4,) and novel.shape == (4,)
    assert torch.allclose(known, emb.embedding.weight[3])


def test_sample_type_shape(tmp_path):
    path = _save_ckpt(tmp_path)
    gen, emb = load_generator(path, z_dim=8, e_dim=4, channels=8, n_emitters=8)
    out = sample_type(gen, emb, SyntheticType(0, 0, 1, 0.5, False), n=6)
    assert out.shape == (6, 10, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_sampler.py -k "load or resolve or sample_type" -q`
Expected: FAIL — `ImportError: cannot import name 'load_generator'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/sampler.py` (extend imports):

```python
from pathlib import Path

import torch

from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding
```

```python
def load_generator(
    checkpoint: str | Path,
    *,
    z_dim: int,
    e_dim: int,
    channels: int,
    n_emitters: int,
    device: str = "cpu",
) -> tuple[PDWGenerator, TypeEmbedding]:
    dev = torch.device(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    generator = PDWGenerator(z_dim, e_dim, channels).to(dev)
    embedding = TypeEmbedding(n_emitters, e_dim).to(dev)
    generator.load_state_dict(state["generator"])
    embedding.load_state_dict(state["embedding"])
    generator.eval()
    embedding.eval()
    return generator, embedding


def resolve_embedding(embedding: TypeEmbedding, stype: SyntheticType) -> torch.Tensor:
    if stype.is_known:
        ids = torch.tensor([stype.source_a], dtype=torch.long, device=embedding.embedding.weight.device)
        return embedding(ids).squeeze(0)
    return embedding.interpolate(stype.source_a, stype.source_b, stype.alpha)


@torch.no_grad()
def sample_type(
    generator: PDWGenerator,
    embedding: TypeEmbedding,
    stype: SyntheticType,
    n: int,
    device: str = "cpu",
) -> torch.Tensor:
    dev = torch.device(device)
    e = resolve_embedding(embedding, stype).to(dev)
    e_batch = e.unsqueeze(0).expand(n, -1)
    return generator.sample(e_batch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_sampler.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/sampler.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/sampler.py tests/gan_signals/test_sampler.py
git commit -m "$(cat <<'EOF'
feat(gan): load_generator (weights_only, eval) + resolve_embedding + sample_type

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `structural_validity` + `diversity`

**Files:**
- Create: `src/cog_ew/gan_signals/validity.py`
- Test: `tests/gan_signals/test_validity.py`

**Interfaces:**
- Produces:
  - `structural_validity(windows: torch.Tensor) -> dict[str, float]` — claves `continuous_in_range_frac`, `categorical_onehot_frac`. `windows` es `(N,10,64)`.
  - `diversity(windows: torch.Tensor, type_ids: NDArray[np.int64]) -> dict[str, float]` — claves `mean_intersample_std`, `n_distinct_categorical_patterns`, `n_types`, `coverage`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch

from cog_ew.gan_signals.validity import diversity, structural_validity


def _valid_windows(n=8):
    cont = torch.rand(n, 5, 64)
    cat = torch.zeros(n, 5, 64)
    codes = torch.randint(0, 5, (n, 64))
    cat.scatter_(1, codes.unsqueeze(1), 1.0)
    return torch.cat([cont, cat], dim=1)


def test_structural_validity_perfect_on_valid_windows():
    out = structural_validity(_valid_windows())
    assert out["continuous_in_range_frac"] == 1.0
    assert out["categorical_onehot_frac"] == 1.0


def test_structural_validity_flags_out_of_range_continuous():
    w = _valid_windows()
    w[0, 0, 0] = 5.0
    assert structural_validity(w)["continuous_in_range_frac"] < 1.0


def test_diversity_detects_mode_collapse():
    one = _valid_windows(1)
    collapsed = one.repeat(10, 1, 1)
    type_ids = np.zeros(10, dtype=np.int64)
    out = diversity(collapsed, type_ids)
    assert out["mean_intersample_std"] == 0.0


def test_diversity_full_coverage_and_variety():
    w = _valid_windows(6)
    type_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = diversity(w, type_ids)
    assert out["mean_intersample_std"] > 0.0
    assert out["n_types"] == 3
    assert out["coverage"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_validity.py -q`
Expected: FAIL — `ImportError: cannot import name 'structural_validity'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Métricas de validez y diversidad de señales PDW sintéticas (Modelo 4, sub-pieza B)."""

from __future__ import annotations

import torch
from numpy.typing import NDArray
import numpy as np


def structural_validity(windows: torch.Tensor) -> dict[str, float]:
    cont = windows[:, :5]
    cat = windows[:, 5:]
    in_range = ((cont >= 0.0) & (cont <= 1.0)).float().mean().item()
    sums = cat.sum(dim=1)
    binary = ((cat == 0.0) | (cat == 1.0)).all(dim=1)
    onehot = (torch.isclose(sums, torch.ones_like(sums)) & binary).float().mean().item()
    return {"continuous_in_range_frac": in_range, "categorical_onehot_frac": onehot}


def diversity(windows: torch.Tensor, type_ids: NDArray[np.int64]) -> dict[str, float]:
    mean_std = windows.std(dim=0).mean().item()
    codes = windows[:, 5:].argmax(dim=1)
    patterns = {tuple(row.tolist()) for row in codes}
    unique_types = {int(t) for t in type_ids.tolist()}
    n_types = len(unique_types)
    max_id = int(type_ids.max()) if type_ids.size > 0 else -1
    coverage = n_types / (max_id + 1) if max_id >= 0 else 0.0
    return {
        "mean_intersample_std": mean_std,
        "n_distinct_categorical_patterns": float(len(patterns)),
        "n_types": float(n_types),
        "coverage": coverage,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_validity.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/validity.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/validity.py tests/gan_signals/test_validity.py
git commit -m "$(cat <<'EOF'
feat(gan): validez estructural + diversidad (proxy mode-collapse, coverage)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `distributional_realism`

**Files:**
- Modify: `src/cog_ew/gan_signals/validity.py`
- Test: `tests/gan_signals/test_validity.py`

**Interfaces:**
- Consumes: nada de tareas previas (función pura sobre tensores).
- Produces:
  - `distributional_realism(generated: torch.Tensor, real: torch.Tensor) -> dict[str, Any]` — claves `wasserstein1_per_feature` (lista de 5 floats ≥ 0), `wasserstein1_mean` (float), `categorical_tv_distance` (float en [0,1]). Ambos tensores `(N,10,64)`.

- [ ] **Step 1: Write the failing test**

```python
from typing import Any

from cog_ew.gan_signals.validity import distributional_realism


def test_distributional_realism_keys_and_ranges():
    gen = _valid_windows(32)
    real = _valid_windows(40)
    out: dict[str, Any] = distributional_realism(gen, real)
    assert len(out["wasserstein1_per_feature"]) == 5
    assert all(v >= 0.0 for v in out["wasserstein1_per_feature"])
    assert out["wasserstein1_mean"] >= 0.0
    assert 0.0 <= out["categorical_tv_distance"] <= 1.0


def test_distributional_realism_zero_for_identical_distributions():
    w = _valid_windows(50)
    out = distributional_realism(w, w)
    assert out["wasserstein1_mean"] == 0.0
    assert out["categorical_tv_distance"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_validity.py -k distributional -q`
Expected: FAIL — `ImportError: cannot import name 'distributional_realism'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/validity.py` (extend imports with `from typing import Any` and `from scipy.stats import wasserstein_distance`):

```python
def distributional_realism(generated: torch.Tensor, real: torch.Tensor) -> dict[str, Any]:
    gen = generated.detach().cpu().numpy()
    rl = real.detach().cpu().numpy()
    per_feature = [
        float(wasserstein_distance(gen[:, f, :].reshape(-1), rl[:, f, :].reshape(-1)))
        for f in range(5)
    ]
    gen_codes = gen[:, 5:, :].argmax(axis=1).reshape(-1)
    real_codes = rl[:, 5:, :].argmax(axis=1).reshape(-1)
    gen_hist = np.bincount(gen_codes, minlength=5).astype(np.float64)
    real_hist = np.bincount(real_codes, minlength=5).astype(np.float64)
    gen_hist /= gen_hist.sum()
    real_hist /= real_hist.sum()
    tv = 0.5 * float(np.abs(gen_hist - real_hist).sum())
    return {
        "wasserstein1_per_feature": per_feature,
        "wasserstein1_mean": float(np.mean(per_feature)),
        "categorical_tv_distance": tv,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_validity.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/validity.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/validity.py tests/gan_signals/test_validity.py
git commit -m "$(cat <<'EOF'
feat(gan): realismo distribucional (Wasserstein-1 por feature + TV categórica)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `ExportConfig` + YAML

**Files:**
- Create: `src/cog_ew/gan_signals/export.py`
- Create: `configs/gan_signals/export.yaml`
- Test: `tests/gan_signals/test_export.py`

**Interfaces:**
- Produces:
  - `ExportConfig` (dataclass): `checkpoint: str`, `z_dim: int = 64`, `e_dim: int = 16`, `channels: int = 64`, `n_emitters: int = 8`, `alphas: tuple[float, ...] = (0.25, 0.5, 0.75)`, `extrapolate: bool = False`, `samples_per_type: int = 2500`, `out_path: str = "data/synthetic/wgan_gp.h5"`, `library_path: str = "configs/temporal_cnn_elint/emitters.yaml"`, `n_real_compare: int = 4000`, `seed: int = 0`, `device: str = "cpu"`.
  - `ExportConfig.from_yaml(path) -> ExportConfig` (convierte `alphas` a tupla).

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.gan_signals.export import ExportConfig


def test_export_config_from_yaml():
    config = ExportConfig.from_yaml("configs/gan_signals/export.yaml")
    assert config.n_emitters == 8
    assert isinstance(config.alphas, tuple)
    assert config.samples_per_type > 0
    assert config.out_path.endswith(".h5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_export.py -q`
Expected: FAIL — `ImportError: cannot import name 'ExportConfig'`.

- [ ] **Step 3: Write the config file**

Create `configs/gan_signals/export.yaml`:

```yaml
checkpoint: runs/gan_signals/wgan_gp/best.pt
z_dim: 64
e_dim: 16
channels: 64
n_emitters: 8
alphas: [0.25, 0.5, 0.75]
extrapolate: false
samples_per_type: 2500
out_path: data/synthetic/wgan_gp.h5
library_path: configs/temporal_cnn_elint/emitters.yaml
n_real_compare: 4000
seed: 0
device: cpu
```

- [ ] **Step 4: Write minimal implementation**

```python
"""Export masivo de señales PDW sintéticas a HDF5 (Modelo 4, sub-pieza B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExportConfig:
    checkpoint: str
    z_dim: int = 64
    e_dim: int = 16
    channels: int = 64
    n_emitters: int = 8
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75)
    extrapolate: bool = False
    samples_per_type: int = 2500
    out_path: str = "data/synthetic/wgan_gp.h5"
    library_path: str = "configs/temporal_cnn_elint/emitters.yaml"
    n_real_compare: int = 4000
    seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExportConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if raw.get("alphas") is not None:
            raw["alphas"] = tuple(raw["alphas"])
        return cls(**raw)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_export.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/export.py`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/cog_ew/gan_signals/export.py configs/gan_signals/export.yaml tests/gan_signals/test_export.py
git commit -m "$(cat <<'EOF'
feat(gan): ExportConfig + export.yaml para el muestreo masivo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `export_synthetic` (integración: HDF5 + run_meta + metrics)

**Files:**
- Modify: `src/cog_ew/gan_signals/export.py`
- Test: `tests/gan_signals/test_export.py`

**Interfaces:**
- Consumes: `build_type_catalog`, `load_generator`, `sample_type` (Tasks 1–2); `structural_validity`, `diversity`, `distributional_realism` (Tasks 3–4); `ExportConfig` (Task 5); `PDWConfig`/`PDWSyntheticDataset` (`cog_ew.data.pdw_dataset`).
- Produces:
  - `export_synthetic(config: ExportConfig) -> dict[str, Any]` — escribe `config.out_path` (HDF5) + `run_meta.json` y `metrics.json` en su carpeta. HDF5: datasets `X (N,10,64) f32`, `type_id`, `source_a`, `source_b` (`i64`), `alpha (f32)`, `is_known (bool)`; attrs `n_types`, `samples_per_type`, `checkpoint_hash`, `seed`. `metrics.json` incluye claves de `structural_validity`/`diversity`/`distributional_realism` + `n_windows`, `n_types`.

- [ ] **Step 1: Write the failing test**

```python
import json
from dataclasses import replace

import h5py
import numpy as np
import torch

from cog_ew.gan_signals.export import ExportConfig, export_synthetic
from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


def _tiny_config(tmp_path) -> ExportConfig:
    gen = PDWGenerator(z_dim=8, e_dim=4, channels=8)
    emb = TypeEmbedding(n_emitters=4, e_dim=4)
    ckpt = tmp_path / "best.pt"
    torch.save({"generator": gen.state_dict(), "embedding": emb.state_dict(), "critic": {}}, ckpt)
    return ExportConfig(
        checkpoint=str(ckpt),
        z_dim=8,
        e_dim=4,
        channels=8,
        n_emitters=4,
        alphas=(0.5,),
        samples_per_type=5,
        out_path=str(tmp_path / "out" / "synth.h5"),
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        n_real_compare=20,
        seed=0,
        device="cpu",
    )


def test_export_writes_hdf5_and_metrics(tmp_path):
    config = _tiny_config(tmp_path)
    result = export_synthetic(config)
    out = tmp_path / "out"
    n_types = 4 + 6  # 4 known + C(4,2)=6 pairs * 1 alpha
    n = n_types * 5
    with h5py.File(config.out_path, "r") as fh:
        assert fh["X"].shape == (n, 10, 64)
        assert fh["type_id"].shape == (n,)
        assert fh["is_known"].dtype == np.bool_
        assert fh.attrs["n_types"] == n_types
        assert fh.attrs["seed"] == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"continuous_in_range_frac", "wasserstein1_mean", "mean_intersample_std", "n_windows"} <= set(metrics)
    assert (out / "run_meta.json").is_file()
    assert result["n_windows"] == n


def test_export_is_reproducible_by_seed(tmp_path):
    base = _tiny_config(tmp_path)
    c1 = replace(base, out_path=str(tmp_path / "r1" / "s.h5"))
    c2 = replace(base, out_path=str(tmp_path / "r2" / "s.h5"))
    export_synthetic(c1)
    export_synthetic(c2)
    with h5py.File(c1.out_path, "r") as f1, h5py.File(c2.out_path, "r") as f2:
        assert np.array_equal(f1["X"][:], f2["X"][:])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_export.py -k "writes or reproducible" -q`
Expected: FAIL — `ImportError: cannot import name 'export_synthetic'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/export.py` (extend imports):

```python
import hashlib
import json
import platform
import random
from dataclasses import asdict
from typing import Any

import h5py
import numpy as np
import torch

from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset
from cog_ew.gan_signals.sampler import build_type_catalog, load_generator, sample_type
from cog_ew.gan_signals.validity import (
    distributional_realism,
    diversity,
    structural_validity,
)
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


def _real_windows(library_path: str, n: int) -> torch.Tensor:
    dataset = PDWSyntheticDataset(PDWConfig(library_path=library_path))
    count = min(n, len(dataset))
    return torch.stack([dataset[i][0] for i in range(count)])


def export_synthetic(config: ExportConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    generator, embedding = load_generator(
        config.checkpoint,
        z_dim=config.z_dim,
        e_dim=config.e_dim,
        channels=config.channels,
        n_emitters=config.n_emitters,
        device=config.device,
    )
    catalog = build_type_catalog(
        config.n_emitters, alphas=config.alphas, extrapolate=config.extrapolate
    )
    n_types = len(catalog)
    spt = config.samples_per_type
    n = n_types * spt

    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_type_val = max(1, -(-config.n_real_compare // n_types))
    val_windows: list[torch.Tensor] = []
    val_types: list[np.ndarray] = []

    with h5py.File(out_path, "w") as fh:
        x_ds = fh.create_dataset("X", shape=(n, 10, 64), dtype="float32")
        type_ds = fh.create_dataset("type_id", shape=(n,), dtype="int64")
        a_ds = fh.create_dataset("source_a", shape=(n,), dtype="int64")
        b_ds = fh.create_dataset("source_b", shape=(n,), dtype="int64")
        alpha_ds = fh.create_dataset("alpha", shape=(n,), dtype="float32")
        known_ds = fh.create_dataset("is_known", shape=(n,), dtype=bool)
        for idx, stype in enumerate(catalog):
            lo = idx * spt
            hi = lo + spt
            windows = sample_type(generator, embedding, stype, spt, config.device)
            x_ds[lo:hi] = windows.cpu().numpy()
            type_ds[lo:hi] = stype.type_id
            a_ds[lo:hi] = stype.source_a
            b_ds[lo:hi] = stype.source_b
            alpha_ds[lo:hi] = stype.alpha
            known_ds[lo:hi] = stype.is_known
            take = min(per_type_val, spt)
            val_windows.append(windows[:take].cpu())
            val_types.append(np.full(take, stype.type_id, dtype=np.int64))
        fh.attrs["n_types"] = n_types
        fh.attrs["samples_per_type"] = spt
        fh.attrs["checkpoint_hash"] = _file_sha256(config.checkpoint)
        fh.attrs["seed"] = config.seed

    val_gen = torch.cat(val_windows)
    val_type_ids = np.concatenate(val_types)
    real = _real_windows(config.library_path, config.n_real_compare)

    metrics: dict[str, Any] = {
        **structural_validity(val_gen),
        **distributional_realism(val_gen, real),
        **diversity(val_gen, val_type_ids),
        "n_windows": n,
        "n_types": n_types,
    }
    (out_path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    run_meta = {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "checkpoint_hash": _file_sha256(config.checkpoint),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    (out_path.parent / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_export.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Full suite + lint + type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/ && .venv/bin/python -m pytest -q`
Expected: all clean; full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/export.py tests/gan_signals/test_export.py
git commit -m "$(cat <<'EOF'
feat(gan): export_synthetic — HDF5 por slices + run_meta/metrics de validez

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Estrategia de novedad (interpolación de pares, conocidos + novedosos, extrapolación) → Task 1. ✅
- Carga del generador desde checkpoint (`weights_only`, `eval`) + resolución de embedding + muestreo → Task 2. ✅
- Validez estructural + diversidad (proxy mode-collapse, coverage) → Task 3. ✅
- Realismo distribucional (Wasserstein-1 por feature + TV categórica) → Task 4. ✅
- Config YAML + reproducibilidad → Task 5, Task 6. ✅
- Esquema HDF5 (X + provenance + attrs) por slices, sin acumular en RAM → Task 6. ✅
- `run_meta.json` (seed, config_hash, checkpoint_hash, deps) + `metrics.json` → Task 6. ✅
- Validez calculada sobre buffer acotado (`n_real_compare`), no sobre N completas → Task 6. ✅
- Seguridad (`weights_only=True`, solo catálogo sintético) → Tasks 2, 6. ✅
- Fuera de alcance (C +22%, loader de M2, logging por intervalo) → no incluidos. ✅

**Placeholder scan:** sin TBD/TODO; código completo en cada step. ✅

**Type consistency:** `SyntheticType(type_id, source_a, source_b, alpha, is_known)`; `build_type_catalog(n_known, *, alphas, extrapolate)`; `load_generator(checkpoint, *, z_dim, e_dim, channels, n_emitters, device)`; `resolve_embedding(embedding, stype)`; `sample_type(generator, embedding, stype, n, device)`; `structural_validity(windows)`; `diversity(windows, type_ids)`; `distributional_realism(generated, real)`; `ExportConfig.from_yaml`; `export_synthetic(config)`. Consistentes entre tareas. ✅

---

## Execution Handoff

Tras guardar el plan, ofrecer al usuario la elección de ejecución (Subagent-Driven recomendado vs Inline).
