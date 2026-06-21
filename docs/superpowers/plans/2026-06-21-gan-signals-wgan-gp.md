# Modelo 4 · Sub-pieza A — Núcleo cWGAN-GP · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el núcleo cWGAN-GP del Modelo 4: un generador condicional y un crítico Wasserstein 1D-CNN que producen/evalúan ventanas PDW sintéticas `(B, 10, 64)` listas para alimentar al Modelo 2.

**Architecture:** Generador `G(z, e)` (ConvTranspose1d, cabeza partida sigmoid + Gumbel-softmax straight-through) y crítico `D(x, e)` (Conv1d + LayerNorm, sin sigmoid) condicionados por un embedding de tipo continuo compartido `TypeEmbedding`. Entrenamiento WGAN-GP (gradient penalty λ=10) sobre `PDWSyntheticDataset`. Reproducibilidad vía seeds + `run_meta.json` + `metrics.json` con latencia del generador.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, pytest. Herramientas vía `.venv/bin/<tool>`; tests vía `.venv/bin/python -m pytest`.

## Global Constraints

- **Convención de ejes:** las ventanas PDW son channels-first `(10, 64)` (10 features = canales, 64 pulsos = longitud). `PDWGenerator` produce y `PDWCritic` consume `(B, 10, 64)`, idéntico a los lotes del Modelo 2. Sin transposiciones.
- **Canales:** 0–4 continuas (`rf, pw, pa, aoa, pri`, normalizadas [0,1]); 5–9 one-hot `intra_pulse_mod` (`none, lfm, barker, fmcw, polyphase`).
- **Datos reales:** `PDWSyntheticDataset` (`cog_ew.data.pdw_dataset`); catálogo por defecto `configs/temporal_cnn_elint/emitters.yaml` → **`n_emitters = 8`**.
- **WGAN-GP:** **prohibido BatchNorm en el crítico** (invalida el gradient penalty); el generador sí puede usar BatchNorm. Crítico sin sigmoid final (Wasserstein). Adam `betas=(0.0, 0.9)`.
- **Reproducibilidad:** `_set_seeds` (`random.seed`, `numpy.random.seed`, `torch.manual_seed`); hiperparámetros **solo en YAML** (`configs/gan_signals/`), nunca hardcodeados; `run_meta.json` con seed + hiperparámetros + `config_hash` + versiones de dependencias.
- **Seguridad:** `torch.load(..., weights_only=True)` siempre; no exponer parámetros de amenazas reales (solo catálogo sintético).
- **Calidad:** `ruff format` + `ruff check` + `mypy` limpios antes de cada commit; type hints en todas las funciones públicas.
- **Tipo de comentarios:** solo el *por qué* cuando no es obvio; nada de comentarios que narren el *qué*.
- **Commits:** terminan con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `TypeEmbedding` (embedding de tipo continuo)

**Files:**
- Create: `src/cog_ew/gan_signals/generator.py`
- Test: `tests/gan_signals/test_generator.py`

**Interfaces:**
- Produces:
  - `TypeEmbedding(n_emitters: int, e_dim: int)` — `nn.Module`.
  - `TypeEmbedding.forward(ids: torch.Tensor) -> torch.Tensor` — `(B,)` long → `(B, e_dim)`.
  - `TypeEmbedding.interpolate(id_a: int, id_b: int, alpha: float) -> torch.Tensor` — `(e_dim,)`, interpolación lineal `(1-alpha)*w[id_a] + alpha*w[id_b]`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.gan_signals.generator import TypeEmbedding


def test_type_embedding_maps_ids_to_vectors():
    torch.manual_seed(0)
    emb = TypeEmbedding(n_emitters=8, e_dim=16)
    ids = torch.tensor([0, 3, 7], dtype=torch.long)
    out = emb(ids)
    assert out.shape == (3, 16)


def test_type_embedding_interpolates_linearly():
    torch.manual_seed(0)
    emb = TypeEmbedding(n_emitters=8, e_dim=16)
    mid = emb.interpolate(0, 1, alpha=0.5)
    expected = 0.5 * emb.embedding.weight[0] + 0.5 * emb.embedding.weight[1]
    assert mid.shape == (16,)
    assert torch.allclose(mid, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_generator.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'TypeEmbedding'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Generador cWGAN-GP y embedding de tipo para señales PDW sintéticas (Modelo 4)."""

from __future__ import annotations

import torch
from torch import nn


class TypeEmbedding(nn.Module):
    def __init__(self, n_emitters: int, e_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_emitters, e_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.embedding(ids)
        return out

    def interpolate(self, id_a: int, id_b: int, alpha: float) -> torch.Tensor:
        weight = self.embedding.weight
        return (1.0 - alpha) * weight[id_a] + alpha * weight[id_b]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_generator.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/generator.py tests/gan_signals/test_generator.py && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/generator.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/generator.py tests/gan_signals/test_generator.py
git commit -m "$(cat <<'EOF'
feat(gan): TypeEmbedding — embedding de tipo continuo con interpolación

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `PDWGenerator` (generador 1D-CNN con cabeza partida)

**Files:**
- Modify: `src/cog_ew/gan_signals/generator.py`
- Test: `tests/gan_signals/test_generator.py`

**Interfaces:**
- Consumes: `TypeEmbedding` (Task 1).
- Produces:
  - `PDWGenerator(z_dim: int, e_dim: int, channels: int, *, n_continuous: int = 5, n_categorical: int = 5, seq_len: int = 64, gumbel_tau: float = 1.0)` — `nn.Module`.
  - `PDWGenerator.forward(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor` — `z (B, z_dim)`, `e (B, e_dim)` → `(B, 10, 64)`.
  - `PDWGenerator.sample(e: torch.Tensor) -> torch.Tensor` — muestrea `z ~ N(0,1)` internamente → `(B, 10, 64)`.

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.gan_signals.generator import PDWGenerator


def _gen() -> PDWGenerator:
    return PDWGenerator(z_dim=8, e_dim=4, channels=8)


def test_generator_output_shape():
    torch.manual_seed(0)
    gen = _gen()
    z = torch.randn(5, 8)
    e = torch.randn(5, 4)
    out = gen(z, e)
    assert out.shape == (5, 10, 64)


def test_generator_continuous_channels_in_unit_range():
    torch.manual_seed(0)
    gen = _gen()
    out = gen(torch.randn(5, 8), torch.randn(5, 4))
    cont = out[:, :5]
    assert torch.all(cont >= 0.0) and torch.all(cont <= 1.0)


def test_generator_categorical_channels_are_one_hot():
    torch.manual_seed(0)
    gen = _gen()
    out = gen(torch.randn(5, 8), torch.randn(5, 4))
    cat = out[:, 5:]
    sums = cat.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))
    assert torch.all((cat == 0.0) | (cat == 1.0))


def test_generator_is_deterministic_by_seed():
    z = torch.randn(3, 8)
    e = torch.randn(3, 4)
    torch.manual_seed(0)
    a = _gen()(z, e)
    torch.manual_seed(0)
    b = _gen()(z, e)
    assert torch.allclose(a, b)


def test_generator_conditioning_changes_output():
    torch.manual_seed(0)
    gen = _gen()
    z = torch.randn(4, 8)
    out_a = gen(z, torch.zeros(4, 4))
    out_b = gen(z, torch.ones(4, 4))
    assert not torch.allclose(out_a, out_b)


def test_generator_sample_shape():
    torch.manual_seed(0)
    gen = _gen()
    out = gen.sample(torch.randn(6, 4))
    assert out.shape == (6, 10, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_generator.py -k generator -q`
Expected: FAIL — `ImportError: cannot import name 'PDWGenerator'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/generator.py` (add `from torch.nn import functional as F` to imports):

```python
class PDWGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        e_dim: int,
        channels: int,
        *,
        n_continuous: int = 5,
        n_categorical: int = 5,
        seq_len: int = 64,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.gumbel_tau = gumbel_tau
        self.channels = channels
        self.init_len = seq_len // 8
        self.project = nn.Linear(z_dim + e_dim, channels * self.init_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(channels // 2, channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(channels // 2, n_continuous + n_categorical, 3, padding=1)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, e], dim=1)
        x = self.project(x).view(-1, self.channels, self.init_len)
        raw = self.head(self.net(x))
        cont = torch.sigmoid(raw[:, : self.n_continuous])
        cat = F.gumbel_softmax(
            raw[:, self.n_continuous :], tau=self.gumbel_tau, hard=True, dim=1
        )
        return torch.cat([cont, cat], dim=1)

    def sample(self, e: torch.Tensor) -> torch.Tensor:
        z = torch.randn(e.size(0), self.z_dim, device=e.device)
        return self.forward(z, e)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_generator.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/generator.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/generator.py tests/gan_signals/test_generator.py
git commit -m "$(cat <<'EOF'
feat(gan): PDWGenerator 1D-CNN (sigmoid continuas + Gumbel-softmax categóricas)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `PDWCritic` (crítico Wasserstein condicional)

**Files:**
- Create: `src/cog_ew/gan_signals/discriminator.py`
- Test: `tests/gan_signals/test_discriminator.py`

**Interfaces:**
- Produces:
  - `PDWCritic(e_dim: int, channels: int, *, in_channels: int = 10)` — `nn.Module`.
  - `PDWCritic.forward(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor` — `x (B, 10, 64)`, `e (B, e_dim)` → `(B, 1)`. Sin sigmoid (admite negativos). Sin BatchNorm.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.gan_signals.discriminator import PDWCritic


def _critic() -> PDWCritic:
    return PDWCritic(e_dim=4, channels=8)


def test_critic_output_shape():
    torch.manual_seed(0)
    out = _critic()(torch.randn(5, 10, 64), torch.randn(5, 4))
    assert out.shape == (5, 1)


def test_critic_admits_negative_scores():
    torch.manual_seed(0)
    critic = _critic()
    scores = critic(torch.randn(64, 10, 64), torch.randn(64, 4))
    assert scores.min().item() < 0.0


def test_critic_has_no_batchnorm():
    critic = _critic()
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in critic.modules())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_discriminator.py -q`
Expected: FAIL — `ImportError: cannot import name 'PDWCritic'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Crítico Wasserstein (WGAN-GP) condicionado para señales PDW (Modelo 4)."""

from __future__ import annotations

import torch
from torch import nn


class PDWCritic(nn.Module):
    def __init__(self, e_dim: int, channels: int, *, in_channels: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(channels)
        self.out = nn.Linear(channels + e_dim, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.net(x)).squeeze(-1)
        h = self.norm(h)
        out: torch.Tensor = self.out(torch.cat([h, e], dim=1))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_discriminator.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/discriminator.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/discriminator.py tests/gan_signals/test_discriminator.py
git commit -m "$(cat <<'EOF'
feat(gan): PDWCritic Wasserstein 1D-CNN condicional (LayerNorm, sin BatchNorm)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `gradient_penalty` (WGAN-GP)

**Files:**
- Create: `src/cog_ew/gan_signals/train.py`
- Test: `tests/gan_signals/test_train.py`

**Interfaces:**
- Consumes: `PDWCritic` (Task 3).
- Produces:
  - `gradient_penalty(critic: PDWCritic, real: torch.Tensor, fake: torch.Tensor, e: torch.Tensor) -> torch.Tensor` — escalar `()`; penaliza `(‖∇_x̂ D(x̂,e)‖₂ − 1)²`, `x̂` interpolación real/fake con el mismo `e`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.gan_signals.discriminator import PDWCritic
from cog_ew.gan_signals.train import gradient_penalty


def test_gradient_penalty_is_finite_nonnegative_scalar():
    torch.manual_seed(0)
    critic = PDWCritic(e_dim=4, channels=8)
    real = torch.randn(6, 10, 64)
    fake = torch.randn(6, 10, 64)
    e = torch.randn(6, 4)
    gp = gradient_penalty(critic, real, fake, e)
    assert gp.shape == ()
    assert torch.isfinite(gp)
    assert gp.item() >= 0.0


def test_gradient_penalty_is_differentiable_wrt_critic():
    torch.manual_seed(0)
    critic = PDWCritic(e_dim=4, channels=8)
    real = torch.randn(6, 10, 64)
    fake = torch.randn(6, 10, 64)
    e = torch.randn(6, 4)
    gp = gradient_penalty(critic, real, fake, e)
    gp.backward()
    assert any(p.grad is not None for p in critic.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -q`
Expected: FAIL — `ImportError: cannot import name 'gradient_penalty'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Entrenamiento WGAN-GP del núcleo cWGAN-GP de señales PDW (Modelo 4)."""

from __future__ import annotations

import torch

from cog_ew.gan_signals.discriminator import PDWCritic


def gradient_penalty(
    critic: PDWCritic, real: torch.Tensor, fake: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(interpolated, e)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    flat = grads.reshape(batch, -1)
    penalty: torch.Tensor = ((flat.norm(2, dim=1) - 1.0) ** 2).mean()
    return penalty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/train.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/train.py tests/gan_signals/test_train.py
git commit -m "$(cat <<'EOF'
feat(gan): gradient_penalty WGAN-GP (interpolación condicionada por e)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `WGANGPConfig` + seeds + metadata + YAML

**Files:**
- Modify: `src/cog_ew/gan_signals/train.py`
- Create: `configs/gan_signals/pdw.yaml`
- Create: `configs/gan_signals/wgan_gp.yaml`
- Test: `tests/gan_signals/test_train.py`

**Interfaces:**
- Consumes: `PDWConfig` (`cog_ew.data.pdw_dataset`).
- Produces:
  - `WGANGPConfig` — dataclass con campos: `pdw: PDWConfig`, `z_dim: int`, `e_dim: int`, `channels: int`, `n_critic: int`, `lambda_gp: float`, `lr: float`, `gumbel_tau: float`, `batch_size: int`, `total_steps: int`, `device: str`, `seed: int`, `out_dir: str`, `tracking: bool`.
  - `WGANGPConfig.from_yaml(path: str | Path) -> WGANGPConfig` — pop `pdw_config` (ruta) → `PDWConfig.from_yaml`.
  - `_set_seeds(seed: int) -> None`.
  - `_run_metadata(config: WGANGPConfig) -> dict[str, Any]` — claves `seed`, `hyperparameters`, `config_hash`, `dependencies`.

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.data.pdw_dataset import PDWConfig
from cog_ew.gan_signals.train import WGANGPConfig, _run_metadata, _set_seeds


def test_wgangp_config_from_yaml_loads_pdw():
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    assert isinstance(config.pdw, PDWConfig)
    assert config.z_dim > 0
    assert config.lambda_gp == 10.0


def test_set_seeds_is_deterministic():
    _set_seeds(0)
    a = torch.rand(4)
    _set_seeds(0)
    b = torch.rand(4)
    assert torch.allclose(a, b)


def test_run_metadata_has_required_keys():
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    meta = _run_metadata(config)
    assert set(meta) == {"seed", "hyperparameters", "config_hash", "dependencies"}
    assert meta["seed"] == config.seed
    assert {"python", "torch", "numpy"} <= set(meta["dependencies"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -k "config or seeds or metadata" -q`
Expected: FAIL — `ImportError: cannot import name 'WGANGPConfig'`.

- [ ] **Step 3: Write the config files**

Create `configs/gan_signals/pdw.yaml`:

```yaml
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
```

Create `configs/gan_signals/wgan_gp.yaml`:

```yaml
pdw_config: configs/gan_signals/pdw.yaml
z_dim: 64
e_dim: 16
channels: 64
n_critic: 5
lambda_gp: 10.0
lr: 0.0001
gumbel_tau: 1.0
batch_size: 64
total_steps: 20000
device: cpu
seed: 0
out_dir: runs/gan_signals/wgan_gp
tracking: false
```

- [ ] **Step 4: Write minimal implementation**

Add to `src/cog_ew/gan_signals/train.py` (extend imports as shown):

```python
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cog_ew.data.pdw_dataset import PDWConfig


@dataclass
class WGANGPConfig:
    pdw: PDWConfig
    z_dim: int = 64
    e_dim: int = 16
    channels: int = 64
    n_critic: int = 5
    lambda_gp: float = 10.0
    lr: float = 1e-4
    gumbel_tau: float = 1.0
    batch_size: int = 64
    total_steps: int = 20000
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/gan_signals/wgan_gp"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> WGANGPConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        pdw = PDWConfig.from_yaml(raw.pop("pdw_config"))
        return cls(pdw=pdw, **raw)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_metadata(config: WGANGPConfig) -> dict[str, Any]:
    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    return {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -q`
Expected: PASS (all train tests, including Task 4's gradient_penalty tests).

- [ ] **Step 6: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/train.py`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/cog_ew/gan_signals/train.py tests/gan_signals/test_train.py configs/gan_signals/pdw.yaml configs/gan_signals/wgan_gp.yaml
git commit -m "$(cat <<'EOF'
feat(gan): WGANGPConfig + seeds + run_meta + YAML (pdw.yaml, wgan_gp.yaml)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `WGANGP` learner (`critic_update` / `generator_update`)

**Files:**
- Modify: `src/cog_ew/gan_signals/train.py`
- Test: `tests/gan_signals/test_train.py`

**Interfaces:**
- Consumes: `TypeEmbedding`, `PDWGenerator` (Task 1–2), `PDWCritic` (Task 3), `gradient_penalty` (Task 4), `WGANGPConfig` (Task 5).
- Produces:
  - `WGANGP(n_emitters: int, config: WGANGPConfig, device: str)` con atributos `generator: PDWGenerator`, `critic: PDWCritic`, `embedding: TypeEmbedding`, `opt_g`, `opt_c`.
  - `WGANGP.critic_update(real_x: torch.Tensor, ids: torch.Tensor) -> float` — `n_critic`-ésimo update del crítico; usa `e` detached.
  - `WGANGP.generator_update(ids: torch.Tensor) -> float` — update del generador + embedding.

- [ ] **Step 1: Write the failing test**

```python
import copy

import numpy as np

from cog_ew.gan_signals.train import WGANGP, WGANGPConfig


def _learner() -> WGANGP:
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    config.z_dim, config.e_dim, config.channels = 8, 4, 8
    return WGANGP(n_emitters=8, config=config, device="cpu")


def test_critic_update_changes_only_critic_params():
    torch.manual_seed(0)
    learner = _learner()
    real = torch.rand(6, 10, 64)
    ids = torch.randint(0, 8, (6,))
    before_c = copy.deepcopy(learner.critic.state_dict())
    before_g = copy.deepcopy(learner.generator.state_dict())
    loss = learner.critic_update(real, ids)
    assert np.isfinite(loss)
    assert any(
        not torch.allclose(before_c[k], v) for k, v in learner.critic.state_dict().items()
    )
    assert all(
        torch.allclose(before_g[k], v) for k, v in learner.generator.state_dict().items()
    )


def test_generator_update_changes_generator_and_embedding():
    torch.manual_seed(0)
    learner = _learner()
    ids = torch.randint(0, 8, (6,))
    before_g = copy.deepcopy(learner.generator.state_dict())
    before_e = copy.deepcopy(learner.embedding.state_dict())
    loss = learner.generator_update(ids)
    assert np.isfinite(loss)
    assert any(
        not torch.allclose(before_g[k], v) for k, v in learner.generator.state_dict().items()
    )
    assert any(
        not torch.allclose(before_e[k], v) for k, v in learner.embedding.state_dict().items()
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -k update -q`
Expected: FAIL — `ImportError: cannot import name 'WGANGP'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/train.py` (extend imports with the generator pieces):

```python
from cog_ew.gan_signals.generator import PDWGenerator, TypeEmbedding


class WGANGP:
    def __init__(self, n_emitters: int, config: WGANGPConfig, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.embedding = TypeEmbedding(n_emitters, config.e_dim).to(self.device)
        self.generator = PDWGenerator(
            config.z_dim, config.e_dim, config.channels, gumbel_tau=config.gumbel_tau
        ).to(self.device)
        self.critic = PDWCritic(config.e_dim, config.channels).to(self.device)
        gen_params = list(self.generator.parameters()) + list(self.embedding.parameters())
        self.opt_g = torch.optim.Adam(gen_params, lr=config.lr, betas=(0.0, 0.9))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=config.lr, betas=(0.0, 0.9))

    def _z(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.config.z_dim, device=self.device)

    def critic_update(self, real_x: torch.Tensor, ids: torch.Tensor) -> float:
        real_x = real_x.to(self.device)
        e = self.embedding(ids.to(self.device)).detach()
        with torch.no_grad():
            fake = self.generator(self._z(real_x.size(0)), e)
        real_score = self.critic(real_x, e).mean()
        fake_score = self.critic(fake, e).mean()
        gp = gradient_penalty(self.critic, real_x, fake, e)
        loss = fake_score - real_score + self.config.lambda_gp * gp
        self.opt_c.zero_grad()
        loss.backward()
        self.opt_c.step()
        return float(loss.item())

    def generator_update(self, ids: torch.Tensor) -> float:
        e = self.embedding(ids.to(self.device))
        fake = self.generator(self._z(ids.size(0)), e)
        loss = -self.critic(fake, e).mean()
        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()
        return float(loss.item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -q`
Expected: PASS (all train tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/train.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/train.py tests/gan_signals/test_train.py
git commit -m "$(cat <<'EOF'
feat(gan): WGANGP learner — critic_update/generator_update (embedding lado G)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `train()` orquestación (integración: run_meta, metrics, latencia, checkpoint)

**Files:**
- Modify: `src/cog_ew/gan_signals/train.py`
- Test: `tests/gan_signals/test_train.py`

**Interfaces:**
- Consumes: `WGANGP`, `WGANGPConfig`, `_set_seeds`, `_run_metadata`, `gradient_penalty`; `PDWSyntheticDataset`, `EmitterLibrary` (`cog_ew.data`); `profile_latency` (`cog_ew.temporal_cnn_elint.metrics`).
- Produces:
  - `train(config: WGANGPConfig) -> dict[str, Any]` — escribe `out_dir/{run_meta.json, metrics.json, best.pt}`; `metrics.json` con claves `wasserstein_estimate`, `gradient_penalty`, `diversity_std`, `critic_loss`, `gen_loss`, `latency_mean_ms`, `latency_p99_ms`. `best.pt` = dict `{"generator", "embedding", "critic"}` de `state_dict`.

- [ ] **Step 1: Write the failing test**

```python
import json


def _tiny_config(tmp_path) -> WGANGPConfig:
    config = WGANGPConfig.from_yaml("configs/gan_signals/wgan_gp.yaml")
    config.pdw.n_trains = 2
    config.z_dim, config.e_dim, config.channels = 8, 4, 8
    config.batch_size, config.total_steps, config.n_critic = 8, 2, 2
    config.out_dir = str(tmp_path / "run")
    return config


def test_train_writes_artifacts_and_checkpoint(tmp_path):
    config = _tiny_config(tmp_path)
    result = train(config)
    out = tmp_path / "run"
    assert (out / "run_meta.json").is_file()
    metrics = json.loads((out / "metrics.json").read_text())
    assert {
        "wasserstein_estimate",
        "gradient_penalty",
        "diversity_std",
        "latency_mean_ms",
        "latency_p99_ms",
    } <= set(metrics)
    assert "final" in result
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"generator", "embedding", "critic"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -k train_writes -q`
Expected: FAIL — `ImportError: cannot import name 'train'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/cog_ew/gan_signals/train.py` (extend imports as shown):

```python
from collections.abc import Iterator

from torch import nn
from torch.utils.data import DataLoader

from cog_ew.data.pdw_dataset import PDWSyntheticDataset
from cog_ew.data.pdw_library import EmitterLibrary
from cog_ew.temporal_cnn_elint.metrics import profile_latency


class _GeneratorForward(nn.Module):
    def __init__(self, generator: PDWGenerator, e: torch.Tensor) -> None:
        super().__init__()
        self.generator = generator
        self.e = e

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z, self.e)


def _cycle(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def train(config: WGANGPConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    library = EmitterLibrary.from_yaml(config.pdw.library_path)
    n_emitters = len(library.emitter_names())
    dataset = PDWSyntheticDataset(config.pdw)
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    learner = WGANGP(n_emitters, config, config.device)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))

    data_iter = _cycle(loader)
    critic_loss = 0.0
    gen_loss = 0.0
    for _ in range(config.total_steps):
        for _ in range(config.n_critic):
            real_x, ids, _, _ = next(data_iter)
            critic_loss = learner.critic_update(real_x, ids)
        _, ids, _, _ = next(data_iter)
        gen_loss = learner.generator_update(ids)

    real_x, ids, _, _ = next(data_iter)
    real_x = real_x.to(learner.device)
    e = learner.embedding(ids.to(learner.device)).detach()
    with torch.no_grad():
        fake = learner.generator(learner._z(real_x.size(0)), e)
        wasserstein = float(
            (learner.critic(real_x, e).mean() - learner.critic(fake, e).mean()).item()
        )
        diversity = float(fake.std(dim=0).mean().item())
    gp = float(gradient_penalty(learner.critic, real_x, fake, e).item())

    torch.save(
        {
            "generator": learner.generator.state_dict(),
            "embedding": learner.embedding.state_dict(),
            "critic": learner.critic.state_dict(),
        },
        out_dir / "best.pt",
    )

    e0 = learner.embedding(torch.zeros(1, dtype=torch.long, device=learner.device)).detach()
    sample = torch.zeros(1, config.z_dim, device=learner.device)
    mean_ms, p99_ms = profile_latency(
        _GeneratorForward(learner.generator, e0), sample, n_warmup=5, n_iter=50, device=config.device
    )

    final = {
        "wasserstein_estimate": wasserstein,
        "gradient_penalty": gp,
        "diversity_std": diversity,
        "critic_loss": critic_loss,
        "gen_loss": gen_loss,
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"final": final}
```

> Nota: `profile_latency` pone el modelo en `eval()`, lo que desactiva el BatchNorm del generador y el muestreo Gumbel sigue siendo válido; el perfilado mide la inferencia del generador con un `e` fijo (mismo patrón que `_AgentForward` en `marl_formation/train.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/gan_signals/test_train.py -q`
Expected: PASS (all train tests).

- [ ] **Step 5: Full suite + lint + type-check**

Run: `.venv/bin/ruff format src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/ruff check src/cog_ew/gan_signals/ tests/gan_signals/ && .venv/bin/mypy src/cog_ew/gan_signals/ && .venv/bin/python -m pytest -q`
Expected: all clean; full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/gan_signals/train.py tests/gan_signals/test_train.py
git commit -m "$(cat <<'EOF'
feat(gan): train() WGAN-GP — bucle, run_meta/metrics, latencia, checkpoint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Representación PDW `(B,10,64)` → Task 2 (generador), Task 3 (crítico). ✅
- Condicionamiento por embedding continuo + interpolación → Task 1. ✅
- 1D-CNN temporal G/D → Task 2/3. ✅
- Gumbel-softmax straight-through categóricas → Task 2 (test one-hot). ✅
- WGAN-GP (gradient penalty λ=10, n_critic, Adam β=(0,0.9)) → Task 4 (GP), Task 6 (updates), Task 5 (config). ✅
- Embedding lado generador, crítico con `e` detached → Task 6. ✅
- Config YAML, hiperparámetros no hardcodeados → Task 5. ✅
- Reproducibilidad (seeds, run_meta, config_hash, deps) → Task 5, Task 7. ✅
- metrics.json con Wasserstein estimate, GP, proxy mode-collapse (`diversity_std`), latencia → Task 7. ✅
- Checkpoint con `weights_only=True` → Task 7 (test). ✅
- Plan de tests TDD (10 comportamientos) → cubiertos en Tasks 1–7. ✅
- Fuera de alcance (export 200k, validez física, +22% Modelo 2) → no incluidos (sub-piezas B/C). ✅

**Placeholder scan:** sin TBD/TODO; todo el código está completo en cada step. ✅

**Type consistency:** `WGANGP` atributos `generator/critic/embedding/opt_g/opt_c`; `critic_update(real_x, ids)` / `generator_update(ids)`; `PDWGenerator.forward(z, e)` / `.sample(e)`; `PDWCritic.forward(x, e)`; `gradient_penalty(critic, real, fake, e)`; `WGANGPConfig.from_yaml`. Nombres y firmas consistentes entre tareas. ✅

---

## Execution Handoff

Tras guardar el plan, ofrecer al usuario la elección de ejecución (Subagent-Driven recomendado vs Inline).
