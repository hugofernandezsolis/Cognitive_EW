# Capa de datos compartida (IQ / RML2018.01A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar la capa de datos compartida para IQ (RadioML 2018.01A) en `src/cog_ew/data/`: preprocessing puro en NumPy y un `Dataset` que carga un subconjunto filtrado a RAM desde un HDF5 resuelto vía Kaggle o ruta local.

**Architecture:** `preprocessing.py` expone funciones puras sobre arrays IQ (normalización de potencia, channels-first, conversión a complejo). `loaders.py` define `RadioMLConfig` (cargable desde YAML), resuelve la ruta del HDF5 (`kagglehub` por defecto, ruta explícita para Drive), y `RadioML2018Dataset` lee los bloques `(modulación, SNR)` solicitados como slices contiguos a RAM, aplica preprocessing vectorizado y sirve tensores CPU. `split_dataset` hace splits deterministas por seed.

**Tech Stack:** Python 3.11, PyTorch 2.6, NumPy 2.4, h5py, PyYAML, kagglehub, pytest.

---

## File Structure

- `src/cog_ew/data/preprocessing.py` — funciones puras sobre IQ (sin I/O).
- `src/cog_ew/data/loaders.py` — config, resolución de ruta, `Dataset`, split.
- `tests/data/conftest.py` — fixture HDF5 sintético con el layout contiguo de RML2018.
- `tests/data/test_preprocessing.py` — tests de las funciones puras.
- `tests/data/test_loaders.py` — tests de config, resolución de ruta, dataset y split.

Convenciones (de CLAUDE.md): type hints en API pública; sin comentarios de *qué*; `ruff` + `mypy --strict`; tests con pytest; tests de integración donde sea práctico (fixture HDF5 real en disco, sin mockear h5py).

---

## Task 1: Añadir dependencias

**Files:**
- Modify: `pyproject.toml` (gestionado por `uv`, no editar a mano)

- [ ] **Step 1: Añadir dependencias de runtime**

Run:
```bash
uv add h5py pyyaml kagglehub
```
Expected: `uv` resuelve e instala; `pyproject.toml` lista `h5py`, `pyyaml`, `kagglehub` en `[project.dependencies]`; `uv.lock` actualizado.

- [ ] **Step 2: Añadir stubs de tipos en el grupo dev**

Run:
```bash
uv add --dev types-pyyaml
```
Expected: `types-pyyaml` aparece en `[dependency-groups].dev`.

- [ ] **Step 3: Verificar imports**

Run:
```bash
.venv/bin/python -c "import h5py, yaml, kagglehub; print('deps OK')"
```
Expected: `deps OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: añadir h5py, pyyaml, kagglehub para la capa de datos IQ

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `preprocessing.py` — normalización de potencia

**Files:**
- Modify: `src/cog_ew/data/preprocessing.py`
- Test: `tests/data/test_preprocessing.py`

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/data/test_preprocessing.py`:

```python
import numpy as np

from cog_ew.data.preprocessing import normalize_power


def test_normalize_power_unit_mean_power_single_example():
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(128, 2)).astype(np.float32) * 5.0

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1))
    assert np.isclose(power, 1.0, atol=1e-5)


def test_normalize_power_batched():
    rng = np.random.default_rng(1)
    iq = rng.normal(size=(4, 128, 2)).astype(np.float32) * np.array([1.0, 2.0, 3.0, 4.0])[:, None, None]

    out = normalize_power(iq)

    power = np.mean(np.sum(out**2, axis=-1), axis=-1)
    assert np.allclose(power, 1.0, atol=1e-5)
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `.venv/bin/pytest tests/data/test_preprocessing.py -v`
Expected: FAIL con `ImportError: cannot import name 'normalize_power'`.

- [ ] **Step 3: Implementación mínima**

En `src/cog_ew/data/preprocessing.py` (mantener el docstring de módulo existente, añadir debajo):

```python
import numpy as np
from numpy.typing import NDArray


def normalize_power(iq: NDArray[np.floating]) -> NDArray[np.float32]:
    iq = iq.astype(np.float32, copy=False)
    power = np.mean(np.sum(iq**2, axis=-1), axis=-1)
    scale = np.sqrt(power)
    return (iq / scale[..., np.newaxis, np.newaxis]).astype(np.float32)
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `.venv/bin/pytest tests/data/test_preprocessing.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/preprocessing.py tests/data/test_preprocessing.py
git commit -m "feat(data): normalize_power para IQ (potencia media unitaria)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `preprocessing.py` — channels-first y conversión a complejo

**Files:**
- Modify: `src/cog_ew/data/preprocessing.py`
- Test: `tests/data/test_preprocessing.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_preprocessing.py`:

```python
from cog_ew.data.preprocessing import complex_to_iq, iq_to_complex, to_channels_first


def test_to_channels_first_single_example():
    iq = np.zeros((128, 2), dtype=np.float32)

    out = to_channels_first(iq)

    assert out.shape == (2, 128)


def test_to_channels_first_batched():
    iq = np.zeros((4, 128, 2), dtype=np.float32)

    out = to_channels_first(iq)

    assert out.shape == (4, 2, 128)


def test_iq_complex_roundtrip():
    rng = np.random.default_rng(2)
    iq = rng.normal(size=(128, 2)).astype(np.float32)

    out = complex_to_iq(iq_to_complex(iq))

    assert np.allclose(out, iq, atol=1e-6)
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_preprocessing.py -v`
Expected: FAIL con `ImportError` para `to_channels_first` / `iq_to_complex` / `complex_to_iq`.

- [ ] **Step 3: Implementación mínima**

Añadir a `src/cog_ew/data/preprocessing.py`:

```python
def to_channels_first(iq: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.swapaxes(iq, -1, -2)


def iq_to_complex(iq: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    return iq[..., 0] + 1j * iq[..., 1]


def complex_to_iq(z: NDArray[np.complexfloating]) -> NDArray[np.float32]:
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_preprocessing.py -v`
Expected: PASS (5 tests en total).

- [ ] **Step 5: Lint + tipos del módulo**

Run:
```bash
.venv/bin/ruff check src/cog_ew/data/preprocessing.py
.venv/bin/mypy src/cog_ew/data/preprocessing.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/data/preprocessing.py tests/data/test_preprocessing.py
git commit -m "feat(data): to_channels_first y conversion IQ<->complejo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `RadioMLConfig` y `MODULATIONS_2018`

**Files:**
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_loaders.py`

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/data/test_loaders.py`:

```python
from cog_ew.data.loaders import MODULATIONS_2018, RadioMLConfig


def test_modulations_count():
    assert len(MODULATIONS_2018) == 24
    assert len(set(MODULATIONS_2018)) == 24


def test_config_defaults():
    config = RadioMLConfig()
    assert config.kaggle_dataset == "pinxau1000/radioml2018"
    assert config.h5_path is None
    assert config.normalize is True
    assert config.seed == 0


def test_config_from_yaml(tmp_path):
    yaml_text = (
        "h5_path: /data/foo.h5\n"
        "snr_range: [0, 18]\n"
        "modulations: [BPSK, QPSK]\n"
        "normalize: false\n"
        "seed: 7\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    config = RadioMLConfig.from_yaml(path)

    assert config.h5_path == "/data/foo.h5"
    assert config.snr_range == (0, 18)
    assert config.modulations == ("BPSK", "QPSK")
    assert config.normalize is False
    assert config.seed == 7
```

- [ ] **Step 2: Ejecutar para verificar que falla**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: FAIL con `ImportError: cannot import name 'MODULATIONS_2018'`.

- [ ] **Step 3: Implementación mínima**

En `src/cog_ew/data/loaders.py` (mantener docstring de módulo, añadir debajo):

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MODULATIONS_2018: tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
)


@dataclass
class RadioMLConfig:
    h5_path: str | None = None
    kaggle_dataset: str | None = "pinxau1000/radioml2018"
    snr_range: tuple[int, int] | None = None
    modulations: tuple[str, ...] | None = None
    normalize: bool = True
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RadioMLConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if "snr_range" in raw and raw["snr_range"] is not None:
            raw["snr_range"] = tuple(raw["snr_range"])
        if "modulations" in raw and raw["modulations"] is not None:
            raw["modulations"] = tuple(raw["modulations"])
        return cls(**raw)
```

> Nota sobre `MODULATIONS_2018`: el índice de clase proviene de `argmax(Y)` del HDF5 (estable). El orden de los nombres se valida contra la documentación del fichero real al cargar datos reales; si difiere, corregir esta tupla (no afecta a la lógica, solo al nombre mostrado).

- [ ] **Step 4: Ejecutar para verificar que pasa**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): RadioMLConfig y MODULATIONS_2018

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `resolve_h5_path`

**Files:**
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_loaders.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_loaders.py`:

```python
import pytest

from cog_ew.data.loaders import resolve_h5_path


def test_resolve_h5_path_explicit_existing(tmp_path):
    h5 = tmp_path / "data.h5"
    h5.write_bytes(b"")
    config = RadioMLConfig(h5_path=str(h5))

    assert resolve_h5_path(config) == h5


def test_resolve_h5_path_explicit_missing():
    config = RadioMLConfig(h5_path="/nope/missing.h5")

    with pytest.raises(FileNotFoundError):
        resolve_h5_path(config)


def test_resolve_h5_path_none_set():
    config = RadioMLConfig(h5_path=None, kaggle_dataset=None)

    with pytest.raises(FileNotFoundError):
        resolve_h5_path(config)


def test_resolve_h5_path_kaggle(tmp_path, monkeypatch):
    download_dir = tmp_path / "kaggle_cache"
    download_dir.mkdir()
    (download_dir / "GOLD_XYZ_OSC.0001_1024x2M.h5").write_bytes(b"")

    import cog_ew.data.loaders as loaders

    def fake_download(dataset: str) -> str:
        assert dataset == "pinxau1000/radioml2018"
        return str(download_dir)

    monkeypatch.setattr(loaders.kagglehub, "dataset_download", fake_download)
    config = RadioMLConfig(h5_path=None)

    assert resolve_h5_path(config).name == "GOLD_XYZ_OSC.0001_1024x2M.h5"
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: FAIL con `ImportError: cannot import name 'resolve_h5_path'`.

- [ ] **Step 3: Implementación mínima**

Añadir el import y la función en `src/cog_ew/data/loaders.py` (poner `import kagglehub` junto a los demás imports top-level):

```python
import kagglehub
```

```python
def resolve_h5_path(config: RadioMLConfig) -> Path:
    if config.h5_path is not None:
        path = Path(config.h5_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"h5_path no existe: {path}")
    if config.kaggle_dataset is not None:
        download_dir = Path(kagglehub.dataset_download(config.kaggle_dataset))
        candidates = sorted(download_dir.rglob("*.h5"))
        if not candidates:
            raise FileNotFoundError(
                f"No se encontró ningún .h5 en la descarga de {config.kaggle_dataset}"
            )
        return candidates[0]
    raise FileNotFoundError("Define h5_path o kaggle_dataset en RadioMLConfig")
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: PASS (7 tests en total).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): resolve_h5_path (kaggle o ruta explícita)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `_mask_to_runs` (mask → slices contiguos)

**Files:**
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_loaders.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_loaders.py`:

```python
import numpy as np

from cog_ew.data.loaders import _mask_to_runs


def test_mask_to_runs_basic():
    mask = np.array([False, True, True, False, True])

    assert _mask_to_runs(mask) == [(1, 3), (4, 5)]


def test_mask_to_runs_empty():
    mask = np.zeros(5, dtype=bool)

    assert _mask_to_runs(mask) == []


def test_mask_to_runs_all_true():
    mask = np.ones(3, dtype=bool)

    assert _mask_to_runs(mask) == [(0, 3)]
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: FAIL con `ImportError: cannot import name '_mask_to_runs'`.

- [ ] **Step 3: Implementación mínima**

Añadir a `src/cog_ew/data/loaders.py` (necesita `import numpy as np` top-level si no está ya):

```python
def _mask_to_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev + 1))
        start = prev = current
    runs.append((start, prev + 1))
    return runs
```

Añadir el import de tipos arriba si falta:

```python
from numpy.typing import NDArray
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: PASS (10 tests en total).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): _mask_to_runs para lectura por slices contiguos

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Fixture HDF5 sintético

**Files:**
- Create: `tests/data/conftest.py`

- [ ] **Step 1: Crear el fixture**

Crear `tests/data/conftest.py`. Construye un HDF5 con el layout contiguo de RML2018
(modulación → SNR → frames), usando las 3 primeras modulaciones de `MODULATIONS_2018`,
3 SNRs `[-4, 0, 4]`, 4 frames por bloque y `N=8` muestras (en lugar de 1024).

```python
import h5py
import numpy as np
import pytest

from cog_ew.data.loaders import MODULATIONS_2018

N_MODS = 3
SNRS = (-4, 0, 4)
FRAMES = 4
N_SAMPLES = 8
N_CLASSES = len(MODULATIONS_2018)


@pytest.fixture
def synthetic_h5(tmp_path):
    rows = N_MODS * len(SNRS) * FRAMES
    x = np.zeros((rows, N_SAMPLES, 2), dtype=np.float32)
    y = np.zeros((rows, N_CLASSES), dtype=np.float32)
    z = np.zeros((rows, 1), dtype=np.int64)

    row = 0
    for mod_idx in range(N_MODS):
        for snr in SNRS:
            for _ in range(FRAMES):
                x[row] = float(row) + 1.0
                y[row, mod_idx] = 1.0
                z[row, 0] = snr
                row += 1

    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=x)
        fh.create_dataset("Y", data=y)
        fh.create_dataset("Z", data=z)
    return path
```

- [ ] **Step 2: Verificar que pytest reconoce el fixture (recopilación)**

Run: `.venv/bin/pytest tests/data/ --collect-only -q`
Expected: recopila los tests existentes sin errores de import en `conftest.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/data/conftest.py
git commit -m "test(data): fixture HDF5 sintético con layout RML2018

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `RadioML2018Dataset`

**Files:**
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_loaders.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_loaders.py` (usa el fixture `synthetic_h5` del conftest):

```python
import torch

from cog_ew.data.loaders import MODULATIONS_2018, RadioML2018Dataset


def test_dataset_loads_full(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 3 * 3 * 4  # mods * snrs * frames

    iq, label, snr = dataset[0]
    assert isinstance(iq, torch.Tensor)
    assert iq.shape == (2, 8)
    assert iq.dtype == torch.float32
    assert iq.device.type == "cpu"
    assert label == 0
    assert snr == -4


def test_dataset_filters_snr(synthetic_h5):
    config = RadioMLConfig(
        h5_path=str(synthetic_h5), kaggle_dataset=None, snr_range=(0, 4), normalize=False
    )
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 3 * 2 * 4  # solo SNR 0 y 4
    snrs = {int(dataset[i][2]) for i in range(len(dataset))}
    assert snrs == {0, 4}


def test_dataset_filters_modulations(synthetic_h5):
    keep = (MODULATIONS_2018[0], MODULATIONS_2018[2])
    config = RadioMLConfig(
        h5_path=str(synthetic_h5), kaggle_dataset=None, modulations=keep, normalize=False
    )
    dataset = RadioML2018Dataset(config)

    assert len(dataset) == 2 * 3 * 4  # 2 mods * 3 snrs * 4 frames
    labels = {int(dataset[i][1]) for i in range(len(dataset))}
    assert labels == {0, 2}


def test_dataset_normalizes(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=True)
    dataset = RadioML2018Dataset(config)

    iq, _, _ = dataset[5]
    power = float((iq**2).sum(dim=0).mean())
    assert abs(power - 1.0) < 1e-4
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: FAIL con `ImportError: cannot import name 'RadioML2018Dataset'`.

- [ ] **Step 3: Implementación mínima**

Añadir a `src/cog_ew/data/loaders.py` (necesita `import h5py`, `import torch` y los imports de `preprocessing`, todos top-level):

```python
import h5py
import torch
from torch.utils.data import Dataset

from cog_ew.data.preprocessing import normalize_power, to_channels_first


class RadioML2018Dataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, config: RadioMLConfig) -> None:
        path = resolve_h5_path(config)
        with h5py.File(path, "r") as fh:
            labels = np.asarray(fh["Y"]).argmax(axis=1)
            snr = np.asarray(fh["Z"])[:, 0].astype(np.int64)
            mask = np.ones(labels.shape[0], dtype=bool)
            if config.snr_range is not None:
                low, high = config.snr_range
                mask &= (snr >= low) & (snr <= high)
            if config.modulations is not None:
                keep_idx = {MODULATIONS_2018.index(name) for name in config.modulations}
                mask &= np.isin(labels, list(keep_idx))
            runs = _mask_to_runs(mask)
            chunks = [np.asarray(fh["X"][start:stop]) for start, stop in runs]

        x = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, 0, 2), dtype=np.float32)
        ).astype(np.float32)
        if config.normalize and x.shape[0] > 0:
            x = normalize_power(x)
        x = to_channels_first(x)

        self._x = torch.from_numpy(np.ascontiguousarray(x))
        self._labels = torch.from_numpy(labels[mask].astype(np.int64))
        self._snr = torch.from_numpy(snr[mask])

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self._x[index], int(self._labels[index]), int(self._snr[index])
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: PASS (14 tests en total).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): RadioML2018Dataset (carga eager de subconjunto filtrado)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: `split_dataset`

**Files:**
- Modify: `src/cog_ew/data/loaders.py`
- Test: `tests/data/test_loaders.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_loaders.py`:

```python
from cog_ew.data.loaders import split_dataset


def test_split_dataset_sizes(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)  # 36 ejemplos

    train, val, test = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)

    assert len(train) + len(val) + len(test) == len(dataset)
    assert len(train) == 18
    assert len(val) == 9
    assert len(test) == 9


def test_split_dataset_deterministic(synthetic_h5):
    config = RadioMLConfig(h5_path=str(synthetic_h5), kaggle_dataset=None, normalize=False)
    dataset = RadioML2018Dataset(config)

    a = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)[0].indices
    b = split_dataset(dataset, (0.5, 0.25, 0.25), seed=0)[0].indices
    c = split_dataset(dataset, (0.5, 0.25, 0.25), seed=1)[0].indices

    assert list(a) == list(b)
    assert list(a) != list(c)
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: FAIL con `ImportError: cannot import name 'split_dataset'`.

- [ ] **Step 3: Implementación mínima**

Añadir a `src/cog_ew/data/loaders.py` (necesita `from collections.abc import Sequence` y `from torch.utils.data import Subset, random_split`):

```python
from collections.abc import Sequence

from torch.utils.data import Subset, random_split


def split_dataset(
    dataset: Dataset[tuple[torch.Tensor, int, int]],
    fractions: Sequence[float],
    seed: int,
) -> list[Subset[tuple[torch.Tensor, int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(fractions), generator=generator)
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_loaders.py -v`
Expected: PASS (16 tests en total).

- [ ] **Step 5: Commit**

```bash
git add src/cog_ew/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): split_dataset determinista por seed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Verificación final (lint, tipos, suite completa)

**Files:** ninguno nuevo (solo correcciones si hace falta)

- [ ] **Step 1: Suite completa de la capa de datos**

Run: `.venv/bin/pytest tests/data/ -v`
Expected: PASS (16 tests).

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check src/cog_ew/data/ tests/data/`
Expected: sin errores (si los hay, `.venv/bin/ruff check --fix` y revisar).

- [ ] **Step 3: Formato**

Run: `.venv/bin/ruff format src/cog_ew/data/ tests/data/`
Expected: ficheros formateados (o "left unchanged").

- [ ] **Step 4: Tipos**

Run: `.venv/bin/mypy src/cog_ew/data/`
Expected: `Success: no issues found`. Si `mypy` se queja de `h5py`/`kagglehub` sin stubs, confirmar que `[tool.mypy].ignore_missing_imports = true` ya lo cubre (lo está en `pyproject.toml`).

- [ ] **Step 5: Commit de correcciones (si las hubo)**

```bash
git add -A
git commit -m "style(data): ruff format y correcciones de lint/tipos

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Si no hubo cambios, omitir este commit.

---

## Self-Review (cobertura del spec)

- **preprocessing.normalize_power / to_channels_first / iq_to_complex / complex_to_iq** → Tasks 2-3. ✅
- **RadioMLConfig + from_yaml** → Task 4. ✅
- **MODULATIONS_2018** → Task 4. ✅
- **resolve_h5_path (kaggle por defecto, ruta explícita, errores)** → Task 5. ✅
- **Lectura por slices contiguos (eager → RAM)** → Tasks 6 (`_mask_to_runs`) + 8 (`RadioML2018Dataset`). ✅
- **`__getitem__` → (Tensor[2,N], int, int) en CPU** → Task 8. ✅
- **split_dataset determinista** → Task 9. ✅
- **Tests con fixture HDF5 sintético, sin tocar los 20GB** → Task 7 + tests de 8/9. ✅
- **Deps h5py/pyyaml/kagglehub + types-pyyaml** → Task 1. ✅
- **Reproducibilidad (seed, config YAML, sin hardcodear)** → Tasks 4, 9. ✅

Fuera de alcance (coherente con el spec): PDW, lazy-mode completo, preprocesado-a-disco, augmentation, loaders GAN, device dentro del Dataset.
