# Pipeline de datos PDW sintético (Modelo 2 ELINT) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un pipeline de datos PDW sintético etiquetado (tipo de emisor / modo / estado de amenaza) en `src/cog_ew/data/`, listo para alimentar un Temporal CNN multi-tarea.

**Architecture:** Una librería de emisores declarativa (YAML versionado) define la taxonomía y los rangos de parámetros por modo. Un generador sintético produce trenes de pulsos con realismo (PRI stagger/jitter, hopping de RF, amplitud por barrido, ruido, pulsos perdidos/espurios), deterministas por seed. Funciones puras de preprocesado convierten cada tren en ventanas de N pulsos channels-first (5 features continuas normalizadas + one-hot de modulación intra-pulso → 10 canales). Un `Dataset` ensambla todo en memoria y sirve `(Tensor[10,64], type, mode, threat)`.

**Tech Stack:** Python 3.11, NumPy 2.4, PyTorch 2.6, PyYAML, pytest. Sin dependencias nuevas.

---

## File Structure

- `src/cog_ew/data/pdw_library.py` — constantes (features, modos, modulaciones, rangos), `mode_to_threat`, dataclasses `ModeSpec`/`EmitterSpec`/`EmitterLibrary` con `from_yaml`.
- `src/cog_ew/data/pdw_generator.py` — `PulseTrain` + `_generate_pri` + `generate_pulse_train`.
- `src/cog_ew/data/pdw_dataset.py` — `PDWConfig` + `PDWSyntheticDataset`.
- `src/cog_ew/data/preprocessing.py` — añadir funciones puras PDW (`toa_to_pri`, `normalize_pdw`, `one_hot_intra_pulse`, `window_sequence`).
- `configs/temporal_cnn_elint/emitters.yaml` — librería de emisores.
- `tests/data/` — `test_pdw_library.py`, `test_pdw_preprocessing.py`, `test_pdw_generator.py`, `test_pdw_dataset.py`, fixture en `conftest.py`.

Convenciones (CLAUDE.md): type hints en API pública; NO comentarios de *qué*; `ruff format` + `ruff check` + `mypy --strict` limpios (correr en cada commit sobre los ficheros tocados, incluidos los de test). Usar `.venv/bin/<tool>`.

---

## Task 1: Constantes y `mode_to_threat`

**Files:**
- Create: `src/cog_ew/data/pdw_library.py`
- Test: `tests/data/test_pdw_library.py`

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/data/test_pdw_library.py`:

```python
import numpy as np

from cog_ew.data.pdw_library import (
    CONTINUOUS_FEATURES,
    CONTINUOUS_RANGES,
    INTRA_PULSE_MODS,
    MODES,
    THREAT_LEVELS,
    mode_to_threat,
)


def test_constant_shapes():
    assert CONTINUOUS_FEATURES == ("rf", "pw", "pa", "aoa", "pri")
    assert INTRA_PULSE_MODS == ("none", "lfm", "barker", "fmcw", "polyphase")
    assert MODES == ("search", "tws", "track", "missile_guidance")
    assert THREAT_LEVELS == ("low", "medium", "high", "critical")
    assert CONTINUOUS_RANGES.shape == (5, 2)


def test_mode_to_threat_mapping():
    assert mode_to_threat("search") == 0
    assert mode_to_threat("tws") == 1
    assert mode_to_threat("track") == 2
    assert mode_to_threat("missile_guidance") == 3
```

- [ ] **Step 2: Ejecutar para verificar que falla**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py -v`
Expected: FAIL con `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Implementación mínima**

Crear `src/cog_ew/data/pdw_library.py`:

```python
"""Taxonomía y parámetros de emisores para el pipeline PDW sintético (ELINT)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

CONTINUOUS_FEATURES: tuple[str, ...] = ("rf", "pw", "pa", "aoa", "pri")
INTRA_PULSE_MODS: tuple[str, ...] = ("none", "lfm", "barker", "fmcw", "polyphase")
MODES: tuple[str, ...] = ("search", "tws", "track", "missile_guidance")
THREAT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")

# Rangos físicos por feature continua (orden de CONTINUOUS_FEATURES):
# rf [GHz], pw [us], pa [lineal], aoa [deg], pri [us].
CONTINUOUS_RANGES: NDArray[np.float64] = np.array(
    [
        [0.5, 18.0],
        [0.05, 200.0],
        [0.0, 1.0],
        [-180.0, 180.0],
        [1.0, 10000.0],
    ],
    dtype=np.float64,
)

_MODE_TO_THREAT = {"search": 0, "tws": 1, "track": 2, "missile_guidance": 3}

LPI_MODS: frozenset[str] = frozenset({"fmcw", "polyphase"})


def mode_to_threat(mode: str) -> int:
    return _MODE_TO_THREAT[mode]
```

- [ ] **Step 4: Ejecutar para verificar que pasa**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py
.venv/bin/ruff check src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py
.venv/bin/mypy src/cog_ew/data/pdw_library.py
git add src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py
git commit -m "feat(data): constantes PDW y mode_to_threat

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Dataclasses de la librería + `from_yaml` + `emitters.yaml`

**Files:**
- Modify: `src/cog_ew/data/pdw_library.py`
- Create: `configs/temporal_cnn_elint/emitters.yaml`
- Test: `tests/data/test_pdw_library.py`

- [ ] **Step 1: Crear el fichero de configuración**

Crear `configs/temporal_cnn_elint/emitters.yaml` (taxonomía inicial; 8 emisores, incl. 2 LPI):

```yaml
emitters:
  - name: SA-2
    modes:
      search: {rf_band: [2.9, 3.1], pri_pattern: fixed, pri_range: [800, 1200], pw_range: [1.0, 3.0], scan_period: 10.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none]}
      track: {rf_band: [2.9, 3.1], pri_pattern: jitter, pri_range: [400, 600], pw_range: [0.8, 1.5], scan_period: 2.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none, lfm]}
      missile_guidance: {rf_band: [3.0, 3.2], pri_pattern: stagger, pri_range: [200, 400], pw_range: [0.5, 1.0], scan_period: 1.0, freq_hopping: false, lpi: false, intra_pulse_mods: [lfm]}
  - name: SA-6
    modes:
      search: {rf_band: [4.0, 4.4], pri_pattern: fixed, pri_range: [700, 1000], pw_range: [1.0, 2.5], scan_period: 8.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none]}
      track: {rf_band: [8.0, 9.0], pri_pattern: jitter, pri_range: [300, 500], pw_range: [0.4, 1.0], scan_period: 1.5, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm]}
      missile_guidance: {rf_band: [8.0, 9.0], pri_pattern: stagger, pri_range: [150, 300], pw_range: [0.3, 0.8], scan_period: 1.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm]}
  - name: S-300
    modes:
      search: {rf_band: [0.6, 1.0], pri_pattern: fixed, pri_range: [1000, 1500], pw_range: [5.0, 15.0], scan_period: 12.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none, lfm]}
      tws: {rf_band: [4.0, 6.0], pri_pattern: stagger, pri_range: [500, 800], pw_range: [2.0, 6.0], scan_period: 4.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm]}
      track: {rf_band: [8.0, 10.0], pri_pattern: jitter, pri_range: [250, 450], pw_range: [0.5, 2.0], scan_period: 1.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm, barker]}
      missile_guidance: {rf_band: [8.0, 10.0], pri_pattern: stagger, pri_range: [120, 260], pw_range: [0.3, 1.0], scan_period: 0.8, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
  - name: S-400
    modes:
      search: {rf_band: [0.6, 1.2], pri_pattern: fixed, pri_range: [900, 1400], pw_range: [5.0, 20.0], scan_period: 10.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none, lfm]}
      tws: {rf_band: [4.0, 8.0], pri_pattern: stagger, pri_range: [400, 700], pw_range: [1.5, 5.0], scan_period: 3.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm, barker]}
      track: {rf_band: [8.0, 12.0], pri_pattern: jitter, pri_range: [200, 400], pw_range: [0.4, 1.5], scan_period: 0.9, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
      missile_guidance: {rf_band: [8.0, 12.0], pri_pattern: stagger, pri_range: [100, 220], pw_range: [0.2, 0.8], scan_period: 0.7, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
  - name: HQ-9
    modes:
      search: {rf_band: [0.8, 1.4], pri_pattern: fixed, pri_range: [900, 1300], pw_range: [4.0, 12.0], scan_period: 9.0, freq_hopping: false, lpi: false, intra_pulse_mods: [none, lfm]}
      track: {rf_band: [8.0, 10.0], pri_pattern: jitter, pri_range: [220, 420], pw_range: [0.5, 1.8], scan_period: 1.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm, barker]}
      missile_guidance: {rf_band: [8.0, 10.0], pri_pattern: stagger, pri_range: [110, 240], pw_range: [0.3, 0.9], scan_period: 0.8, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
  - name: AESA
    modes:
      search: {rf_band: [8.0, 12.0], pri_pattern: stagger, pri_range: [300, 600], pw_range: [1.0, 5.0], scan_period: 2.0, freq_hopping: true, lpi: false, intra_pulse_mods: [lfm, barker]}
      tws: {rf_band: [8.0, 12.0], pri_pattern: stagger, pri_range: [200, 400], pw_range: [0.8, 3.0], scan_period: 1.0, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
      track: {rf_band: [8.0, 12.0], pri_pattern: jitter, pri_range: [120, 280], pw_range: [0.4, 1.2], scan_period: 0.5, freq_hopping: true, lpi: false, intra_pulse_mods: [barker]}
  - name: LPI-FMCW
    modes:
      search: {rf_band: [9.0, 10.0], pri_pattern: jitter, pri_range: [50, 150], pw_range: [50.0, 200.0], scan_period: 3.0, freq_hopping: true, lpi: true, intra_pulse_mods: [fmcw]}
      track: {rf_band: [9.0, 10.0], pri_pattern: jitter, pri_range: [30, 90], pw_range: [40.0, 150.0], scan_period: 1.0, freq_hopping: true, lpi: true, intra_pulse_mods: [fmcw]}
  - name: LPI-polyphase
    modes:
      search: {rf_band: [14.0, 16.0], pri_pattern: jitter, pri_range: [40, 120], pw_range: [40.0, 180.0], scan_period: 2.5, freq_hopping: true, lpi: true, intra_pulse_mods: [polyphase]}
      track: {rf_band: [14.0, 16.0], pri_pattern: jitter, pri_range: [25, 80], pw_range: [30.0, 120.0], scan_period: 0.9, freq_hopping: true, lpi: true, intra_pulse_mods: [polyphase]}
```

- [ ] **Step 2: Escribir los tests que fallan**

Añadir a `tests/data/test_pdw_library.py`:

```python
from pathlib import Path

from cog_ew.data.pdw_library import EmitterLibrary, ModeSpec

CONFIG = Path("configs/temporal_cnn_elint/emitters.yaml")


def test_library_from_yaml_loads_all_emitters():
    lib = EmitterLibrary.from_yaml(CONFIG)
    assert lib.emitter_names() == (
        "SA-2", "SA-6", "S-300", "S-400", "HQ-9", "AESA", "LPI-FMCW", "LPI-polyphase",
    )


def test_modespec_tuples_and_types():
    lib = EmitterLibrary.from_yaml(CONFIG)
    sa2 = lib.emitters[0]
    search = sa2.modes["search"]
    assert isinstance(search, ModeSpec)
    assert search.rf_band == (2.9, 3.1)
    assert search.intra_pulse_mods == ("none",)
    assert search.freq_hopping is False


def test_lpi_emitters_declare_lpi_modulation():
    lib = EmitterLibrary.from_yaml(CONFIG)
    for emitter in lib.emitters:
        for mode_spec in emitter.modes.values():
            if mode_spec.lpi:
                assert any(m in ("fmcw", "polyphase") for m in mode_spec.intra_pulse_mods)
```

- [ ] **Step 3: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py -v`
Expected: FAIL con `ImportError` para `EmitterLibrary`/`ModeSpec`.

- [ ] **Step 4: Implementación**

Añadir a `src/cog_ew/data/pdw_library.py` (imports `dataclass`, `Path`, `yaml` arriba):

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModeSpec:
    rf_band: tuple[float, float]
    pri_pattern: str
    pri_range: tuple[float, float]
    pw_range: tuple[float, float]
    scan_period: float
    freq_hopping: bool
    lpi: bool
    intra_pulse_mods: tuple[str, ...]


@dataclass(frozen=True)
class EmitterSpec:
    name: str
    modes: dict[str, ModeSpec]


@dataclass(frozen=True)
class EmitterLibrary:
    emitters: tuple[EmitterSpec, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmitterLibrary:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        emitters = []
        for entry in raw["emitters"]:
            modes = {
                mode_name: ModeSpec(
                    rf_band=tuple(spec["rf_band"]),
                    pri_pattern=spec["pri_pattern"],
                    pri_range=tuple(spec["pri_range"]),
                    pw_range=tuple(spec["pw_range"]),
                    scan_period=float(spec["scan_period"]),
                    freq_hopping=bool(spec["freq_hopping"]),
                    lpi=bool(spec["lpi"]),
                    intra_pulse_mods=tuple(spec["intra_pulse_mods"]),
                )
                for mode_name, spec in entry["modes"].items()
            }
            emitters.append(EmitterSpec(name=entry["name"], modes=modes))
        return cls(emitters=tuple(emitters))

    def emitter_names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.emitters)
```

> Si `mypy --strict` se queja de que `tuple(spec["rf_band"])` produce `tuple[Any, ...]` y no `tuple[float, float]`, envolver con `tuple(float(x) for x in spec["rf_band"])` y, si persiste, añadir un `# type: ignore[arg-type]` localizado y anotarlo. No cambiar las firmas públicas.

- [ ] **Step 5: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_pdw_library.py -v`
Expected: PASS (5 tests en total).

- [ ] **Step 6: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py
.venv/bin/ruff check src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py
.venv/bin/mypy src/cog_ew/data/pdw_library.py
git add src/cog_ew/data/pdw_library.py tests/data/test_pdw_library.py configs/temporal_cnn_elint/emitters.yaml
git commit -m "feat(data): librería de emisores PDW (dataclasses + from_yaml + config)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Funciones puras de preprocesado PDW

**Files:**
- Modify: `src/cog_ew/data/preprocessing.py`
- Test: `tests/data/test_pdw_preprocessing.py`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/data/test_pdw_preprocessing.py`:

```python
import numpy as np

from cog_ew.data.preprocessing import (
    normalize_pdw,
    one_hot_intra_pulse,
    toa_to_pri,
    window_sequence,
)


def test_toa_to_pri_diffs_with_padding():
    toa = np.array([0.0, 10.0, 25.0, 45.0])

    pri = toa_to_pri(toa)

    assert pri.shape == (4,)
    assert np.allclose(pri, [10.0, 10.0, 15.0, 20.0])  # pri[0] == pri[1]


def test_normalize_pdw_maps_ranges_to_unit():
    ranges = np.array([[0.0, 10.0], [0.0, 100.0]], dtype=np.float64)
    cont = np.array([[0.0, 50.0], [10.0, 100.0]], dtype=np.float64)

    out = normalize_pdw(cont, ranges)

    assert np.allclose(out, [[0.0, 0.5], [1.0, 1.0]])


def test_normalize_pdw_clips_out_of_range():
    ranges = np.array([[0.0, 10.0]], dtype=np.float64)
    cont = np.array([[-5.0], [15.0]], dtype=np.float64)

    out = normalize_pdw(cont, ranges)

    assert np.allclose(out, [[0.0], [1.0]])


def test_one_hot_intra_pulse():
    codes = np.array([0, 2, 4])

    out = one_hot_intra_pulse(codes, 5)

    assert out.shape == (3, 5)
    assert np.array_equal(out, np.eye(5)[codes])


def test_window_sequence_channels_first_and_discards_tail():
    seq = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)  # 7 pulsos, 2 features

    out = window_sequence(seq, 3)

    assert out.shape == (2, 2, 3)  # W=2, C=2, n=3 (descarta el 7º pulso)
    assert np.allclose(out[0, :, 0], seq[0])
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_pdw_preprocessing.py -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementación**

Añadir a `src/cog_ew/data/preprocessing.py` (ya tiene `import numpy as np` y `from numpy.typing import NDArray`):

```python
def toa_to_pri(toa: NDArray[np.floating]) -> NDArray[np.float32]:
    pri = np.empty_like(toa, dtype=np.float32)
    pri[1:] = np.diff(toa)
    pri[0] = pri[1] if pri.shape[0] > 1 else 0.0
    return pri


def normalize_pdw(
    cont: NDArray[np.floating], ranges: NDArray[np.floating]
) -> NDArray[np.float32]:
    lo = ranges[:, 0]
    span = ranges[:, 1] - ranges[:, 0]
    out = (cont - lo) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def one_hot_intra_pulse(codes: NDArray[np.integer], k: int) -> NDArray[np.float32]:
    return np.eye(k, dtype=np.float32)[codes]


def window_sequence(seq: NDArray[np.floating], n: int) -> NDArray[np.float32]:
    n_windows = seq.shape[0] // n
    trimmed = seq[: n_windows * n]
    reshaped = trimmed.reshape(n_windows, n, seq.shape[1])
    return np.transpose(reshaped, (0, 2, 1)).astype(np.float32)
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_pdw_preprocessing.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/preprocessing.py tests/data/test_pdw_preprocessing.py
.venv/bin/ruff check src/cog_ew/data/preprocessing.py tests/data/test_pdw_preprocessing.py
.venv/bin/mypy src/cog_ew/data/preprocessing.py
git add src/cog_ew/data/preprocessing.py tests/data/test_pdw_preprocessing.py
git commit -m "feat(data): preprocesado PDW (toa_to_pri, normalize_pdw, one_hot, window)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `PulseTrain` y `_generate_pri`

**Files:**
- Create: `src/cog_ew/data/pdw_generator.py`
- Test: `tests/data/test_pdw_generator.py`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/data/test_pdw_generator.py`:

```python
import numpy as np

from cog_ew.data.pdw_generator import _generate_pri


def test_generate_pri_fixed_is_constant():
    pri = _generate_pri("fixed", (100.0, 300.0), 10, np.random.default_rng(0))
    assert np.allclose(pri, 200.0)


def test_generate_pri_jitter_within_range():
    pri = _generate_pri("jitter", (100.0, 300.0), 1000, np.random.default_rng(0))
    assert pri.min() >= 100.0
    assert pri.max() <= 300.0


def test_generate_pri_stagger_uses_three_levels():
    pri = _generate_pri("stagger", (100.0, 300.0), 9, np.random.default_rng(0))
    assert set(np.round(np.unique(pri), 6)) == {100.0, 200.0, 300.0}


def test_generate_pri_unknown_pattern_raises():
    import pytest

    with pytest.raises(ValueError):
        _generate_pri("nope", (1.0, 2.0), 4, np.random.default_rng(0))
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_pdw_generator.py -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementación**

Crear `src/cog_ew/data/pdw_generator.py`:

```python
"""Generador sintético de trenes de pulsos PDW etiquetados (ELINT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cog_ew.data.pdw_library import INTRA_PULSE_MODS, ModeSpec


@dataclass
class PulseTrain:
    toa: NDArray[np.float64]
    rf: NDArray[np.float64]
    pw: NDArray[np.float64]
    pa: NDArray[np.float64]
    aoa: NDArray[np.float64]
    intra_pulse_mod: NDArray[np.int64]


def _generate_pri(
    pattern: str, pri_range: tuple[float, float], n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    lo, hi = pri_range
    if pattern == "fixed":
        return np.full(n, (lo + hi) / 2.0)
    if pattern == "jitter":
        return rng.uniform(lo, hi, n)
    if pattern == "stagger":
        levels = np.linspace(lo, hi, 3)
        return np.resize(levels, n).astype(np.float64)
    raise ValueError(f"Patrón de PRI desconocido: {pattern}")
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_pdw_generator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
.venv/bin/ruff check src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
.venv/bin/mypy src/cog_ew/data/pdw_generator.py
git add src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
git commit -m "feat(data): PulseTrain y _generate_pri (patrones de PRI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `generate_pulse_train` (realismo completo)

**Files:**
- Modify: `src/cog_ew/data/pdw_generator.py`
- Test: `tests/data/test_pdw_generator.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/data/test_pdw_generator.py`:

```python
from cog_ew.data.pdw_generator import generate_pulse_train
from cog_ew.data.pdw_library import ModeSpec

CLEAN_MODE = ModeSpec(
    rf_band=(9.0, 10.0),
    pri_pattern="fixed",
    pri_range=(100.0, 100.0),
    pw_range=(1.0, 2.0),
    scan_period=1.0,
    freq_hopping=False,
    lpi=False,
    intra_pulse_mods=("lfm", "barker"),
)


def test_generate_pulse_train_clean_count_and_shapes():
    train = generate_pulse_train(CLEAN_MODE, 50, np.random.default_rng(1))
    assert train.toa.shape == (50,)
    assert train.rf.shape == (50,)
    assert train.intra_pulse_mod.shape == (50,)


def test_generate_pulse_train_deterministic():
    a = generate_pulse_train(CLEAN_MODE, 30, np.random.default_rng(7))
    b = generate_pulse_train(CLEAN_MODE, 30, np.random.default_rng(7))
    assert np.array_equal(a.toa, b.toa)
    assert np.array_equal(a.intra_pulse_mod, b.intra_pulse_mod)


def test_generate_pulse_train_mods_within_allowed_set():
    train = generate_pulse_train(CLEAN_MODE, 200, np.random.default_rng(2))
    allowed = {INTRA_PULSE_MODS.index(m) for m in CLEAN_MODE.intra_pulse_mods}
    assert set(np.unique(train.intra_pulse_mod)).issubset(allowed)


def test_generate_pulse_train_drop_reduces_count():
    train = generate_pulse_train(CLEAN_MODE, 200, np.random.default_rng(3), drop_prob=1.0)
    assert train.toa.shape[0] == 0


def test_generate_pulse_train_spurious_increases_count():
    train = generate_pulse_train(
        CLEAN_MODE, 200, np.random.default_rng(4), spurious_prob=0.5
    )
    assert train.toa.shape[0] > 200
    assert np.all(np.diff(train.toa) >= 0)  # ordenado por TOA
```

(añadir `from cog_ew.data.pdw_generator import INTRA_PULSE_MODS` no es necesario; ya importado vía library en el módulo, pero en el test usa `INTRA_PULSE_MODS` — importarlo: `from cog_ew.data.pdw_library import INTRA_PULSE_MODS` al inicio del fichero de test.)

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_pdw_generator.py -v`
Expected: FAIL con `ImportError: cannot import name 'generate_pulse_train'`.

- [ ] **Step 3: Implementación**

Añadir a `src/cog_ew/data/pdw_generator.py`:

```python
def generate_pulse_train(
    mode_spec: ModeSpec,
    n_pulses: int,
    rng: np.random.Generator,
    *,
    noise_std: float = 0.0,
    drop_prob: float = 0.0,
    spurious_prob: float = 0.0,
) -> PulseTrain:
    pri = _generate_pri(mode_spec.pri_pattern, mode_spec.pri_range, n_pulses, rng)
    toa = np.cumsum(pri)

    band_lo, band_hi = mode_spec.rf_band
    if mode_spec.freq_hopping:
        rf = rng.uniform(band_lo, band_hi, n_pulses)
    else:
        rf = np.full(n_pulses, (band_lo + band_hi) / 2.0)

    pw = rng.uniform(mode_spec.pw_range[0], mode_spec.pw_range[1], n_pulses)

    scan_us = mode_spec.scan_period * 1e6
    pa = 0.5 * (1.0 + np.cos(2.0 * np.pi * (toa % scan_us) / scan_us))

    aoa = np.full(n_pulses, rng.uniform(-180.0, 180.0))

    mod_codes = np.array([INTRA_PULSE_MODS.index(m) for m in mode_spec.intra_pulse_mods])
    intra = rng.choice(mod_codes, n_pulses)

    if noise_std > 0.0:
        rf = rf + rng.normal(0.0, noise_std, n_pulses)
        pw = pw + rng.normal(0.0, noise_std, n_pulses)
        pa = np.clip(pa + rng.normal(0.0, noise_std, n_pulses), 0.0, 1.0)
        aoa = aoa + rng.normal(0.0, noise_std, n_pulses)

    if drop_prob > 0.0:
        keep = rng.random(n_pulses) >= drop_prob
        toa, rf, pw, pa, aoa, intra = (
            arr[keep] for arr in (toa, rf, pw, pa, aoa, intra)
        )

    if spurious_prob > 0.0 and toa.shape[0] > 0:
        n_sp = int(rng.binomial(n_pulses, spurious_prob))
        if n_sp > 0:
            sp_toa = rng.uniform(toa.min(), toa.max(), n_sp)
            sp_rf = rng.uniform(band_lo, band_hi, n_sp)
            sp_pw = rng.uniform(mode_spec.pw_range[0], mode_spec.pw_range[1], n_sp)
            sp_pa = rng.random(n_sp)
            sp_aoa = rng.uniform(-180.0, 180.0, n_sp)
            sp_intra = rng.choice(mod_codes, n_sp)
            toa = np.concatenate([toa, sp_toa])
            rf = np.concatenate([rf, sp_rf])
            pw = np.concatenate([pw, sp_pw])
            pa = np.concatenate([pa, sp_pa])
            aoa = np.concatenate([aoa, sp_aoa])
            intra = np.concatenate([intra, sp_intra])
            order = np.argsort(toa)
            toa, rf, pw, pa, aoa, intra = (
                arr[order] for arr in (toa, rf, pw, pa, aoa, intra)
            )

    return PulseTrain(
        toa=toa.astype(np.float64),
        rf=rf.astype(np.float64),
        pw=pw.astype(np.float64),
        pa=pa.astype(np.float64),
        aoa=aoa.astype(np.float64),
        intra_pulse_mod=intra.astype(np.int64),
    )
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_pdw_generator.py -v`
Expected: PASS (9 tests en total).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
.venv/bin/ruff check src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
.venv/bin/mypy src/cog_ew/data/pdw_generator.py
git add src/cog_ew/data/pdw_generator.py tests/data/test_pdw_generator.py
git commit -m "feat(data): generate_pulse_train con realismo (hopping, scan, ruido, drop, espurios)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `PDWConfig` y `PDWSyntheticDataset`

**Files:**
- Create: `src/cog_ew/data/pdw_dataset.py`
- Test: `tests/data/test_pdw_dataset.py`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/data/test_pdw_dataset.py`:

```python
import torch

from cog_ew.data.loaders import split_dataset
from cog_ew.data.pdw_dataset import PDWConfig, PDWSyntheticDataset

CONFIG_PATH = "configs/temporal_cnn_elint/emitters.yaml"


def _config(**kw):
    base = dict(
        library_path=CONFIG_PATH,
        emitters=("LPI-FMCW",),
        modes=("search",),
        window=64,
        n_pulses=128,
        n_trains=3,
        seed=0,
    )
    base.update(kw)
    return PDWConfig(**base)


def test_dataset_item_shape_and_labels():
    ds = PDWSyntheticDataset(_config())
    assert len(ds) == 3 * (128 // 64)  # n_trains * ventanas por tren

    pdw, type_idx, mode_idx, threat_idx = ds[0]
    assert isinstance(pdw, torch.Tensor)
    assert pdw.shape == (10, 64)
    assert pdw.dtype == torch.float32
    assert pdw.device.type == "cpu"
    assert type_idx == 6  # índice de LPI-FMCW en la librería
    assert mode_idx == 0  # search
    assert threat_idx == 0  # search -> low


def test_dataset_filters_modes():
    ds = PDWSyntheticDataset(_config(modes=("search", "track")))
    modes = {int(ds[i][2]) for i in range(len(ds))}
    assert modes == {0, 2}  # search=0, track=2


def test_dataset_split_deterministic():
    ds = PDWSyntheticDataset(_config(n_trains=8))
    a = split_dataset(ds, (0.5, 0.25, 0.25), seed=0)[0].indices
    b = split_dataset(ds, (0.5, 0.25, 0.25), seed=0)[0].indices
    assert list(a) == list(b)
```

- [ ] **Step 2: Ejecutar para verificar que fallan**

Run: `.venv/bin/pytest tests/data/test_pdw_dataset.py -v`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementación**

Crear `src/cog_ew/data/pdw_dataset.py`:

```python
"""Dataset sintético de secuencias PDW etiquetadas (tipo / modo / amenaza)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from cog_ew.data.pdw_generator import generate_pulse_train
from cog_ew.data.pdw_library import (
    CONTINUOUS_RANGES,
    INTRA_PULSE_MODS,
    MODES,
    EmitterLibrary,
    mode_to_threat,
)
from cog_ew.data.preprocessing import (
    normalize_pdw,
    one_hot_intra_pulse,
    toa_to_pri,
    window_sequence,
)


@dataclass
class PDWConfig:
    library_path: str
    emitters: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None
    window: int = 64
    n_pulses: int = 256
    n_trains: int = 8
    normalize: bool = True
    noise_std: float = 0.0
    drop_prob: float = 0.0
    spurious_prob: float = 0.0
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> PDWConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        for key in ("emitters", "modes"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)


class PDWSyntheticDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    def __init__(self, config: PDWConfig) -> None:
        library = EmitterLibrary.from_yaml(config.library_path)
        names = library.emitter_names()
        rng = np.random.default_rng(config.seed)

        windows: list[np.ndarray] = []
        types: list[int] = []
        modes: list[int] = []
        threats: list[int] = []

        for emitter in library.emitters:
            if config.emitters is not None and emitter.name not in config.emitters:
                continue
            type_idx = names.index(emitter.name)
            for mode_name, mode_spec in emitter.modes.items():
                if config.modes is not None and mode_name not in config.modes:
                    continue
                mode_idx = MODES.index(mode_name)
                threat_idx = mode_to_threat(mode_name)
                for _ in range(config.n_trains):
                    train = generate_pulse_train(
                        mode_spec,
                        config.n_pulses,
                        rng,
                        noise_std=config.noise_std,
                        drop_prob=config.drop_prob,
                        spurious_prob=config.spurious_prob,
                    )
                    if train.toa.shape[0] < config.window:
                        continue
                    pri = toa_to_pri(train.toa)
                    cont = np.stack(
                        [train.rf, train.pw, train.pa, train.aoa, pri], axis=1
                    )
                    if config.normalize:
                        cont = normalize_pdw(cont, CONTINUOUS_RANGES)
                    onehot = one_hot_intra_pulse(
                        train.intra_pulse_mod, len(INTRA_PULSE_MODS)
                    )
                    feat = np.concatenate([cont, onehot], axis=1)
                    for window in window_sequence(feat, config.window):
                        windows.append(window)
                        types.append(type_idx)
                        modes.append(mode_idx)
                        threats.append(threat_idx)

        self._x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        self._type = torch.tensor(types, dtype=torch.int64)
        self._mode = torch.tensor(modes, dtype=torch.int64)
        self._threat = torch.tensor(threats, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        return (
            self._x[index],
            int(self._type[index]),
            int(self._mode[index]),
            int(self._threat[index]),
        )
```

- [ ] **Step 4: Ejecutar para verificar que pasan**

Run: `.venv/bin/pytest tests/data/test_pdw_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + tipos + commit**

```bash
.venv/bin/ruff format src/cog_ew/data/pdw_dataset.py tests/data/test_pdw_dataset.py
.venv/bin/ruff check src/cog_ew/data/pdw_dataset.py tests/data/test_pdw_dataset.py
.venv/bin/mypy src/cog_ew/data/pdw_dataset.py
git add src/cog_ew/data/pdw_dataset.py tests/data/test_pdw_dataset.py
git commit -m "feat(data): PDWSyntheticDataset multi-etiqueta (tipo/modo/amenaza)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Verificación final

**Files:** ninguno nuevo (solo correcciones si hace falta)

- [ ] **Step 1: Suite completa de datos**

Run: `.venv/bin/pytest tests/data/ -v`
Expected: PASS (todos: IQ existentes + PDW nuevos).

- [ ] **Step 2: Lint + formato**

Run:
```bash
.venv/bin/ruff check src/cog_ew/data/ tests/data/
.venv/bin/ruff format --check src/cog_ew/data/ tests/data/
```
Expected: sin errores / "already formatted".

- [ ] **Step 3: Tipos**

Run: `.venv/bin/mypy src/cog_ew/data/`
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit de correcciones (si las hubo)**

```bash
git add -A
git commit -m "style(data): ruff format y correcciones de lint/tipos PDW

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
> Si no hubo cambios, omitir.

---

## Self-Review (cobertura del spec)

- **Taxonomía + librería YAML versionada** → Tasks 1-2. ✅
- **Generador sintético con realismo** (PRI patterns, hopping, scan, ruido, drop/espurios, seeded) → Tasks 4-5. ✅
- **Features 5 continuas + one-hot modulación intra-pulso → 10 canales × 64** → Tasks 3, 6. ✅
- **Preprocesado puro** (toa_to_pri, normalize por rangos físicos, one-hot, window) → Task 3. ✅
- **Dataset multi-etiqueta** `(Tensor[10,64], type, mode, threat)` CPU → Task 6. ✅
- **Amenaza derivada del modo** → Tasks 1, 6. ✅
- **Pipeline separado del de IQ; reutiliza `split_dataset`** → Task 6 (import de `loaders`). ✅
- **Validación LPI declara modulación LPI** → Task 2. ✅
- **Reproducibilidad** (seed, rangos físicos versionados, YAML) → Tasks 1-6. ✅
- **Sin dependencias nuevas** → no hay tarea de deps. ✅

Fuera de alcance (coherente con el spec): modelo Temporal CNN, Turing, deinterleaving multi-emisor, cadena IQ→PDW, augmentation, longitud variable.
