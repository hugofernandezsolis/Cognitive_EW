# Librería de respuestas EW (Modelo 5, baseline) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar el baseline rule-based del Modelo 5: un selector determinista `(emisor, modo) → combinación priorizada de técnicas de jamming`, con una librería de reglas YAML versionada y un vocabulario de técnicas compartido.

**Architecture:** Un `Enum` `JammingTechnique` (vocabulario compartido que reutilizará el Modelo 1) y un dataclass inmutable `EWResponseLibrary` con `from_yaml` (carga + valida técnicas) y `select` (resolución determinista en 3 niveles: par catalogado → default por modo → `ValueError`). Reglas y defaults en `configs/ew_library/responses.yaml`. Sin estado, sin aleatoriedad, sin dependencia de PyTorch.

**Tech Stack:** Python 3.11, PyYAML, pytest, ruff, mypy. Herramientas vía `.venv/bin/<tool>`.

---

## File Structure

- Create: `src/cog_ew/ew_library/library.py` — `JammingTechnique` (Enum), `EWResponseLibrary` (dataclass + `from_yaml` + `select`), helper `_parse_techniques`.
- Create: `configs/ew_library/responses.yaml` — librería de reglas versionada (pares `(emisor, modo)` + `defaults` por modo).
- Test: `tests/ew_library/test_library.py` — vocabulario del Enum, parseo/validación de `from_yaml`, resolución de `select`.

El stub actual `src/cog_ew/ew_library/library.py` solo contiene el docstring de módulo; se conserva y se añade el código debajo.

---

## Task 1: `JammingTechnique` (vocabulario de técnicas)

**Files:**
- Modify: `src/cog_ew/ew_library/library.py`
- Test: `tests/ew_library/test_library.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/ew_library/test_library.py`:

```python
from cog_ew.ew_library.library import JammingTechnique


def test_jamming_technique_has_expected_vocabulary():
    values = {t.value for t in JammingTechnique}
    assert values == {
        "noise",
        "drfm_repeater",
        "deception",
        "cross_eye",
        "vgpo",
        "rgpo",
        "chaff",
        "decoy",
        "evasive",
        "none",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: FAIL con `ImportError` / `cannot import name 'JammingTechnique'`.

- [ ] **Step 3: Write minimal implementation**

En `src/cog_ew/ew_library/library.py` (mantener el docstring de módulo existente, añadir debajo):

```python
from __future__ import annotations

from enum import Enum


class JammingTechnique(Enum):
    NOISE = "noise"
    DRFM_REPEATER = "drfm_repeater"
    DECEPTION = "deception"
    CROSS_EYE = "cross_eye"
    VGPO = "vgpo"
    RGPO = "rgpo"
    CHAFF = "chaff"
    DECOY = "decoy"
    EVASIVE = "evasive"
    NONE = "none"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/ruff check src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/mypy src/cog_ew/ew_library/library.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
git commit -m "feat(ew-library): JammingTechnique (vocabulario de técnicas)"
```

---

## Task 2: `EWResponseLibrary.from_yaml` + `responses.yaml`

**Files:**
- Create: `configs/ew_library/responses.yaml`
- Modify: `src/cog_ew/ew_library/library.py`
- Test: `tests/ew_library/test_library.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/ew_library/test_library.py` (añadir `import pytest` y `EWResponseLibrary` al import existente):

```python
import pytest

from cog_ew.ew_library.library import EWResponseLibrary, JammingTechnique

LIB = "configs/ew_library/responses.yaml"


def test_from_yaml_loads_rules_and_defaults():
    lib = EWResponseLibrary.from_yaml(LIB)
    assert ("S-400", "missile_guidance") in lib.rules
    assert set(lib.defaults) == {"search", "tws", "track", "missile_guidance"}
    assert all(
        isinstance(t, JammingTechnique)
        for combo in lib.rules.values()
        for t in combo
    )


def test_from_yaml_rejects_unknown_technique(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "rules:\n"
        "  X:\n"
        "    search: [teleport]\n"
        "defaults:\n"
        "  search: [noise]\n"
        "  tws: [noise]\n"
        "  track: [noise]\n"
        "  missile_guidance: [noise]\n"
    )
    with pytest.raises(ValueError):
        EWResponseLibrary.from_yaml(bad)
```

(El primer import `from cog_ew.ew_library.library import JammingTechnique` del Task 1 queda subsumido por esta línea; deja una sola línea de import combinada.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: FAIL con `cannot import name 'EWResponseLibrary'`.

- [ ] **Step 3: Crear `configs/ew_library/responses.yaml`**

```yaml
version: 1
rules:
  SA-2:
    search: [noise]
    track: [vgpo, noise]
    missile_guidance: [rgpo, chaff]
  SA-6:
    search: [noise]
    track: [vgpo, drfm_repeater]
    missile_guidance: [rgpo, chaff, evasive]
  S-300:
    search: [noise]
    tws: [noise, deception]
    track: [vgpo, drfm_repeater]
    missile_guidance: [rgpo, drfm_repeater, chaff]
  S-400:
    search: [noise]
    tws: [noise, deception]
    track: [vgpo, drfm_repeater]
    missile_guidance: [rgpo, drfm_repeater, chaff, evasive]
  HQ-9:
    search: [noise]
    track: [vgpo, drfm_repeater]
    missile_guidance: [rgpo, drfm_repeater, chaff]
  AESA:
    search: [noise]
    tws: [noise, deception]
    track: [drfm_repeater, cross_eye]
  LPI-FMCW:
    search: [noise, evasive]
    track: [evasive]
  LPI-polyphase:
    search: [noise, evasive]
    track: [evasive]
defaults:
  search: [noise]
  tws: [noise, deception]
  track: [vgpo, chaff]
  missile_guidance: [rgpo, chaff, evasive]
```

- [ ] **Step 4: Write minimal implementation**

Añadir a `src/cog_ew/ew_library/library.py` (añadir los imports `dataclass`, `Path`, `yaml` arriba junto a los existentes; `MODES` se importa en el Task 3):

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


def _parse_techniques(names: list[str]) -> tuple[JammingTechnique, ...]:
    try:
        return tuple(JammingTechnique(name) for name in names)
    except ValueError as exc:
        raise ValueError(f"técnica desconocida en la librería EW: {exc}") from exc


@dataclass(frozen=True)
class EWResponseLibrary:
    rules: dict[tuple[str, str], tuple[JammingTechnique, ...]]
    defaults: dict[str, tuple[JammingTechnique, ...]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EWResponseLibrary:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        rules: dict[tuple[str, str], tuple[JammingTechnique, ...]] = {}
        for emitter, modes in raw["rules"].items():
            for mode, techniques in modes.items():
                rules[(emitter, mode)] = _parse_techniques(techniques)
        defaults = {
            mode: _parse_techniques(techniques)
            for mode, techniques in raw["defaults"].items()
        }
        return cls(rules=rules, defaults=defaults)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: PASS (los 3 tests).

- [ ] **Step 6: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/ruff check src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/mypy src/cog_ew/ew_library/library.py
```
Expected: sin errores.

- [ ] **Step 7: Commit**

```bash
git add src/cog_ew/ew_library/library.py configs/ew_library/responses.yaml tests/ew_library/test_library.py
git commit -m "feat(ew-library): EWResponseLibrary.from_yaml + responses.yaml versionada"
```

---

## Task 3: `EWResponseLibrary.select` (resolución en 3 niveles)

**Files:**
- Modify: `src/cog_ew/ew_library/library.py`
- Test: `tests/ew_library/test_library.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/ew_library/test_library.py`:

```python
def test_select_returns_ordered_combination_for_known_pair():
    lib = EWResponseLibrary.from_yaml(LIB)
    assert lib.select("S-400", "missile_guidance") == (
        JammingTechnique.RGPO,
        JammingTechnique.DRFM_REPEATER,
        JammingTechnique.CHAFF,
        JammingTechnique.EVASIVE,
    )


def test_select_falls_back_to_mode_default_for_uncatalogued_emitter():
    lib = EWResponseLibrary.from_yaml(LIB)
    assert lib.select("UNKNOWN-SAM", "track") == lib.defaults["track"]


def test_select_raises_on_invalid_mode():
    lib = EWResponseLibrary.from_yaml(LIB)
    with pytest.raises(ValueError):
        lib.select("S-400", "navigation")


def test_select_lpi_response_is_deliberately_poor():
    lib = EWResponseLibrary.from_yaml(LIB)
    techniques = lib.select("LPI-FMCW", "track")
    assert JammingTechnique.DRFM_REPEATER not in techniques
    assert JammingTechnique.CROSS_EYE not in techniques
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: FAIL con `AttributeError: 'EWResponseLibrary' object has no attribute 'select'`.

- [ ] **Step 3: Write minimal implementation**

Añadir el import de `MODES` arriba (junto a los demás imports):

```python
from cog_ew.data.pdw_library import MODES
```

Añadir el método dentro de la clase `EWResponseLibrary` (después de `from_yaml`):

```python
    def select(self, emitter: str, mode: str) -> tuple[JammingTechnique, ...]:
        if mode not in MODES:
            raise ValueError(f"modo desconocido: {mode!r}")
        if (emitter, mode) in self.rules:
            return self.rules[(emitter, mode)]
        return self.defaults[mode]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ew_library/test_library.py -v`
Expected: PASS (los 7 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/ruff check src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
.venv/bin/mypy src/cog_ew/ew_library/library.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/ew_library/library.py tests/ew_library/test_library.py
git commit -m "feat(ew-library): select con resolución en 3 niveles (catalogado/default/error)"
```

---

## Task 4: Verificación final (lint, tipos, suite completa)

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: todos los tests pasan (los anteriores + 7 nuevos de `ew_library`).

- [ ] **Step 2: Lint y formato de todo el árbol tocado**

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```
Expected: `All checks passed!` y todos los ficheros formateados.

- [ ] **Step 3: Tipos sobre `src/`**

Run: `.venv/bin/mypy src`
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit (si hubo algún ajuste de formato)**

```bash
git add -A
git commit -m "chore(ew-library): verificación final del baseline EW" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **`JammingTechnique` Enum (10 técnicas):** Task 1 + test de vocabulario exacto. ✅
- **`EWResponseLibrary` + `from_yaml` con validación de técnicas:** Task 2 (`_parse_techniques` re-lanza `ValueError`); test de técnica desconocida. ✅
- **`responses.yaml` versionado, cubre pares `(emisor, modo)` reales + `defaults`:** Task 2; cubre los 8 emisores con sus modos de `emitters.yaml` y los 4 defaults. ✅
- **`select` resolución en 3 niveles (catalogado → default por modo → `ValueError`):** Task 3 + 3 tests (par conocido, fallback, modo inválido). ✅
- **Baseline deliberadamente pobre ante LPI:** Task 3 test `test_select_lpi_response_is_deliberately_poor` (sin DRFM/cross-eye). ✅
- **Reproducibilidad (reglas solo en YAML, determinista):** sin estado ni RNG; reglas externas. ✅
- **Fuera de alcance respetado:** sin entorno/comparativa, sin parámetros de técnica, sin adaptador índice→nombre. ✅
- **Consistencia de tipos:** `select` y `defaults`/`rules` usan `tuple[JammingTechnique, ...]` en todo el plan; `_parse_techniques` los produce; `from_yaml` los almacena. ✅
