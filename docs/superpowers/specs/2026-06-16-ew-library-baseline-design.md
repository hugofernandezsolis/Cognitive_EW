# Diseño — Librería de respuestas EW pre-programadas (Modelo 5, baseline)

**Fecha:** 2026-06-16
**Ámbito:** `src/cog_ew/ew_library/library.py` + `configs/ew_library/responses.yaml` + `tests/ew_library/`
**Estado:** aprobado en brainstorming, pendiente de revisión del spec

## Contexto y alineación

Fuente de verdad: **`Propuesta.md`**. El Modelo 5 es la **librería de respuestas EW pre-programadas por tipo
de amenaza**, el **baseline rule-based** (no es ML) contra el que se mide la mejora de los Modelos 1–4. El
estado del arte (`docs/research/estado-del-arte.md` §5) recomienda un **selector determinista
amenaza→contramedida** que reproduzca la doctrina clásica, con una **librería de reglas explícita y
versionada**, usado como **fila de referencia** en las comparativas (el objetivo >92 % de victorias del
Modelo 1 se mide *contra* este baseline). La debilidad documentada del baseline (§5.2) — falla ante radares
LPI/multifunción y amenazas "zero-day", cobertura limitada — es **exactamente lo que debe perder** frente a
los modelos cognitivos; el diseño la conserva a propósito.

Decisiones tomadas en brainstorming:
- **Clave de decisión:** el par **`(tipo de emisor, modo)`**, con *fallback* por modo cuando el emisor no
  está catalogado (doctrina por sistema + degradación genérica).
- **Salida:** **combinación fija priorizada** de técnicas (lista ordenada), no una técnica única — fiel a la
  EW real y directamente comparable con el Modelo 1, que "genera la combinación óptima de técnicas".
- **Vocabulario de técnicas (`JammingTechnique`, Enum):** vive en `ew_library/`, no en la capa de datos,
  porque es un concepto de *respuesta* EW; el Modelo 1 lo importará como su espacio de acción.
- **Alcance:** solo selector + librería + Enum + manejo de "zero-day" + tests. Las métricas comparativas
  (win rate, J/S, burnthrough vs. cognitivo) requieren el entorno radar del Modelo 1 y se calculan en Fase 6.

## Representación y flujo de datos

```
(emisor, modo)  ──select()──►  tuple[JammingTechnique, ...]   (combinación priorizada)
```

`select` trabaja por **nombre** (la librería doctrinal es por sistema). La salida del Modelo 2 son índices
(`type_idx`, `mode_idx`); la conversión índice→nombre (vía `EmitterLibrary.emitter_names()` y `MODES`) es un
adaptador fino que vivirá en la integración de Fase 6 — **fuera de alcance aquí**.

Estructuras ya fijadas con las que encaja (de `src/cog_ew/data/pdw_library.py` y
`configs/temporal_cnn_elint/emitters.yaml`):

- **Emisores (8):** SA-2, SA-6, S-300, S-400, HQ-9, AESA, LPI-FMCW, LPI-polyphase.
- **Modos (4):** `search`, `tws`, `track`, `missile_guidance`.

## Componentes

### `library.py`

#### `JammingTechnique(Enum)`

Vocabulario compartido de contramedidas (de `Propuesta.md` Modelo 1 + EMSOPEDIA):
`NOISE, DRFM_REPEATER, DECEPTION, CROSS_EYE, VGPO, RGPO, CHAFF, DECOY, EVASIVE, NONE`.
Los valores son los nombres en minúscula usados en el YAML (`noise`, `drfm_repeater`, …) para parseo directo.

#### `EWResponseLibrary` (dataclass + `from_yaml`)

Espejo del patrón de `EmitterLibrary` (dataclass inmutable, `from_yaml`).

- `select(emitter: str, mode: str) -> tuple[JammingTechnique, ...]`: resolución determinista en 3 niveles:
  1. par `(emitter, mode)` catalogado → su combinación específica.
  2. emisor **no catalogado** pero modo válido → **default por modo** (respuesta genérica, más débil) —
     ilustra la "cobertura limitada" de §5.2.
  3. modo fuera de `MODES` → `ValueError` (error de programación; los 4 modos son fijos).
- `from_yaml(path)`: carga `responses.yaml`, construye el mapa de reglas y los defaults, y **valida** que cada
  técnica listada exista en `JammingTechnique` (técnica desconocida → `ValueError`). Reproducibilidad y
  detección temprana de erratas en la librería.

### `configs/ew_library/responses.yaml`

Librería de reglas explícita y versionada:

```yaml
version: 1
rules:
  <emisor>:
    <modo>: [<técnica>, ...]   # combinación priorizada
  ...
defaults:                       # emisor no catalogado, por modo
  search:           [...]
  tws:              [...]
  track:            [...]
  missile_guidance: [...]
```

Cubre los pares `(emisor, modo)` que existen en `emitters.yaml` (cada emisor con los modos que declara) más el
bloque `defaults` por modo. La respuesta a emisores LPI (LPI-FMCW, LPI-polyphase) es deliberadamente pobre
(p. ej. `[noise, evasive]`, sin DRFM/cross-eye sofisticados) para fijar el comportamiento de baseline que los
modelos cognitivos deben superar.

## Manejo de errores

- Técnica desconocida en el YAML → `ValueError` en `from_yaml`.
- Modo fuera de `MODES` en `select` → `ValueError`.
- Emisor no catalogado → **no es error**: cae a `defaults[modo]` (comportamiento esperado del baseline).

## Tests (`tests/ew_library/test_library.py`)

- `select` devuelve la combinación correcta **y en orden** para un `(emisor, modo)` catalogado.
- Emisor no catalogado con modo válido → cae al `default` del modo.
- Modo inválido → `ValueError`.
- `from_yaml` parsea y valida: una técnica desconocida en el YAML → `ValueError`.
- La respuesta a un emisor LPI es deliberadamente pobre (no incluye técnicas sofisticadas) — fija el
  comportamiento de baseline.
- `JammingTechnique` cubre exactamente el vocabulario acordado (10 miembros).

## Reproducibilidad

- Reglas solo en `responses.yaml` versionado; nada hardcodeado.
- Selección puramente determinista (sin estado, sin aleatoriedad).
- Validación de técnicas en carga → la librería no puede contener contramedidas inexistentes.

## Dependencias

Ninguna nueva (`PyYAML` ya está). No depende de PyTorch ni de la capa de datos en runtime; comparte
únicamente la nomenclatura de emisores/modos (strings).

## Fuera de alcance (YAGNI)

- Arnés de comparación / entorno radar y métricas win-rate, J/S, burnthrough (Fase 6 / Modelo 1).
- Parámetros de técnica (potencia J/S, nº de falsos blancos).
- Adaptador índice→nombre desde la salida del Modelo 2.
- Razonamiento probabilístico u optimización heurística (familias 2-3 de §5.1); el baseline del TFM es la
  familia 1 (rule-based), que es la canónica.

## Decisiones clave (resumen)

1. **Selector determinista `(emisor, modo)` → combinación priorizada**, fiel a la doctrina rule-based.
2. **Fallback por modo** para emisores no catalogados; modo inválido es error.
3. **`JammingTechnique` Enum** como vocabulario compartido, dueño en `ew_library/` (lo reusará el Modelo 1).
4. **Librería en YAML versionado** con validación de técnicas en carga.
5. **Baseline deliberadamente pobre ante LPI/zero-day** — es lo que los modelos cognitivos deben superar.
