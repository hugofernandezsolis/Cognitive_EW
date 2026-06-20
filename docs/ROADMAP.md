# Roadmap de desarrollo — Cognitive Electronic Warfare System

**Última actualización:** 2026-06-20

## 1. Propósito y relación con `Propuesta.md`

Este documento ordena **la ejecución** del proyecto: el estado de cada modelo, sus dependencias,
el orden de ataque y los hitos. Es el *cómo/cuándo*.

La **fuente de verdad** sigue siendo [`Propuesta.md`](../Propuesta.md) (el *qué*: los 5 modelos, el
dataset, la contribución y las revistas objetivo). Ante cualquier discrepancia, **manda `Propuesta.md`**,
que es de **solo lectura** y no se modifica bajo ningún concepto. El *cómo* operativo (stack, convenciones,
flujo de trabajo) vive en [`CLAUDE.md`](../CLAUDE.md).

La justificación técnica de cada modelo (huecos del estado del arte, casos de estudio, anclas cuantitativas)
está en [`docs/research/estado-del-arte.md`](research/estado-del-arte.md), referenciada por sección en cada
ficha.

## 2. Estado actual

| Pieza | Módulo | Estado |
|---|---|---|
| Capa de datos IQ (RadioML) | `src/cog_ew/data/` (`preprocessing`, `loaders`) | ✅ Completa, mergeada |
| Capa de datos PDW/ELINT | `src/cog_ew/data/` (`pdw_library`, `pdw_generator`, `pdw_dataset`) | ✅ Completa, mergeada |
| **Modelo 2** — Temporal CNN ELINT | `src/cog_ew/temporal_cnn_elint/` | ✅ Completo, mergeado |
| **Modelo 5** — Baseline EW convencional | `src/cog_ew/ew_library/` | ✅ Completo, mergeado |
| **Modelo 1** — Deep RL jamming | `src/cog_ew/deep_rl_jamming/` | ✅ Completo (A entorno + B agente/train + C comparación), mergeado |
| **Modelo 3** — MARL en formación | `src/cog_ew/marl_formation/` | 🔄 A entorno IADS ✅ mergeado; B agentes QMIX+train ✅ implementado en rama (pendiente review final + merge); C comparación 🔄 rehaciéndose vía spec→plan→TDD |
| **Modelo 4** — GAN señales sintéticas | `src/cog_ew/gan_signals/` | 📭 Stub |

> **Nota sobre la capa IQ (RadioML):** es **infraestructura compartida**, no específica de un modelo. La
> capa PDW/ELINT alimenta directamente al Modelo 2; la capa IQ queda disponible como representación de señal
> para los Modelos 1 (observaciones del agente RL) y 4 (entrada/salida de la GAN), pero su **consumidor final
> y su forma exacta de uso se fijarán en el ciclo `spec → plan` de cada uno** de esos modelos.

## 3. Fichas por modelo

Cada ficha resume objetivo (de `Propuesta.md`), métrica ancla (la cifra de la contribución que define el
"done publicable"), datos/entradas, dependencias y entregable. La columna de estado del arte apunta a la
sección de `estado-del-arte.md` con el hueco/oportunidad concreto.

### Modelo 1 — Deep RL para jamming adaptativo

- **Objetivo:** agente RL que observa la señal del radar amenaza (frecuencia, PRI, waveform, scan pattern,
  modo ECCM) y genera en tiempo real la combinación óptima de técnicas de jamming (noise, DRFM repeater
  deception, cross-eye, velocity/range gate pull-off), adaptándose cuando el radar cambia de modo o activa
  ECCM.
- **Métrica ancla:** supera al adversario en **>92 %** de enfrentamientos de espectro (vs. **58 %** con
  librería fija); adapta el jamming en **<4 ms** a cambios de waveform (objetivo de latencia de inferencia
  **<5 ms**).
- **Datos/entradas:** entorno simulado del ciclo radar (PRI, frecuencia, modo ECCM). Usa la representación
  PDW/IQ ya existente para describir las observaciones.
- **Dependencias:** el **Modelo 5** (baseline) como referencia de comparación del 92 % vs 58 %.
- **Estado del arte:** `estado-del-arte.md` §1 (Improved-SAC + Wolpertinger, D3QN anti-jamming; hueco §1.4).
- **Entregable:** `src/cog_ew/deep_rl_jamming/{env,agent,train}.py` + tests (env del ciclo radar, reward EW,
  latencia de inferencia).

### Modelo 2 — Temporal CNN para clasificación ELINT ✅

- **Objetivo:** CNN temporal que procesa la secuencia de pulsos del RWR y clasifica en tiempo real tipo de
  emisor, modo de operación y estado de amenaza, **incluyendo radares LPI**.
- **Métrica ancla:** accuracy **>96 %** incluyendo LPI (vs. **<65 %** convencional); latencia **<1 ms**
  (media + p99 en hardware fijo).
- **Estado:** completo y mergeado. Slice entrenable (modelo TCN dilatada, métricas, perfilado de latencia,
  bucle de entrenamiento, logging de reproducibilidad). Falta la **ejecución real** del entrenamiento en
  Colab (`/experiment-run`) para reportar las cifras finales.
- **Estado del arte:** `estado-del-arte.md` §2 (hueco §2.5: nadie reporta latencia; el SoA parte de ~15 ms).

### Modelo 3 — Multi-Agent RL para coordinación en formación

- **Objetivo:** MARL donde cada aeronave de una formación es un agente EW que coordina emisiones para
  maximizar la supresión del IADS: distribución de tareas, gestión de potencia (evitar fratricidio
  electrónico), engaño coordinado (blancos fantasma) y escolta electrónica.
- **Métrica ancla:** mejora la supresión del IADS un **45 %** coordinando **4 aeronaves** frente a actuación
  independiente.
- **Datos/entradas:** entorno multi-agente IADS, extensión del entorno del ciclo radar del Modelo 1 a varios
  emisores/agentes.
- **Dependencias:** el **Modelo 1** (reutiliza su entorno radar/jamming y sus primitivas de agente, escaladas
  a multi-agente).
- **Estado:** en curso en la rama `feat/marl-formation-qmix` (sin mergear todavía). Sub-pieza A (entorno IADS
  multi-agente) ✅ mergeada. Sub-pieza B (slice entrenable CTDE/QMIX: agentes, replay por episodios,
  entrenamiento y perfilado de latencia) ✅ implementada en rama, pendiente de review final + merge. Sub-pieza
  C (comparación coordinado vs. actuación independiente, ancla +45 %) 🔄 rehaciéndose por el ciclo
  brainstorm→spec→plan→TDD. La **ejecución real** en Colab (`/experiment-run`) para reportar el +45 % final
  llega en la Fase 6.
- **Estado del arte:** `estado-del-arte.md` §3 (MA-CJD jamming cooperativo, QMIX en enjambres; hueco §3.4).
- **Entregable:** `src/cog_ew/marl_formation/{env,agents,train,compare}.py` + tests (entorno IADS, reward de
  supresión coordinada).

### Modelo 4 — GAN para señales de amenaza sintéticas

- **Objetivo:** GAN que genera señales radar sintéticas realistas (AESA con waveform cognitiva, LPI, banda
  ancha, pasivo), proporcionando datos de entrenamiento ilimitados, incluyendo sistemas nunca catalogados.
- **Métrica ancla:** genera **>200 000** señales sintéticas con **50+** tipos de radar cognitivo futuro;
  mejora la robustez del clasificador (Modelo 2) un **+22 %** frente a sistemas no catalogados.
- **Datos/entradas:** representación PDW/IQ ya existente; las señales generadas se vuelcan en `data/synthetic/`.
- **Dependencias:** la representación de señal de la capa de datos; su **valor** se mide reevaluando el
  **Modelo 2** con datos aumentados (que ya está hecho).
- **Estado del arte:** `estado-del-arte.md` §4 (cWGAN-GP para formas de onda LPD; hueco §4.3).
- **Entregable:** `src/cog_ew/gan_signals/{generator,discriminator,train}.py` + tests (estabilidad del
  entrenamiento, validez de las señales generadas).

### Modelo 5 — Librería de respuestas EW pre-programadas (baseline)

- **Objetivo:** librería de contramedidas pre-programadas seleccionadas por tipo de amenaza identificada,
  como baseline del EW convencional actual.
- **Métrica ancla:** es **el baseline** contra el que se miden los Modelos 1 y 3 (el "58 %" / actuación fija).
- **Datos/entradas:** taxonomía de amenazas y sus respuestas asociadas (determinista, sin entrenamiento ML).
- **Dependencias:** ninguna; es la pieza más pequeña y no requiere entrenamiento.
- **Estado del arte:** `estado-del-arte.md` §5 (taxonomía convencional y por qué falla; ancla §5.3).
- **Entregable:** `src/cog_ew/ew_library/library.py` + tests (selección correcta de respuesta por amenaza).

## 4. Dependencias y orden de ataque

```
Capa de datos (IQ + PDW) ✅
        │
        ├── Modelo 2 (Temporal CNN) ✅
        │
        ├── Modelo 5 (baseline) ──────────┐
        │                                 │ (referencia de comparación)
        ├── Modelo 1 (RL jamming) ◄───────┘
        │        │
        │        └── Modelo 3 (MARL) ◄── extiende el entorno de M1
        │
        └── Modelo 4 (GAN) ── reevalúa robustez de M2
```

**Orden acordado: 5 → 1 → 3 → 4.**

1. **Modelo 5** primero: pieza pequeña, determinista, sin entrenamiento. Desbloquea las comparativas
   cuantitativas (el "vs. 58 %") que necesitan los Modelos 1 y 3.
2. **Modelo 1** después: contribución estrella; crea el entorno simulado del ciclo radar y las primitivas de
   jamming/agente.
3. **Modelo 3**: extiende el entorno del Modelo 1 de un agente a una formación multi-agente.
4. **Modelo 4** al final: aumenta los datos y robustece al clasificador (Modelo 2), que ya está completo, por
   lo que su valor se puede medir de inmediato sin bloquear a nadie.

## 5. Fases e hitos

| Fase | Contenido | Estado |
|---|---|---|
| **0** | Capa de datos (IQ RadioML + PDW/ELINT) | ✅ Hecho |
| **1** | Modelo 2 — Temporal CNN ELINT (slice entrenable) | ✅ Hecho |
| **2** | Modelo 5 — Baseline EW convencional | ✅ Hecho |
| **3** | Modelo 1 — Deep RL jamming (entorno + agente + entrenamiento) | ✅ Hecho (A entorno + B agente/train + C comparación) |
| **4** | Modelo 3 — MARL en formación (entorno + QMIX/train + comparación) | 🔄 En curso (A entorno ✅ mergeado; B QMIX/train ✅ en rama, pendiente review+merge; C comparación 🔄 rehaciéndose) |
| **5** | Modelo 4 — GAN señales sintéticas | 📭 Stub / pendiente de spec + plan + implementación |
| **6** | Evaluación transversal (anclas Q1) + ejecución real en Colab + redacción del paper | ⬜ Pendiente |

La **ejecución real de los entrenamientos** de los modelos ya implementados (Modelos 1, 2 y 3) se realiza en
la Fase 6 sobre Colab/Kaggle vía la skill `/experiment-run`, que es la que produce las cifras finales para el
paper. El Modelo 4 entra en esa fase cuando deje de ser stub y tenga su slice entrenable.

## 6. Flujo de trabajo por modelo

Cada modelo (Fases 2–5) sigue su propio ciclo independiente con superpowers, igual que se hizo con la capa de
datos y el Modelo 2:

1. `superpowers:brainstorming` → diseño + spec en `docs/superpowers/specs/`.
2. `superpowers:writing-plans` → plan TDD ejecutable en `docs/superpowers/plans/`.
3. `superpowers:subagent-driven-development` (o `test-driven-development` directo) → implementación tarea a
   tarea con revisión del diff.
4. `/code-review` + `security-reviewer` → calidad y verificación de que no se exponen datos sensibles.
5. `superpowers:verification-before-completion` → suite + ruff + format + mypy en verde.
6. `superpowers:finishing-a-development-branch` → merge a `main`.
7. `/experiment-run` (Fase 6) → entrenamiento real reproducible en Colab y registro de métricas.

## 7. Criterios de "done publicable" (Q1)

Un modelo está "terminado para publicar" cuando alcanza su ancla cuantitativa de `Propuesta.md`:

| Modelo | Umbral de aceptación |
|---|---|
| 1 — RL jamming | >92 % de victorias de espectro (vs 58 % baseline); adaptación <4 ms; inferencia <5 ms |
| 2 — Temporal CNN | accuracy >96 % incl. LPI (vs <65 %); latencia <1 ms (media + p99 reportadas) |
| 3 — MARL formación | +45 % de supresión del IADS con 4 aeronaves vs actuación independiente |
| 4 — GAN | >200 000 señales (50+ tipos); +22 % de robustez del clasificador frente a no catalogados |
| 5 — Baseline | sirve de referencia coherente para las comparativas de 1 y 3 |

**Revistas Q1 objetivo (JCR, CS-AI):** IEEE Transactions on Neural Networks and Learning Systems,
Knowledge-Based Systems, Information Sciences, Expert Systems with Applications.

## 8. Riesgos y restricciones transversales

- **VRAM limitada (GTX 1060, 6 GB):** estimar memoria con la skill `huggingface-skills:hf-mem` antes de
  entrenar cualquier modelo; el entrenamiento real va a Colab/Kaggle (L4/T4/A100).
- **Reproducibilidad (prioridad alta):** seeds explícitos, configs YAML versionadas, logging de seed +
  versiones + hash de datos + hiperparámetros completos; tracking vía `trackio`/W&B/MLflow.
- **Seguridad de datos EW:** no exponer en logs ni artefactos parámetros de amenazas reales ni datos
  sensibles. Los parámetros usados son sintéticos/publicados.
- **`Propuesta.md` es inmutable:** no editar, reescribir ni reformatear bajo ningún concepto.
- **Latencia como métrica de primera clase:** los Modelos 1 y 2 tienen objetivos de latencia duros; perfilar
  siempre media y p99 con batch=1.
