# Changes

## 2026-06-22 - Reporte Fase 6 aceptado

- Se acepta como referencia actual `runs (1)/runs/anchors_full/anchors_report.json`.
- M2-v2 queda reportado con accuracy estricta cumplida (`macro_acc_type=0.98445`, `macro_acc_mode=1.0`,
  `macro_acc_threat=1.0`, `lpi_accuracy=1.0`) y la reducción de `latency_p99_ms=1.3045` a `<1 ms` pasa a
  mejoras futuras.
- M4 queda reportado como aprobado con `relative_improvement=0.99528`, ya sin el problema anterior de
  `Infinity`.

## 2026-06-22 - M4 robustness score finito

- Sustituido el caso `relative_improvement = Infinity` de M4 cuando el baseline held-out es `0.0` por una métrica finita: ratio relativo si `baseline > 0`, ganancia absoluta si `baseline == 0`.
- Reforzado el test del ancla GAN para exigir `achieved` finito, evitando que `_passed()` rechace resultados semánticamente buenos por división entre cero.

## 2026-06-22 - M2-v2 ELINT estricto

- Añadido `feature_set: v2` al dataset PDW para exponer señales temporales derivadas sin romper el flujo v1 usado por otros modelos, especialmente M4.
- Añadida `TemporalCNNV2`, con cabezales explícitos de tipo, modo y amenaza, porque la propuesta pide clasificar el estado de amenaza y no solo inferirlo indirectamente.
- Actualizado el entrenamiento de M2 para seleccionar arquitectura `v1`/`v2`, entrenar la cabeza de amenaza y emitir `macro_acc_threat`, `confusion_threat` y `strict_elint_score`.
- Actualizada el ancla ELINT para que ya no pueda pasar usando solo `lpi_accuracy`; ahora exige el peor valor entre type/mode/threat/LPI y además bloquea el pase si `latency_p99_ms >= 1 ms`.
- Añadidas configs `train_v2.yaml` y `train_v2_quick.yaml`: la primera para ejecución full orientada a propuesta, la segunda para mantener los tests y el perfil quick ágiles sin volver a M2-v1.
- Actualizados `configs/experiments/{quick,full}.yaml` para ejecutar M2-v2 en el arnés de anclas.
- Actualizado `docs/ROADMAP.md` para reflejar que M2 pasa a validarse con métrica estricta alineada con `Propuesta.md`.
