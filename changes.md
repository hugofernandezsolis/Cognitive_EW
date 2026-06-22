# Changes

## 2026-06-22 - M2-v2 ELINT estricto

- Añadido `feature_set: v2` al dataset PDW para exponer señales temporales derivadas sin romper el flujo v1 usado por otros modelos, especialmente M4.
- Añadida `TemporalCNNV2`, con cabezales explícitos de tipo, modo y amenaza, porque la propuesta pide clasificar el estado de amenaza y no solo inferirlo indirectamente.
- Actualizado el entrenamiento de M2 para seleccionar arquitectura `v1`/`v2`, entrenar la cabeza de amenaza y emitir `macro_acc_threat`, `confusion_threat` y `strict_elint_score`.
- Actualizada el ancla ELINT para que ya no pueda pasar usando solo `lpi_accuracy`; ahora exige el peor valor entre type/mode/threat/LPI y además bloquea el pase si `latency_p99_ms >= 1 ms`.
- Añadidas configs `train_v2.yaml` y `train_v2_quick.yaml`: la primera para ejecución full orientada a propuesta, la segunda para mantener los tests y el perfil quick ágiles sin volver a M2-v1.
- Actualizados `configs/experiments/{quick,full}.yaml` para ejecutar M2-v2 en el arnés de anclas.
- Actualizado `docs/ROADMAP.md` para reflejar que M2 pasa a validarse con métrica estricta alineada con `Propuesta.md`.
