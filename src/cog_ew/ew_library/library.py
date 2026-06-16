"""Librería de respuestas EW pre-programadas por tipo de amenaza (baseline convencional)."""

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
