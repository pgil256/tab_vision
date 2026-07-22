"""Inharmonicity from string specifications, not from a fitted dataset.

The Q6 pilot's stiffness table was fitted from GuitarSet players' labelled
notes, which made the whole channel an artefact of one dataset's instruments
(`docs/EVAL_REPORTS/q6_self_calibration_2026-07-22.md`). ``B`` is not a
mystery quantity though — it follows from published string specifications:

    B = pi^3 * E * d_core^4 / (64 * T * L^2)          [stiff-string theory]
    T = 4 * mu * L^2 * f^2                            [ideal string tension]
    => B = pi^3 * E * d_core^4 / (256 * mu * L^4 * f^2)

with ``E`` the core's Young modulus, ``d_core`` the core wire diameter,
``mu`` the *total* linear mass density (core plus winding), ``L`` the scale
length and ``f`` the open-string frequency.

Two consequences worth stating.

**The fret law is derived, not assumed.** Fretting at ``n`` gives
``L_n = L*2^(-n/12)`` and ``f_n = f*2^(n/12)``, so
``B_n = B * 2^(4n/12) / 2^(2n/12) = B0 * 2^(n/6)`` — exactly the scaling
:mod:`tabvision.fusion.inharmonicity` already uses, for ideal fretting.

**Wound strings are far less inharmonic than plain ones at the same pitch**,
because the winding raises ``mu`` (and therefore tension) while contributing
nothing to ``d_core``. That is what makes the low strings separable from the
high ones rather than merely different in pitch.

The specification constants below are *typical* values for a light-gauge
phosphor-bronze acoustic set. Core diameters in particular are
manufacturer-specific and not always published, so they are approximations
and every field is overridable. The honest test of the table is agreement
with an independently fitted one — see
`scripts/eval/q6_physics_table.py`, which compares it against GuitarSet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tabvision.fusion.inharmonicity import StringStiffnessModel

STEEL_YOUNGS_MODULUS_PA = 2.0e11
"""Young's modulus of the steel core (~200 GPa). Wrap material is irrelevant:
it adds mass, not bending stiffness."""

DEFAULT_SCALE_LENGTH_IN = 25.4
"""Typical steel-string acoustic scale length (Martin long scale)."""

LB_PER_IN_TO_KG_PER_M = 17.858
"""Manufacturer unit weights are published in lb/in; SI is needed here."""

IN_TO_M = 0.0254


@dataclass(frozen=True)
class StringSpec:
    """One string's published specification.

    ``unit_weight_lb_per_in`` is the manufacturer's own figure (the number
    used for tension charts) and already includes the winding.
    ``core_diameter_in`` equals the gauge for a plain string and is the inner
    core wire for a wound one.
    """

    name: str
    gauge_in: float
    core_diameter_in: float
    unit_weight_lb_per_in: float
    wound: bool
    open_midi: int


# Typical light-gauge phosphor bronze (.012-.053), e.g. D'Addario EJ16 class.
# Unit weights follow published tension-chart values; wound-core diameters are
# approximations, which is precisely why the table is validated rather than
# trusted.
ACOUSTIC_LIGHT_SET: tuple[StringSpec, ...] = (
    StringSpec("E2", 0.053, 0.018, 0.00059427, True, 40),
    StringSpec("A2", 0.042, 0.016, 0.00037339, True, 45),
    StringSpec("D3", 0.032, 0.014, 0.00022110, True, 50),
    StringSpec("G3", 0.024, 0.012, 0.00012905, True, 55),
    StringSpec("B3", 0.016, 0.016, 0.00005732, False, 59),
    StringSpec("E4", 0.012, 0.012, 0.00003239, False, 64),
)


def open_frequency_hz(open_midi: int) -> float:
    return 440.0 * 2 ** ((open_midi - 69) / 12.0)


def inharmonicity_coefficient(
    spec: StringSpec,
    *,
    scale_length_in: float = DEFAULT_SCALE_LENGTH_IN,
    youngs_modulus_pa: float = STEEL_YOUNGS_MODULUS_PA,
) -> float:
    """Open-string ``B`` for one specification, in SI throughout."""
    length_m = scale_length_in * IN_TO_M
    core_m = spec.core_diameter_in * IN_TO_M
    mu_kg_per_m = spec.unit_weight_lb_per_in * LB_PER_IN_TO_KG_PER_M
    frequency = open_frequency_hz(spec.open_midi)
    numerator = math.pi**3 * youngs_modulus_pa * core_m**4
    denominator = 256.0 * mu_kg_per_m * length_m**4 * frequency**2
    return numerator / denominator


def reference_stiffness_model(
    specs: Sequence[StringSpec] = ACOUSTIC_LIGHT_SET,
    *,
    scale_length_in: float = DEFAULT_SCALE_LENGTH_IN,
    youngs_modulus_pa: float = STEEL_YOUNGS_MODULUS_PA,
) -> StringStiffnessModel:
    """A stiffness model computed from specifications alone.

    Requires no dataset, no labels and no user interaction, so it applies to
    any instrument whose string set and scale length are known — the default
    being a standard light acoustic set.
    """
    table = {
        index: math.log(
            inharmonicity_coefficient(
                spec, scale_length_in=scale_length_in, youngs_modulus_pa=youngs_modulus_pa
            )
        )
        for index, spec in enumerate(specs)
    }
    return StringStiffnessModel(log_b0=table)


__all__ = [
    "ACOUSTIC_LIGHT_SET",
    "DEFAULT_SCALE_LENGTH_IN",
    "StringSpec",
    "inharmonicity_coefficient",
    "open_frequency_hz",
    "reference_stiffness_model",
]
