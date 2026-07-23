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

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from tabvision.errors import ConfigurationError
from tabvision.fusion.artifact_registry import load_artifact_manifest
from tabvision.fusion.inharmonicity import StringStiffnessModel
from tabvision.types import GuitarConfig, SessionConfig

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
    youngs_modulus_pa: float | None = None
    """Core Young modulus; ``None`` uses the caller default (steel). Nylon
    strings set it explicitly."""


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


NYLON_TREBLE_MODULUS_PA = 5.0e9
"""Drawn nylon monofilament, ~5 GPa. Two orders below steel (~200 GPa), which
is why a classical guitar is far less inharmonic than a steel-string."""

NYLON_DENSITY_KG_M3 = 1150.0
"""Polyamide density, used to derive treble linear mass from gauge alone."""

CLASSICAL_SCALE_LENGTH_IN = 25.6
"""650 mm, the standard classical scale."""


# Classical normal-tension set. The three trebles (G3, B3, E4) are plain nylon
# monofilament: their linear mass is computed from NYLON_DENSITY and gauge, so
# those rows are first-principles. The three basses (E2, A2, D3) are a nylon
# multifilament floss core under metal winding — the winding mass is essential
# and the *effective* bending core is ill-defined, so their core diameters and
# unit weights are DOCUMENTED APPROXIMATIONS (unit weight derived from typical
# normal tension via mu = T/(4 L^2 f^2); core ~0.55x gauge). Since B ~ d_core^4
# the bass predictions are rough by construction; the GAPS eval is the honest
# test of whether they are good enough. Repo string order is low-E (0) to
# high-E (5).
def _nylon_treble_mu_lb_per_in(gauge_in: float) -> float:
    radius_m = (gauge_in * IN_TO_M) / 2.0
    mu_kg_per_m = NYLON_DENSITY_KG_M3 * math.pi * radius_m**2
    return mu_kg_per_m / LB_PER_IN_TO_KG_PER_M


CLASSICAL_NYLON_SET: tuple[StringSpec, ...] = (
    StringSpec("E2", 0.043, 0.024, 3.354e-4, True, 40, NYLON_TREBLE_MODULUS_PA),
    StringSpec("A2", 0.035, 0.020, 1.747e-4, True, 45, NYLON_TREBLE_MODULUS_PA),
    StringSpec("D3", 0.029, 0.017, 8.460e-5, True, 50, NYLON_TREBLE_MODULUS_PA),
    StringSpec(
        "G3", 0.0403, 0.0403, _nylon_treble_mu_lb_per_in(0.0403), False, 55, NYLON_TREBLE_MODULUS_PA
    ),
    StringSpec(
        "B3", 0.0322, 0.0322, _nylon_treble_mu_lb_per_in(0.0322), False, 59, NYLON_TREBLE_MODULUS_PA
    ),
    StringSpec(
        "E4", 0.0280, 0.0280, _nylon_treble_mu_lb_per_in(0.0280), False, 64, NYLON_TREBLE_MODULUS_PA
    ),
)


def classical_stiffness_model() -> StringStiffnessModel:
    """Specification-derived stiffness for a classical (nylon) guitar."""
    table = {
        index: math.log(inharmonicity_coefficient(spec, scale_length_in=CLASSICAL_SCALE_LENGTH_IN))
        for index, spec in enumerate(CLASSICAL_NYLON_SET)
    }
    return StringStiffnessModel(log_b0=table)


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
    modulus = spec.youngs_modulus_pa if spec.youngs_modulus_pa is not None else youngs_modulus_pa
    numerator = math.pi**3 * modulus * core_m**4
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
    "CLASSICAL_NYLON_SET",
    "NYLON_YOUNGS_MODULUS_PA",
    "DEFAULT_SCALE_LENGTH_IN",
    "StringSpec",
    "inharmonicity_coefficient",
    "open_frequency_hz",
    "REGISTERED_TABLE",
    "StringEvidenceConfig",
    "load_string_evidence",
    "classical_stiffness_model",
    "reference_stiffness_model",
    "stiffness_model_for_session",
]


NYLON_YOUNGS_MODULUS_PA = 3.0e9
"""Polyamide trebles, ~3 GPa against steel's ~200 GPa.

Recorded to make the scale of the mismatch explicit rather than implied:
``B`` is linear in ``E``, so a nylon treble is roughly **65x less
inharmonic** than a steel string of the same geometry and tension. Classical
basses are a nylon multifilament core under metal winding, different again.
Scoring nylon against :data:`ACOUSTIC_LIGHT_SET` would not be approximate, it
would be wrong by more than the entire signal the channel measures
(candidates separate by only 1.6-1.8x).
"""


def stiffness_model_for_session(
    session: SessionConfig,
    cfg: GuitarConfig | None = None,
) -> StringStiffnessModel | None:
    """The stiffness table for this session, or ``None`` to abstain.

    The channel is only meaningful where the shipped table actually describes
    the strings on the instrument. That is steel-string acoustic in standard
    tuning at capo 0 — the domain :data:`ACOUSTIC_LIGHT_SET` was derived for.

    Everything else returns ``None`` and the channel contributes nothing:

    * **classical** — nylon, see :data:`NYLON_YOUNGS_MODULUS_PA`. The physics
      applies perfectly well to nylon, but it needs a nylon table and none
      exists here; the steel one would be wrong by far more than the effect.
    * **electric** — different string sets and scale lengths, and outside the
      v1 acoustic scope entirely.
    * **capo or non-standard tuning** — the speaking length and tension both
      move, so ``B0`` no longer describes the open string.

    Abstaining is free: the GAPS clean-12 classical no-regression check is
    then satisfied by construction rather than by measurement.
    """
    cfg = cfg or GuitarConfig()
    if session.tone != "clean" or cfg.capo != 0:
        return None
    if tuple(cfg.tuning_midi) != tuple(spec.open_midi for spec in ACOUSTIC_LIGHT_SET):
        return None
    if session.instrument == "acoustic":
        return reference_stiffness_model()
    if session.instrument == "classical":
        return classical_stiffness_model()
    return None


REGISTERED_TABLE = "acoustic-physics-v1"
"""The registered artifact carrying the gate-passed table and decode config."""


@dataclass(frozen=True)
class StringEvidenceConfig:
    """A registered stiffness table plus the decode settings it was gated at."""

    model: StringStiffnessModel
    weight: float
    min_r2: float
    sigma: float


def load_string_evidence(name: str = REGISTERED_TABLE) -> StringEvidenceConfig:
    """Load a registered ``string_evidence`` artifact, hash-verified.

    The decode settings travel with the table because they were frozen before
    the full-dev run; loading them from the artifact rather than from module
    defaults means a later change to the defaults cannot silently alter what
    a registered artifact does.
    """
    manifest = load_artifact_manifest(name, expected_kind="string_evidence")
    payload = json.loads(manifest.artifact_path.read_text(encoding="utf-8"))
    if payload.get("method") != "inharmonicity":
        raise ConfigurationError(
            f"string evidence {name!r} uses unsupported method {payload.get('method')!r}"
        )
    table = {int(k): float(v) for k, v in payload["log_b0"].items()}
    decode = payload["decode"]
    return StringEvidenceConfig(
        model=StringStiffnessModel(
            log_b0=table, fret_exponent=float(payload.get("fret_exponent", 1.0))
        ),
        weight=float(decode["weight"]),
        min_r2=float(decode["min_r2"]),
        sigma=float(decode["sigma"]),
    )
