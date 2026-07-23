"""Tests for the specification-derived stiffness table and ritual fit.

The physics table replaces a dataset-fitted one, so its scaling laws are the
thing to pin down: get an exponent wrong and the table stays plausible while
silently mis-ordering strings. The ritual fit is tested for the property that
motivated it — recovering the fret exponent rather than assuming it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabvision.fusion.inharmonicity import (
    LOG2,
    StiffnessObservation,
    attach_inharmonicity_evidence,
    calibrate_from_ritual,
)
from tabvision.fusion.string_physics import (
    ACOUSTIC_LIGHT_SET,
    DEFAULT_SCALE_LENGTH_IN,
    StringSpec,
    inharmonicity_coefficient,
    open_frequency_hz,
    reference_stiffness_model,
    stiffness_model_for_session,
)
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig


def _plain(gauge: float, open_midi: int, unit_weight: float) -> StringSpec:
    return StringSpec("test", gauge, gauge, unit_weight, False, open_midi)


def test_open_frequency_matches_equal_temperament() -> None:
    assert open_frequency_hz(69) == pytest.approx(440.0)
    assert open_frequency_hz(40) == pytest.approx(82.41, abs=0.01)  # low E
    assert open_frequency_hz(64) == pytest.approx(329.63, abs=0.01)  # high E


def test_b_scales_with_the_fourth_power_of_core_diameter() -> None:
    thin = _plain(0.012, 64, 0.00003239)
    thick = _plain(0.024, 64, 0.00003239)  # same mass on purpose: isolate d
    ratio = inharmonicity_coefficient(thick) / inharmonicity_coefficient(thin)
    assert ratio == pytest.approx(2.0**4, rel=1e-6)


def test_b_scales_with_the_inverse_fourth_power_of_scale_length() -> None:
    spec = ACOUSTIC_LIGHT_SET[0]
    short = inharmonicity_coefficient(spec, scale_length_in=12.7)
    long = inharmonicity_coefficient(spec, scale_length_in=25.4)
    assert short / long == pytest.approx(2.0**4, rel=1e-6)


def test_winding_lowers_inharmonicity_at_the_same_pitch() -> None:
    # A wound string carries a thin core inside heavy wrap; a plain string of
    # the same pitch needs a much thicker (stiffer) wire. The wound one must
    # come out less inharmonic — this is what separates low from high strings.
    wound = StringSpec("D3w", 0.032, 0.014, 0.00022110, True, 50)
    plain = StringSpec("D3p", 0.032, 0.032, 0.00022110, False, 50)
    assert inharmonicity_coefficient(wound) < inharmonicity_coefficient(plain)


def test_reference_model_covers_all_six_strings() -> None:
    model = reference_stiffness_model()
    assert sorted(model.log_b0) == [0, 1, 2, 3, 4, 5]
    assert all(math.isfinite(value) for value in model.log_b0.values())
    assert model.fret_exponent == 1.0


def test_fret_law_quadruples_b_at_the_octave() -> None:
    model = reference_stiffness_model()
    open_value = model.predicted_log_b(0, 0)
    octave = model.predicted_log_b(0, 12)
    assert open_value is not None and octave is not None
    # B(12) = B0 * 2^(12/6) = 4*B0.
    assert math.exp(octave - open_value) == pytest.approx(4.0, rel=1e-9)


def test_scale_length_default_is_a_real_acoustic_scale() -> None:
    assert 24.0 < DEFAULT_SCALE_LENGTH_IN < 26.0


def _ritual(b0_by_string: dict[int, float], exponent: float) -> list[StiffnessObservation]:
    """Synthesize the 18-note take: three frets on each of six strings."""
    return [
        StiffnessObservation(
            string_idx=string,
            fret=fret,
            log_b=math.log(b0) + exponent * (fret / 6.0) * LOG2,
            r2=0.99,
        )
        for string, b0 in b0_by_string.items()
        for fret in (0, 5, 12)
    ]


def test_ritual_recovers_b0_and_an_ideal_exponent() -> None:
    truth = {s: 5e-5 * (1.4**s) for s in range(6)}
    model = calibrate_from_ritual(_ritual(truth, 1.0))
    assert model is not None
    assert model.fret_exponent == pytest.approx(1.0, abs=1e-6)
    for string, b0 in truth.items():
        assert model.log_b0[string] == pytest.approx(math.log(b0), abs=1e-6)


def test_ritual_measures_a_non_ideal_exponent_instead_of_assuming_it() -> None:
    # The whole point of asking for three frets per string: if real fretting
    # does not scale ideally, the take detects it.
    truth = {s: 5e-5 * (1.4**s) for s in range(6)}
    model = calibrate_from_ritual(_ritual(truth, 1.35))
    assert model is not None
    assert model.fret_exponent == pytest.approx(1.35, abs=1e-6)


def test_ritual_falls_back_to_the_ideal_exponent_when_undersampled() -> None:
    # Only one fret per string: the exponent is unmeasurable, so it must not
    # be invented from noise.
    observations = [
        StiffnessObservation(string_idx=s, fret=0, log_b=math.log(5e-5), r2=0.99) for s in range(6)
    ]
    model = calibrate_from_ritual(observations)
    assert model is not None
    assert model.fret_exponent == 1.0


def test_ritual_rejects_takes_with_no_usable_fits() -> None:
    observations = [StiffnessObservation(string_idx=0, fret=0, log_b=math.log(5e-5), r2=0.01)]
    assert calibrate_from_ritual(observations, min_r2=0.5) is None


def _acoustic() -> SessionConfig:
    return SessionConfig(instrument="acoustic", tone="clean")


def test_steel_table_applies_to_clean_steel_acoustic() -> None:
    model = stiffness_model_for_session(_acoustic())
    assert model is not None
    assert sorted(model.log_b0) == [0, 1, 2, 3, 4, 5]


def test_classical_sessions_no_longer_get_the_steel_table() -> None:
    # Before N2 the channel abstained on classical (nylon is ~65x less
    # inharmonic, so the steel table would be wrong by far more than the
    # decision margin). N2 gave it a nylon table, so it now returns a model —
    # but it must never be the steel one.
    model = stiffness_model_for_session(SessionConfig(instrument="classical"))
    assert model is not None
    assert model.log_b0 != reference_stiffness_model().log_b0


def test_electric_and_distorted_sessions_get_no_table() -> None:
    assert stiffness_model_for_session(SessionConfig(instrument="electric")) is None
    assert (
        stiffness_model_for_session(SessionConfig(instrument="acoustic", tone="distorted")) is None
    )


def test_capo_and_alternate_tuning_get_no_table() -> None:
    # B0 describes the *open* string; a capo or retune moves both speaking
    # length and tension, so the table no longer applies.
    assert stiffness_model_for_session(_acoustic(), GuitarConfig(capo=3)) is None
    dropped = GuitarConfig(tuning_midi=(38, 45, 50, 55, 59, 64))
    assert stiffness_model_for_session(_acoustic(), dropped) is None


def test_out_of_domain_sessions_are_bit_identical_to_baseline() -> None:
    """The GAPS classical no-regression check, satisfied by construction.

    A classical session yields no table, and a ``None`` table must leave every
    event untouched — so classical routing cannot change, and the cross-domain
    gate needs no transcription to confirm it.
    """
    events = [
        AudioEvent(onset_s=0.0, offset_s=0.5, pitch_midi=64, velocity=0.8, confidence=0.9),
        AudioEvent(onset_s=1.0, offset_s=1.5, pitch_midi=59, velocity=0.8, confidence=0.9),
    ]
    model = stiffness_model_for_session(SessionConfig(instrument="classical"))
    out, tally = attach_inharmonicity_evidence(events, np.zeros(44100), 44100, model)
    assert out == events
    assert all(event.fret_prior is None for event in out)
    assert tally["applied"] == 0


def test_classical_session_gets_the_nylon_table() -> None:
    from tabvision.fusion.string_physics import (
        CLASSICAL_NYLON_SET,
        classical_stiffness_model,
    )

    model = stiffness_model_for_session(SessionConfig(instrument="classical"))
    assert model is not None
    assert model.log_b0 == classical_stiffness_model().log_b0
    assert len(CLASSICAL_NYLON_SET) == 6


def test_steel_and_nylon_tables_are_distinct() -> None:
    from tabvision.fusion.string_physics import classical_stiffness_model

    steel = reference_stiffness_model().log_b0
    nylon = classical_stiffness_model().log_b0
    # Nylon is far less inharmonic on the wound basses (floss core, ~200x
    # lower modulus), so no string should coincide.
    assert all(abs(steel[s] - nylon[s]) > 0.05 for s in range(6))


def test_acoustic_still_gets_the_steel_table() -> None:
    # N2 widened the routing; the steel path must be untouched.
    model = stiffness_model_for_session(SessionConfig(instrument="acoustic"))
    assert model is not None
    assert model.log_b0 == reference_stiffness_model().log_b0


def test_nylon_treble_mass_is_derived_from_density_not_fitted() -> None:
    # The three trebles are plain monofilament: mu = rho * pi * r^2, so a
    # thicker treble is proportionally heavier. This guards the first-
    # principles claim for those rows.
    from tabvision.fusion.string_physics import CLASSICAL_NYLON_SET

    g3, b3, e4 = CLASSICAL_NYLON_SET[3], CLASSICAL_NYLON_SET[4], CLASSICAL_NYLON_SET[5]
    for spec in (g3, b3, e4):
        assert not spec.wound
        assert spec.core_diameter_in == spec.gauge_in  # plain: core is the gauge
    ratio = g3.unit_weight_lb_per_in / e4.unit_weight_lb_per_in
    assert ratio == pytest.approx((g3.gauge_in / e4.gauge_in) ** 2, rel=1e-6)


def test_electric_and_capo_still_abstain_after_widening() -> None:
    assert stiffness_model_for_session(SessionConfig(instrument="electric")) is None
    assert (
        stiffness_model_for_session(SessionConfig(instrument="classical"), GuitarConfig(capo=2))
        is None
    )
