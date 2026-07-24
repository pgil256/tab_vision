"""Tests for the N5 table-mismatch study.

Two things carry the study's validity. The alternative string sets must be
**anchored to the shipped table** — at the shipped gauges the derived model
has to reproduce `ACOUSTIC_LIGHT_SET`, or a variant's measured effect mixes
the gauge change with an unrelated modelling difference. And the offline
replay must equal `attach_inharmonicity_evidence` exactly, since every
variant's number comes from the replay rather than from the shipped path.
"""

from __future__ import annotations

import math

import numpy as np

from scripts.eval.n5_table_mismatch import (
    LIGHT_GAUGES,
    MIN_R2,
    SIGMA,
    WEIGHT,
    apply_banked,
    derive_set,
    fit_wound_model,
    model_from_specs,
    offset_model,
    perturbed_model,
    plain_unit_weight_lb_per_in,
)
from tabvision.fusion.inharmonicity import (
    attach_inharmonicity_evidence,
    measure_events,
)
from tabvision.fusion.string_physics import (
    ACOUSTIC_LIGHT_SET,
    DEFAULT_SCALE_LENGTH_IN,
    reference_stiffness_model,
)
from tabvision.types import AudioEvent, GuitarConfig

SR = 44100


def _stiff_string(f0: float, b_value: float, *, seconds: float = 0.45) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(t)
    for k in range(1, 11):
        freq = k * f0 * math.sqrt(1.0 + b_value * k * k)
        if freq > SR / 2.5:
            break
        signal += np.sin(2 * np.pi * freq * t) / k
    return signal


def _event(onset: float, pitch: int, *, duration: float = 0.45) -> AudioEvent:
    return AudioEvent(
        onset_s=onset, offset_s=onset + duration, pitch_midi=pitch, velocity=0.8, confidence=0.9
    )


def test_derived_light_set_reproduces_the_shipped_table() -> None:
    """The anchor: same gauges in, shipped specifications out."""
    derived = derive_set(LIGHT_GAUGES, fit_wound_model())
    for got, want in zip(derived, ACOUSTIC_LIGHT_SET, strict=True):
        assert got.gauge_in == want.gauge_in
        assert got.wound is want.wound
        assert got.open_midi == want.open_midi
        assert (
            got.core_diameter_in == want.core_diameter_in
            or abs(got.core_diameter_in - want.core_diameter_in) < 6e-4
        )
        assert abs(got.unit_weight_lb_per_in - want.unit_weight_lb_per_in) < 0.06 * (
            want.unit_weight_lb_per_in
        )


def test_derived_light_table_is_close_to_the_registered_one() -> None:
    """A residual in log-B space well under the 0.35 sigma the channel uses."""
    derived = model_from_specs(derive_set(LIGHT_GAUGES, fit_wound_model()))
    shipped = reference_stiffness_model()
    for index in range(6):
        assert abs(derived.log_b0[index] - shipped.log_b0[index]) < 0.15


def test_the_null_perturbation_is_exactly_the_registered_table() -> None:
    """Variants are differences, so the residual must cancel to zero.

    Without this the wound model's fit error (0.09 log-B on G3) would ride
    along inside every real-set arm and be read as a string effect.
    """
    null = perturbed_model(derive_set(LIGHT_GAUGES, fit_wound_model()))
    shipped = reference_stiffness_model()
    for index in range(6):
        assert abs(null.log_b0[index] - shipped.log_b0[index]) < 1e-12


def test_gauge_moves_the_plain_strings_far_more_than_the_wound_ones() -> None:
    """A gauge change is not a uniform offset.

    A plain string's B goes as d^2, while a wound string's core and total mass
    move together and largely cancel — so a lighter set reshapes the top of
    the table and barely touches the bottom.
    """
    wound = fit_wound_model()
    extra = perturbed_model(derive_set((0.010, 0.014, 0.023, 0.030, 0.039, 0.047), wound))
    shipped = reference_stiffness_model()
    shifts = [extra.log_b0[index] - shipped.log_b0[index] for index in range(6)]
    assert max(abs(shift) for shift in shifts[:4]) < 0.05
    assert min(abs(shift) for shift in shifts[4:]) > 0.20


def test_plain_unit_weight_matches_the_published_figures() -> None:
    for spec in ACOUSTIC_LIGHT_SET:
        if spec.wound:
            continue
        derived = plain_unit_weight_lb_per_in(spec.gauge_in)
        assert abs(derived - spec.unit_weight_lb_per_in) < 0.02 * spec.unit_weight_lb_per_in


def test_scale_length_is_exactly_a_uniform_offset() -> None:
    """B ~ L^-4 at fixed strings and pitch, so scale length cannot reshape."""
    wound = fit_wound_model()
    specs = derive_set(LIGHT_GAUGES, wound)
    short = model_from_specs(specs, scale_length_in=24.75)
    reference = model_from_specs(specs, scale_length_in=DEFAULT_SCALE_LENGTH_IN)
    expected = 4.0 * math.log(DEFAULT_SCALE_LENGTH_IN / 24.75)
    for index in range(6):
        assert abs((short.log_b0[index] - reference.log_b0[index]) - expected) < 1e-9


def test_offset_model_shifts_every_string_equally() -> None:
    base = reference_stiffness_model()
    shifted = offset_model(base, 0.25)
    assert shifted.fret_exponent == base.fret_exponent
    for index in range(6):
        assert abs((shifted.log_b0[index] - base.log_b0[index]) - 0.25) < 1e-12


def test_replay_equals_the_shipped_attach_path() -> None:
    """The study's load-bearing equivalence, on a signal with a known B."""
    cfg = GuitarConfig()
    model = reference_stiffness_model()
    b_value = math.exp(model.log_b0[3])  # G3 open, a genuinely ambiguous pitch
    pitch = 55
    wav = np.concatenate(
        [
            np.zeros(SR // 10),
            _stiff_string(440.0 * 2 ** ((pitch - 69) / 12.0), b_value),
            np.zeros(SR // 10),
        ]
    )
    events = [_event(0.1, pitch)]

    shipped, tally = attach_inharmonicity_evidence(
        events, wav, SR, model, cfg, weight=WEIGHT, min_r2=MIN_R2, sigma=SIGMA
    )
    fits = {
        index: (fit.log_b, fit.r2) for index, fit in measure_events(events, wav, SR, cfg).items()
    }
    replayed, applied = apply_banked(
        sorted(events, key=lambda event: event.onset_s),
        fits,
        model,
        cfg,
        weight=WEIGHT,
        min_r2=MIN_R2,
        sigma=SIGMA,
    )
    assert tally["applied"] == 1, "the fixture must exercise the applied path"
    assert applied == tally["applied"]
    for left, right in zip(shipped, replayed, strict=True):
        assert np.array_equal(left.fret_prior, right.fret_prior)


def test_replay_abstains_exactly_where_the_shipped_path_does() -> None:
    """Same equivalence on material the channel refuses: a poor fit."""
    cfg = GuitarConfig()
    model = reference_stiffness_model()
    rng = np.random.default_rng(42)
    wav = rng.standard_normal(SR // 2) * 0.01
    events = [_event(0.05, 55)]

    shipped, tally = attach_inharmonicity_evidence(
        events, wav, SR, model, cfg, weight=WEIGHT, min_r2=MIN_R2, sigma=SIGMA
    )
    fits = {
        index: (fit.log_b, fit.r2) for index, fit in measure_events(events, wav, SR, cfg).items()
    }
    replayed, applied = apply_banked(
        sorted(events, key=lambda event: event.onset_s),
        fits,
        model,
        cfg,
        weight=WEIGHT,
        min_r2=MIN_R2,
        sigma=SIGMA,
    )
    assert applied == tally["applied"]
    for left, right in zip(shipped, replayed, strict=True):
        assert np.array_equal(left.fret_prior, right.fret_prior)


def test_heavier_gauge_raises_stiffness_on_the_plain_strings() -> None:
    """Direction check: B ~ d^2 for a plain string, so medium > light > extra."""
    wound = fit_wound_model()
    extra = model_from_specs(derive_set((0.010, 0.014, 0.023, 0.030, 0.039, 0.047), wound))
    light = model_from_specs(derive_set(LIGHT_GAUGES, wound))
    medium = model_from_specs(derive_set((0.013, 0.017, 0.026, 0.035, 0.045, 0.056), wound))
    for index in (4, 5):  # the two plain strings
        assert extra.log_b0[index] < light.log_b0[index] < medium.log_b0[index]
