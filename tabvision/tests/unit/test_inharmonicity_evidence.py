"""Tests for the inharmonicity string-evidence channel.

Two properties are load-bearing. The channel must **abstain** rather than
guess — on chorded notes, unfittable notes and zero weight it has to return
the stream untouched, because a bounded evidence term that quietly fires on
material it cannot measure is how A14-style regressions happen. And it must
never touch onset, offset or pitch: it is a string-assignment channel, and
the fusion eval's bit-identical onset/pitch F1 depends on that being true by
construction rather than by luck.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabvision.fusion.inharmonicity import (
    StringStiffnessModel,
    attach_inharmonicity_evidence,
    estimate_inharmonicity,
    inharmonicity_matrix,
)
from tabvision.types import AudioEvent, GuitarConfig

SR = 44100
OPEN_MIDI = (40, 45, 50, 55, 59, 64)


def _stiff_string(f0: float, b_value: float, *, seconds: float = 0.45) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(t)
    for k in range(1, 11):
        freq = k * f0 * math.sqrt(1.0 + b_value * k * k)
        if freq > SR / 2.5:
            break
        signal += np.sin(2 * np.pi * freq * t) / k
    return signal


def _model() -> StringStiffnessModel:
    # Low strings are stiffer per unit length in this toy table; what matters
    # is that the six entries are distinct enough to be separable.
    return StringStiffnessModel(log_b0={s: math.log(6e-5 + 4e-5 * s) for s in range(6)})


def _event(onset: float, pitch: int, *, duration: float = 0.45) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + duration,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=0.9,
    )


def test_estimate_recovers_known_inharmonicity() -> None:
    fit = estimate_inharmonicity(_stiff_string(110.0, 1e-4), SR, 110.0)
    assert fit is not None
    assert fit.f0_hz == pytest.approx(110.0, rel=0.01)
    assert fit.b == pytest.approx(1e-4, rel=0.25)
    assert fit.r2 > 0.9
    assert fit.log_b == pytest.approx(math.log(fit.b))


def test_matrix_favours_the_candidate_matching_the_measurement() -> None:
    cfg = GuitarConfig()
    model = _model()
    pitch = 64  # E4: open high-E, or high frets on lower strings
    # Feed exactly string 5's predicted log B; it must win its own lookup.
    target = model.predicted_log_b(5, pitch - OPEN_MIDI[5])
    assert target is not None
    matrix = inharmonicity_matrix(pitch, cfg, target, model)
    assert matrix is not None
    assert matrix.sum() == pytest.approx(1.0)
    best = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    assert best[0] == 5


def test_matrix_is_none_for_unambiguous_pitches() -> None:
    cfg = GuitarConfig()
    # MIDI 40 is the open low E and playable nowhere else.
    assert inharmonicity_matrix(40, cfg, math.log(1e-4), _model()) is None


def test_attach_never_alters_onsets_offsets_or_pitch() -> None:
    events = [_event(0.0, 64), _event(2.0, 59)]
    wav = np.concatenate(
        [_stiff_string(330.0, 1e-4), np.zeros(int(SR * 1.55)), _stiff_string(247.0, 1e-4)]
    )
    out, _tally = attach_inharmonicity_evidence(events, wav, SR, _model())
    assert [e.onset_s for e in out] == [e.onset_s for e in events]
    assert [e.offset_s for e in out] == [e.offset_s for e in events]
    assert [e.pitch_midi for e in out] == [e.pitch_midi for e in events]


def test_attach_abstains_on_overlapping_notes() -> None:
    # Two notes sounding together: neither is isolated, so nothing applies.
    events = [_event(0.0, 64), _event(0.05, 59)]
    wav = _stiff_string(330.0, 1e-4)
    out, tally = attach_inharmonicity_evidence(events, wav, SR, _model())
    assert tally["isolated"] == 0
    assert tally["applied"] == 0
    assert all(event.fret_prior is None for event in out)


def test_zero_weight_is_an_exact_identity() -> None:
    events = [_event(0.0, 64)]
    wav = _stiff_string(330.0, 1e-4)
    out, tally = attach_inharmonicity_evidence(events, wav, SR, _model(), weight=0.0)
    assert out == events
    assert tally["applied"] == 0


def test_attach_abstains_when_the_fit_is_poor() -> None:
    events = [_event(0.0, 64)]
    noise = np.random.default_rng(0).normal(0.0, 1.0, int(SR * 0.45))
    out, tally = attach_inharmonicity_evidence(events, noise, SR, _model(), min_r2=0.99)
    assert tally["applied"] == 0
    assert out[0].fret_prior is None


def test_attach_applies_to_a_clean_isolated_note() -> None:
    events = [_event(0.0, 64)]
    wav = _stiff_string(330.0, 1e-4)
    out, tally = attach_inharmonicity_evidence(events, wav, SR, _model(), min_r2=0.0)
    assert tally == {"events": 1, "isolated": 1, "fitted": 1, "applied": 1}
    prior = out[0].fret_prior
    assert prior is not None
    assert prior.sum() == pytest.approx(1.0)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        attach_inharmonicity_evidence(
            [_event(0.0, 64)], _stiff_string(330.0, 1e-4), SR, _model(), weight=-1.0
        )


def test_sigma_must_be_positive() -> None:
    with pytest.raises(ValueError):
        inharmonicity_matrix(64, GuitarConfig(), math.log(1e-4), _model(), sigma=0.0)


def _overlapping_pair() -> list[AudioEvent]:
    """Two notes sounding together — neither is isolated."""
    return [_event(0.0, 64), _event(0.05, 59)]


def test_strict_mode_is_the_shipped_v1_behaviour() -> None:
    # v1 froze `strict`, and its gates are only valid if this stays a no-op
    # on overlapped notes.
    events = _overlapping_pair()
    out, tally = attach_inharmonicity_evidence(
        events, _stiff_string(330.0, 1e-4), SR, _model(), isolation="strict"
    )
    assert tally["applied"] == 0
    assert out == events


def test_partial_aware_attempts_overlapped_notes() -> None:
    # The whole point of N1: an overlapped note is no longer discarded
    # unheard, it is fitted on whatever partials survive.
    events = _overlapping_pair()
    _out, tally = attach_inharmonicity_evidence(
        events,
        _stiff_string(330.0, 1e-4),
        SR,
        _model(),
        isolation="partial_aware",
        min_r2=0.0,
        min_clean_partials=1,
    )
    assert tally["isolated"] == len(events)  # both now reach the fit stage


def test_min_clean_partials_gates_contaminated_fits() -> None:
    # Demanding more surviving partials than a contaminated note can supply
    # must make it abstain rather than trust a thin fit.
    events = _overlapping_pair()
    _out, tally = attach_inharmonicity_evidence(
        events,
        _stiff_string(330.0, 1e-4),
        SR,
        _model(),
        isolation="partial_aware",
        min_r2=0.0,
        min_clean_partials=99,
    )
    assert tally["applied"] == 0


def test_unknown_isolation_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        attach_inharmonicity_evidence(
            [_event(0.0, 64)], _stiff_string(330.0, 1e-4), SR, _model(), isolation="nonsense"
        )


def test_blocked_partials_are_skipped_by_the_estimator() -> None:
    # A blocked band must reduce the partial count rather than be measured.
    clean = estimate_inharmonicity(_stiff_string(110.0, 1e-4), SR, 110.0)
    blocked = estimate_inharmonicity(
        _stiff_string(110.0, 1e-4),
        SR,
        110.0,
        blocked_hz=[220.0, 330.0],
        min_separation_hz=25.0,
    )
    assert clean is not None and blocked is not None
    assert blocked.partials < clean.partials
