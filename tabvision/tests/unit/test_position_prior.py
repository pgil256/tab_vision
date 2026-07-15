"""Unit tests for learned pitch-position priors."""

from __future__ import annotations

import numpy as np

import tabvision.fusion.position_prior as position_prior
from tabvision.fusion import fuse
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
)
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


def _gold(t: float, string_idx: int, fret: int, pitch: int) -> TabEvent:
    return TabEvent(
        onset_s=t,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def _audio(t: float, pitch: int) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=pitch,
        velocity=1.0,
        confidence=1.0,
    )


def test_learned_prior_prefers_observed_string_fret_for_pitch():
    prior = learn_pitch_position_prior(
        [_gold(0.0, string_idx=3, fret=14, pitch=69)],
        alpha=0.1,
        power=2.0,
    )

    matrix = prior.matrix_for_pitch(69)

    assert matrix.shape == (6, 25)
    assert matrix[3, 14] > matrix[5, 5]
    assert matrix[0, 0] == 0.0
    assert np.isclose(matrix.sum(), 1.0)


def test_prior_attachment_copies_audio_events_without_mutating_original():
    matrix = np.zeros((6, 25), dtype=np.float64)
    matrix[5, 5] = 1.0
    prior = PitchPositionPrior({69: matrix})
    event = _audio(0.0, 69)

    attached = apply_pitch_position_prior([event], prior)

    assert attached[0] is not event
    assert event.fret_prior is None
    assert attached[0].fret_prior is prior.matrix_for_pitch(69)


def test_learned_prior_can_override_lowest_fret_audio_only_pick():
    prior = learn_pitch_position_prior(
        [_gold(0.0, string_idx=3, fret=14, pitch=69) for _ in range(4)],
        alpha=0.1,
        power=2.0,
    )

    event = apply_pitch_position_prior([_audio(0.0, 69)], prior)[0]
    decoded = fuse([event], [], GuitarConfig(), lambda_vision=0.0)

    assert [(ev.string_idx, ev.fret) for ev in decoded] == [(3, 14)]


def test_named_prior_artifact_loads_normalized_versioned_matrices():
    prior = position_prior.load_pitch_position_prior("guitarset-v1")

    matrix = prior.matrix_for_pitch(69)

    assert matrix is not None
    assert matrix.shape == (6, 25)
    assert np.isclose(matrix.sum(), 1.0)
    assert prior.matrix_for_pitch(20) is None


def test_unknown_named_prior_artifact_fails_with_clear_error():
    try:
        position_prior.load_pitch_position_prior("missing-prior")
    except ValueError as exc:
        assert "unknown pitch-position prior" in str(exc)
    else:
        raise AssertionError("unknown prior name should fail")
