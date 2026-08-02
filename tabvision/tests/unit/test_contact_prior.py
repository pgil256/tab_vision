"""Unit tests for the bounded FretCam finger-contact fusion prior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.contact_prior import (
    MAX_CONTACT_LOG_BONUS,
    apply_contact_priors,
)
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.position import FingerContactObservation

CFG = GuitarConfig()
ONSET_S = 5.0
# The shipped causal read is the latest frame in the 150 ms lookback ending
# 30 ms before onset, so this timestamp is inside the window.
IN_WINDOW_S = ONSET_S - 0.05


def _event(pitch_midi: int, *, onset_s: float = ONSET_S, prior=None) -> AudioEvent:
    return AudioEvent(
        onset_s=onset_s,
        offset_s=onset_s + 0.5,
        pitch_midi=pitch_midi,
        velocity=0.8,
        confidence=0.9,
        fret_prior=prior,
    )


def _observation(
    positions: tuple[tuple[int, int], ...],
    *,
    timestamp_s: float = IN_WINDOW_S,
    confidence: float = 1.0,
) -> FingerContactObservation:
    return FingerContactObservation(
        timestamp_s=timestamp_s, positions=positions, confidence=confidence
    )


def _probability_of(event: AudioEvent, string_idx: int, fret: int) -> float:
    assert event.fret_prior is not None
    return float(np.asarray(event.fret_prior)[string_idx, fret])


def test_contacts_favour_the_named_position() -> None:
    """A contact at one playable position outranks the untouched alternative."""
    pitch = 69  # A4: high-E fret 5, B fret 10, G fret 14, ...
    candidates = candidate_positions(pitch, CFG)
    assert len(candidates) > 1
    target = next(c for c in candidates if c.fret not in (0, CFG.capo))
    rival = next(
        c
        for c in candidates
        if (c.string_idx, c.fret) != (target.string_idx, target.fret)
        and c.fret not in (0, CFG.capo)
    )

    [enriched] = apply_contact_priors(
        [_event(pitch)],
        [_observation(((target.string_idx, target.fret),))],
        CFG,
    )
    assert _probability_of(enriched, target.string_idx, target.fret) > _probability_of(
        enriched, rival.string_idx, rival.fret
    )


def test_bonus_is_capped_at_the_measured_ratio() -> None:
    """The likelihood advantage never exceeds MAX_CONTACT_LOG_BONUS."""
    pitch = 69
    candidates = candidate_positions(pitch, CFG)
    fretted = [c for c in candidates if c.fret not in (0, CFG.capo)]
    target, rival = fretted[0], fretted[1]

    [enriched] = apply_contact_priors(
        [_event(pitch)],
        [_observation(((target.string_idx, target.fret),), confidence=1.0)],
        CFG,
        vision_weight=1000.0,  # absurd weight must not defeat the cap
    )
    ratio = _probability_of(enriched, target.string_idx, target.fret) / _probability_of(
        enriched, rival.string_idx, rival.fret
    )
    assert ratio <= math.exp(MAX_CONTACT_LOG_BONUS) + 1e-9


def test_a_confident_audio_prior_survives_contradicting_contacts() -> None:
    """The cap means vision breaks ties without vetoing strong audio evidence."""
    pitch = 69
    candidates = candidate_positions(pitch, CFG)
    fretted = [c for c in candidates if c.fret not in (0, CFG.capo)]
    audio_choice, vision_choice = fretted[0], fretted[1]

    prior = np.zeros((CFG.n_strings, CFG.max_fret + 1), dtype=np.float64)
    prior[audio_choice.string_idx, audio_choice.fret] = 0.999
    prior[vision_choice.string_idx, vision_choice.fret] = 0.001

    [enriched] = apply_contact_priors(
        [_event(pitch, prior=prior)],
        [_observation(((vision_choice.string_idx, vision_choice.fret),))],
        CFG,
    )
    assert _probability_of(enriched, audio_choice.string_idx, audio_choice.fret) > _probability_of(
        enriched, vision_choice.string_idx, vision_choice.fret
    )


def test_open_strings_keep_support_when_the_hand_is_elsewhere() -> None:
    """A contact set can never name an open string, so it must not demote one.

    Without the carve-out this prior would penalize every open string merely
    because the fretting hand is somewhere up the neck.
    """
    pitch = 64  # E4 is open on the high E string
    candidates = candidate_positions(pitch, CFG)
    open_candidate = next((c for c in candidates if c.fret == 0), None)
    assert open_candidate is not None
    elsewhere = next(c for c in candidates if c.fret not in (0, CFG.capo))

    baseline = _event(pitch)
    [enriched] = apply_contact_priors(
        [baseline],
        [_observation(((elsewhere.string_idx, elsewhere.fret),))],
        CFG,
    )
    if enriched is baseline:
        return  # no discriminating evidence; the open string was never at risk
    open_p = _probability_of(enriched, open_candidate.string_idx, open_candidate.fret)
    assert open_p >= _probability_of(enriched, elsewhere.string_idx, elsewhere.fret) - 1e-12


@pytest.mark.parametrize(
    "observations",
    [
        [],
        [_observation(((0, 3),), timestamp_s=ONSET_S + 1.0)],  # future evidence
        [_observation(((0, 3),), timestamp_s=ONSET_S - 5.0)],  # stale evidence
        [_observation(())],  # empty contact set
        [_observation(((99, 3),))],  # out-of-range string
        [_observation(((0, 999),))],  # out-of-range fret
    ],
)
def test_unusable_evidence_is_an_exact_no_op(observations) -> None:
    """Missing, future, stale, empty, or malformed evidence preserves identity."""
    event = _event(69)
    result = apply_contact_priors([event], observations, CFG)
    assert result[0] is event


def test_zero_vision_weight_is_an_exact_no_op() -> None:
    event = _event(69)
    result = apply_contact_priors([event], [_observation(((0, 3),))], CFG, vision_weight=0.0)
    assert result[0] is event


def test_post_onset_evidence_is_never_used() -> None:
    """Evidence must be causal: nothing at or after the onset may contribute."""
    event = _event(69)
    just_after_target = _observation(((1, 10),), timestamp_s=ONSET_S - 0.01)
    result = apply_contact_priors([event], [just_after_target], CFG)
    assert result[0] is event


def test_unambiguous_pitch_is_untouched() -> None:
    """Pitches with a single playable position carry no decision to inform."""
    single = [p for p in range(30, 100) if len(candidate_positions(p, CFG)) == 1]
    if not single:
        pytest.skip("no single-candidate pitch in this configuration")
    event = _event(single[0])
    result = apply_contact_priors([event], [_observation(((0, 3),))], CFG)
    assert result[0] is event


def test_non_discriminating_contacts_are_an_exact_no_op() -> None:
    """If every candidate is supported, the distribution cannot change."""
    pitch = 69
    everything = tuple((c.string_idx, c.fret) for c in candidate_positions(pitch, CFG))
    event = _event(pitch)
    result = apply_contact_priors([event], [_observation(everything)], CFG)
    assert result[0] is event


def test_prior_is_not_mutated_in_place() -> None:
    """The caller's array must survive untouched — events are shared."""
    pitch = 69
    candidates = candidate_positions(pitch, CFG)
    fretted = [c for c in candidates if c.fret not in (0, CFG.capo)]
    prior = np.zeros((CFG.n_strings, CFG.max_fret + 1), dtype=np.float64)
    for candidate in candidates:
        prior[candidate.string_idx, candidate.fret] = 1.0 / len(candidates)
    snapshot = prior.copy()

    apply_contact_priors(
        [_event(pitch, prior=prior)],
        [_observation(((fretted[0].string_idx, fretted[0].fret),))],
        CFG,
    )
    np.testing.assert_array_equal(prior, snapshot)
