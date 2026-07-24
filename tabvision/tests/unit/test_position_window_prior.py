"""Tests for bounded causal FretCam position-window fusion."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tabvision.fusion import fuse
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_window_prior import (
    MAX_POSITION_LOG_BONUS,
    POSITION_OBSERVATION_LEAD_S,
    POSITION_OBSERVATION_LOOKBACK_S,
    apply_position_window_priors,
)
from tabvision.fusion.viterbi import assignment_decoder_context
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.position import PositionWindowObservation


def _event(
    *,
    onset_s: float = 1.0,
    pitch_midi: int = 69,
    prior: np.ndarray | None = None,
) -> AudioEvent:
    return AudioEvent(
        onset_s=onset_s,
        offset_s=onset_s + 0.25,
        pitch_midi=pitch_midi,
        velocity=0.8,
        confidence=0.9,
        fret_prior=prior,
    )


def _observation(
    timestamp_s: float = 0.95,
    *,
    position: int = 10,
    window_frets: tuple[int, ...] = (0, 9, 10, 11, 12, 13, 14),
    confidence: float = 0.8,
    state: str = "locked",
) -> PositionWindowObservation:
    return PositionWindowObservation(
        timestamp_s=timestamp_s,
        position=position,
        window_frets=window_frets,
        confidence=confidence,
        state=state,  # type: ignore[arg-type]
    )


def _candidate_value(prior: np.ndarray, cfg: GuitarConfig, pitch: int, fret: int) -> float:
    candidate = next(item for item in candidate_positions(pitch, cfg) if item.fret == fret)
    return float(prior[candidate.string_idx, candidate.fret])


@pytest.mark.parametrize(
    ("observations", "weight"),
    [
        ([], 1.0),
        ([_observation()], 0.0),
        ([_observation()], -1.0),
        ([_observation()], math.nan),
    ],
)
def test_no_evidence_or_disabled_weight_is_exact_event_noop(
    observations: list[PositionWindowObservation],
    weight: float,
) -> None:
    event = _event()

    enriched = apply_position_window_priors(
        [event],
        observations,
        GuitarConfig(),
        vision_weight=weight,
    )

    assert enriched[0] is event
    assert enriched[0].fret_prior is None


def test_latest_valid_observation_is_selected_causally() -> None:
    cfg = GuitarConfig()
    event = _event()
    observations = [
        _observation(0.84, position=5, window_frets=(0, 4, 5, 6, 7, 8, 9)),
        _observation(0.96, position=10),
        _observation(0.98, position=5, window_frets=(0, 4, 5, 6, 7, 8, 9)),
    ]

    (enriched,) = apply_position_window_priors([event], observations, cfg)

    assert enriched.fret_prior is not None
    assert _candidate_value(enriched.fret_prior, cfg, 69, 10) > _candidate_value(
        enriched.fret_prior, cfg, 69, 5
    )


@pytest.mark.parametrize(
    "timestamp_s",
    [
        1.0 - POSITION_OBSERVATION_LEAD_S - POSITION_OBSERVATION_LOOKBACK_S - 1e-6,
        1.0 - POSITION_OBSERVATION_LEAD_S + 1e-6,
        1.01,
    ],
)
def test_stale_or_post_target_observation_is_rejected(timestamp_s: float) -> None:
    event = _event()

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(timestamp_s)],
        GuitarConfig(),
    )

    assert enriched is event


@pytest.mark.parametrize(
    "timestamp_s",
    [
        1.0 - POSITION_OBSERVATION_LEAD_S - POSITION_OBSERVATION_LOOKBACK_S,
        1.0 - POSITION_OBSERVATION_LEAD_S,
    ],
)
def test_causal_window_boundaries_are_inclusive(timestamp_s: float) -> None:
    event = _event()

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(timestamp_s)],
        GuitarConfig(),
    )

    assert enriched is not event


def test_negative_pre_audio_timestamp_can_support_an_early_note() -> None:
    event = _event(onset_s=0.10)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(timestamp_s=-0.01)],
        GuitarConfig(),
    )

    assert enriched is not event


@pytest.mark.parametrize("state", ["acquiring", "shifting", "lost"])
def test_unstable_fretcam_states_are_strict_noop(state: str) -> None:
    event = _event()

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(state=state)],
        GuitarConfig(),
    )

    assert enriched is event


@pytest.mark.parametrize(
    "observation",
    [
        _observation(confidence=0.199999),
        _observation(confidence=math.nan),
        _observation(timestamp_s=math.nan),
        _observation(position=25),
        _observation(window_frets=()),
        _observation(window_frets=(0, 10, 11)),
        _observation(window_frets=(0, 9, 25)),
        _observation(window_frets=(0, 9, math.nan)),  # type: ignore[arg-type]
    ],
)
def test_invalid_or_out_of_range_observation_is_strict_noop(
    observation: PositionWindowObservation,
) -> None:
    event = _event()

    (enriched,) = apply_position_window_priors(
        [event],
        [observation],
        GuitarConfig(),
    )

    assert enriched is event


@pytest.mark.parametrize("state", ["locked", "holding"])
def test_fixed_window_bonus_always_preserves_open_strings(state: str) -> None:
    cfg = GuitarConfig()
    event = _event(pitch_midi=64)
    observation = _observation(
        position=9,
        window_frets=(0, 8, 9, 10, 11, 12, 13),
        state=state,
    )

    (enriched,) = apply_position_window_priors([event], [observation], cfg)

    assert enriched.fret_prior is not None
    open_probability = _candidate_value(enriched.fret_prior, cfg, 64, 0)
    in_window_probability = _candidate_value(enriched.fret_prior, cfg, 64, 9)
    outside_probability = _candidate_value(enriched.fret_prior, cfg, 64, 5)
    assert open_probability == pytest.approx(in_window_probability)
    assert open_probability / outside_probability == pytest.approx(math.exp(0.8))


@pytest.mark.parametrize("one_dimensional", [False, True])
def test_existing_prior_is_not_mutated_and_vision_odds_are_capped(
    one_dimensional: bool,
) -> None:
    cfg = GuitarConfig()
    candidates = candidate_positions(69, cfg)
    if one_dimensional:
        existing = np.ones(cfg.max_fret + 1, dtype=np.float64)
        existing[5] = 0.6
        existing[10] = 0.2
    else:
        existing = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
        for candidate in candidates:
            existing[candidate.string_idx, candidate.fret] = 0.1
        low = next(item for item in candidates if item.fret == 5)
        high = next(item for item in candidates if item.fret == 10)
        existing[low.string_idx, low.fret] = 0.6
        existing[high.string_idx, high.fret] = 0.2
    before = existing.copy()
    event = _event(prior=existing)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(confidence=1.0)],
        cfg,
        vision_weight=100.0,
    )

    assert event.fret_prior is existing
    np.testing.assert_array_equal(existing, before)
    assert enriched.fret_prior is not None
    before_odds = _candidate_value_from_any(existing, cfg, 69, 10) / _candidate_value_from_any(
        existing, cfg, 69, 5
    )
    after_odds = _candidate_value(enriched.fret_prior, cfg, 69, 10) / _candidate_value(
        enriched.fret_prior, cfg, 69, 5
    )
    assert after_odds / before_odds == pytest.approx(math.exp(MAX_POSITION_LOG_BONUS))


def _candidate_value_from_any(
    prior: np.ndarray,
    cfg: GuitarConfig,
    pitch: int,
    fret: int,
) -> float:
    if prior.ndim == 1:
        return float(prior[fret])
    return _candidate_value(prior, cfg, pitch, fret)


def test_position_window_rescues_ambiguous_right_pitch_wrong_position() -> None:
    cfg = GuitarConfig()
    event = _event(onset_s=1.0, pitch_midi=69)
    with assignment_decoder_context("baseline"):
        baseline = fuse([event], [], cfg, lambda_vision=0.0)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(0.96, confidence=1.0)],
        cfg,
    )
    with assignment_decoder_context("baseline"):
        assisted = fuse([enriched], [], cfg, lambda_vision=0.0)

    assert (baseline[0].string_idx, baseline[0].fret) == (5, 5)
    assert (assisted[0].string_idx, assisted[0].fret) == (4, 10)
    assert assisted[0].pitch_midi == baseline[0].pitch_midi == 69


def test_bounded_vision_does_not_overwhelm_strong_conflicting_audio_prior() -> None:
    cfg = GuitarConfig()
    prior = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    for candidate in candidate_positions(69, cfg):
        prior[candidate.string_idx, candidate.fret] = 0.001
    low = next(item for item in candidate_positions(69, cfg) if item.fret == 5)
    high = next(item for item in candidate_positions(69, cfg) if item.fret == 10)
    prior[low.string_idx, low.fret] = 0.90
    prior[high.string_idx, high.fret] = 0.09
    event = _event(prior=prior)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(confidence=1.0)],
        cfg,
        vision_weight=100.0,
    )
    with assignment_decoder_context("baseline"):
        (decoded,) = fuse([enriched], [], cfg, lambda_vision=0.0)

    assert (decoded.string_idx, decoded.fret) == (low.string_idx, low.fret)


@pytest.mark.parametrize("surviving_fret", [5, 10])
def test_hard_audio_elimination_is_an_exact_noop(surviving_fret: int) -> None:
    cfg = GuitarConfig()
    prior = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    survivor = next(item for item in candidate_positions(69, cfg) if item.fret == surviving_fret)
    prior[survivor.string_idx, survivor.fret] = 1.0
    event = _event(prior=prior)

    (unchanged,) = apply_position_window_priors(
        [event],
        [_observation(confidence=1.0)],
        cfg,
    )

    assert unchanged is event
    assert unchanged.fret_prior is prior


def test_high_position_observation_keeps_open_high_e_decoded_open() -> None:
    cfg = GuitarConfig()
    event = _event(pitch_midi=64)
    observation = _observation(
        position=19,
        window_frets=(0, 18, 19, 20, 21, 22, 23),
        confidence=1.0,
    )

    (enriched,) = apply_position_window_priors([event], [observation], cfg)
    with assignment_decoder_context("baseline"):
        (decoded,) = fuse([enriched], [], cfg, lambda_vision=0.0)

    assert (decoded.string_idx, decoded.fret, decoded.pitch_midi) == (5, 0, 64)


def test_unambiguous_pitch_and_non_discriminating_window_are_exact_noops() -> None:
    cfg = GuitarConfig()
    unambiguous = _event(pitch_midi=40)
    # MIDI 45 is playable as open A or low-E fret 5. Position I supports
    # both, so the valid observation cannot distinguish its candidates.
    all_candidates_supported = _event(pitch_midi=45)

    result = apply_position_window_priors(
        [unambiguous, all_candidates_supported],
        [_observation(position=1, window_frets=(0, 1, 2, 3, 4, 5))],
        cfg,
    )

    assert result[0] is unambiguous
    assert result[1] is all_candidates_supported


def test_custom_max_fret_and_capo_limit_playable_candidate_support() -> None:
    cfg = GuitarConfig(capo=5, max_fret=12)
    event = _event(pitch_midi=69)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(position=12, window_frets=(0, 11, 12))],
        cfg,
    )

    assert enriched.fret_prior is not None
    assert enriched.fret_prior.shape == (cfg.n_strings, cfg.max_fret + 1)
    nonzero = {
        (int(string_idx), int(fret)) for string_idx, fret in np.argwhere(enriched.fret_prior > 0.0)
    }
    playable = {
        (candidate.string_idx, candidate.fret)
        for candidate in candidate_positions(event.pitch_midi, cfg)
    }
    assert nonzero == playable
    assert all(fret >= cfg.capo for _, fret in nonzero)

    invalid = replace(
        _observation(position=10),
        window_frets=(0, 9, 10, 11, 12, 13),
    )
    (unchanged,) = apply_position_window_priors([event], [invalid], cfg)
    assert unchanged is event


def test_capoed_open_string_is_preserved_outside_the_hand_window() -> None:
    cfg = GuitarConfig(capo=5)
    # MIDI 69 is the capoed-open high E string (physical fret 5) and is also
    # playable at higher frets on lower strings.
    event = _event(pitch_midi=69)

    (enriched,) = apply_position_window_priors(
        [event],
        [_observation(position=14, window_frets=(0, 13, 14, 15, 16, 17, 18))],
        cfg,
    )

    assert enriched.fret_prior is not None
    capo_open = _candidate_value(enriched.fret_prior, cfg, 69, cfg.capo)
    in_window = _candidate_value(enriched.fret_prior, cfg, 69, 14)
    outside = _candidate_value(enriched.fret_prior, cfg, 69, 10)
    assert capo_open == pytest.approx(in_window)
    assert capo_open > outside
