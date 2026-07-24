"""Causal, bounded FretCam position evidence for audio tab decoding.

FretCam estimates a coarse left-hand position rather than an exact string.
This module turns only stable, pre-onset position windows into a soft
likelihood over the playable positions for an audio event.  The likelihood
ratio contributed by vision is capped at :data:`MAX_POSITION_LOG_BONUS`, so
vision can break an ambiguous same-pitch tie without becoming a hard veto.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from numbers import Integral, Real

import numpy as np

from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.position import PositionWindowObservation

POSITION_OBSERVATION_LEAD_S = 0.03
"""Read vision this many seconds before the audio onset."""

POSITION_OBSERVATION_LOOKBACK_S = 0.15
"""Maximum age of a position observation relative to the pre-onset target."""

MIN_POSITION_OBSERVATION_CONFIDENCE = 0.20
"""Minimum composite FretCam confidence accepted by fusion."""

MAX_POSITION_OBSERVATION_CONFIDENCE = 1.0
"""Upper bound of the FretCam confidence contract."""

MAX_POSITION_LOG_BONUS = 1.0
"""Maximum vision log-likelihood advantage between two candidates."""

VALID_POSITION_OBSERVATION_STATES = frozenset({"locked", "holding"})
"""FretCam states that represent usable, stabilized position evidence."""


def apply_position_window_priors(
    events: Sequence[AudioEvent],
    observations: Sequence[PositionWindowObservation],
    cfg: GuitarConfig,
    *,
    vision_weight: float = 1.0,
) -> list[AudioEvent]:
    """Attach bounded, causal position-window likelihoods to audio events.

    For an event at time ``t``, the latest valid observation in
    ``[t - lead - lookback, t - lead]`` is used.  Only pitches with multiple
    playable string/fret assignments are changed.  Frets in the observed
    window, plus the configured open/capo fret, receive a log bonus of at most
    one nat.  Existing one- or two-dimensional fret priors are multiplied by
    that likelihood over playable candidates and are never mutated.

    Events retain object identity whenever vision contributes no
    discriminating evidence.
    """
    weight = _finite_non_negative_weight(vision_weight)
    if not observations or weight == 0.0:
        return list(events)

    valid_observations = [
        observation for observation in observations if _is_valid_observation(observation, cfg)
    ]
    if not valid_observations:
        return list(events)

    output: list[AudioEvent] = []
    for event in events:
        candidates = candidate_positions(event.pitch_midi, cfg)
        if len(candidates) <= 1 or not math.isfinite(float(event.onset_s)):
            output.append(event)
            continue

        observation = _latest_causal_observation(
            valid_observations,
            float(event.onset_s),
        )
        if observation is None:
            output.append(event)
            continue

        # ``candidate_positions`` records a capoed open string at
        # ``fret == cfg.capo`` (the physical fret under the capo), not at
        # fret zero. Preserve both representations so the vision prior never
        # penalizes an open/capoed string merely because the fretting hand is
        # elsewhere on the neck.
        supported_frets = frozenset(observation.window_frets) | {0, cfg.capo}
        supported = np.asarray(
            [candidate.fret in supported_frets for candidate in candidates],
            dtype=np.bool_,
        )
        if bool(np.all(supported)) or not bool(np.any(supported)):
            output.append(event)
            continue

        existing_values = _existing_candidate_values(event.fret_prior, candidates, cfg)
        if existing_values is None:
            output.append(event)
            continue
        if (
            float(existing_values[supported].sum()) <= 0.0
            or float(existing_values[~supported].sum()) <= 0.0
        ):
            # The existing evidence has already eliminated one side of the
            # visual distinction. Multiplying the surviving mass would
            # normalize back to the same distribution, so preserve the exact
            # event and keep the pipeline's "notes affected" count honest.
            output.append(event)
            continue

        log_bonus = min(
            MAX_POSITION_LOG_BONUS,
            weight * float(observation.confidence),
        )
        likelihood = np.ones(len(candidates), dtype=np.float64)
        likelihood[supported] = math.exp(log_bonus)
        combined_values = existing_values * likelihood
        total = float(combined_values.sum())
        if not math.isfinite(total) or total <= 0.0:
            output.append(event)
            continue

        prior = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
        for candidate, probability in zip(
            candidates,
            combined_values / total,
            strict=True,
        ):
            prior[candidate.string_idx, candidate.fret] = float(probability)
        output.append(replace(event, fret_prior=prior))

    return output


def _finite_non_negative_weight(value: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(weight):
        return 0.0
    return max(0.0, weight)


def _is_valid_observation(
    observation: PositionWindowObservation,
    cfg: GuitarConfig,
) -> bool:
    if observation.state not in VALID_POSITION_OBSERVATION_STATES:
        return False
    try:
        timestamp = float(observation.timestamp_s)
        confidence = float(observation.confidence)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(timestamp):
        return False
    if (
        not math.isfinite(confidence)
        or confidence < MIN_POSITION_OBSERVATION_CONFIDENCE
        or confidence > MAX_POSITION_OBSERVATION_CONFIDENCE
    ):
        return False

    position = _integral_value(observation.position)
    if position is None or not 1 <= position <= cfg.max_fret:
        return False
    if not observation.window_frets:
        return False
    window: list[int] = []
    for fret_value in observation.window_frets:
        fret = _integral_value(fret_value)
        if fret is None or not 0 <= fret <= cfg.max_fret:
            return False
        window.append(fret)
    expected_window = (
        0,
        *range(max(1, position - 1), min(cfg.max_fret, position + 4) + 1),
    )
    return tuple(window) == expected_window


def _integral_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return None


def _latest_causal_observation(
    observations: Sequence[PositionWindowObservation],
    onset_s: float,
) -> PositionWindowObservation | None:
    target_s = onset_s - POSITION_OBSERVATION_LEAD_S
    earliest_s = target_s - POSITION_OBSERVATION_LOOKBACK_S
    eligible = [
        observation
        for observation in observations
        if earliest_s <= float(observation.timestamp_s) <= target_s
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda observation: (
            float(observation.timestamp_s),
            float(observation.confidence),
        ),
    )


def _existing_candidate_values(
    prior: np.ndarray | None,
    candidates: Sequence[Candidate],
    cfg: GuitarConfig,
) -> np.ndarray | None:
    if prior is None:
        return np.ones(len(candidates), dtype=np.float64)

    array = np.asarray(prior, dtype=np.float64)
    if array.shape == (cfg.max_fret + 1,):
        values = np.asarray([array[candidate.fret] for candidate in candidates])
    elif array.shape == (cfg.n_strings, cfg.max_fret + 1):
        values = np.asarray(
            [array[candidate.string_idx, candidate.fret] for candidate in candidates]
        )
    else:
        return None

    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        return None
    if float(values.sum()) <= 0.0:
        return np.ones(len(candidates), dtype=np.float64)
    return values


__all__ = [
    "MAX_POSITION_LOG_BONUS",
    "MAX_POSITION_OBSERVATION_CONFIDENCE",
    "MIN_POSITION_OBSERVATION_CONFIDENCE",
    "POSITION_OBSERVATION_LEAD_S",
    "POSITION_OBSERVATION_LOOKBACK_S",
    "VALID_POSITION_OBSERVATION_STATES",
    "apply_position_window_priors",
]
