"""Causal, bounded FretCam finger-contact evidence for audio tab decoding.

This is the finger-level sibling of :mod:`tabvision.fusion.position_window_prior`.
Where that module supports a whole six-fret window because FretCam's position
estimator only reports a coarse hand position, this one supports the specific
``(string, fret)`` pairs the detection chain says are under a finger.

Two things make it a different proposition from the rejected exact-string video
posterior that :func:`tabvision.fusion.playability.emission_cost` applies on the
legacy route:

* the likelihood advantage is **capped**, so vision can break a tie but never
  veto a confident audio prior; and
* the cap is set from a **measured** likelihood ratio rather than chosen.

The measurement is
``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``: over GAPS
clean-12, contacts name the gold ``(string, fret)`` on 13.9% of covered
audio-wrong notes against 1.6% for the decoder's own wrong choice, a ratio of
8.45 (2.13 nats). :data:`MAX_CONTACT_LOG_BONUS` floors that to 2.0 nats.

Two caveats that measurement carries, both deliberate:

* the ratio is conditioned on audio-wrong notes, so using it as an odds
  multiplier over all ambiguous notes is an approximation, not exact Bayes;
* the same report measures **exposure** — 68 currently-correct notes where the
  contacts name a rival position and not the gold, against 93 rescues. This
  channel is not free upside, and the cap is what keeps the exposure bounded.

Open and capoed strings receive unconditional support. A contact set can never
name an open string, so without the carve-out this prior would demote every
open string merely because the fretting hand is somewhere else — the same
reasoning as ``position_window_prior``'s ``{0, cfg.capo}`` union.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from tabvision.fusion.candidates import candidate_positions
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.position import FingerContactObservation

CONTACT_OBSERVATION_LEAD_S = 0.03
"""Read vision this many seconds before the audio onset."""

CONTACT_OBSERVATION_LOOKBACK_S = 0.15
"""Maximum age of a contact observation relative to the pre-onset target."""

MIN_CONTACT_OBSERVATION_CONFIDENCE = 0.0
"""Contacts are not gated on the position estimator's lock confidence.

The estimator's threshold exists so a HUD does not flicker a number at a human.
Fusion wants the opposite — soft evidence, continuously, weighed by the
decoder. Gating contacts the same way is what reduced the shipped window to
2.6% coverage.
"""

MAX_CONTACT_LOG_BONUS = 2.0
"""Maximum vision log-likelihood advantage between two candidates, in nats.

Floored from the measured 2.13 nats (likelihood ratio 8.45). Fixed before any
end-to-end run; not swept against Tab F1.
"""


def apply_contact_priors(
    events: Sequence[AudioEvent],
    observations: Sequence[FingerContactObservation],
    cfg: GuitarConfig,
    *,
    vision_weight: float = 1.0,
) -> list[AudioEvent]:
    """Attach bounded, causal finger-contact likelihoods to audio events.

    For an event at time ``t`` the latest valid observation in
    ``[t - lead - lookback, t - lead]`` is used. Only pitches with multiple
    playable assignments are changed. Candidates whose ``(string, fret)`` is
    under a finger, plus open/capoed candidates, receive a log bonus of at most
    :data:`MAX_CONTACT_LOG_BONUS`. Existing priors are multiplied by that
    likelihood and never mutated.

    Events retain object identity whenever vision contributes no
    discriminating evidence.
    """
    weight = _finite_non_negative(vision_weight)
    if not observations or weight == 0.0:
        return list(events)

    valid = [obs for obs in observations if _is_valid(obs, cfg)]
    if not valid:
        return list(events)
    valid.sort(key=lambda obs: float(obs.timestamp_s))
    timestamps = [float(obs.timestamp_s) for obs in valid]

    output: list[AudioEvent] = []
    for event in events:
        candidates = candidate_positions(event.pitch_midi, cfg)
        if len(candidates) <= 1 or not math.isfinite(float(event.onset_s)):
            output.append(event)
            continue

        observation = _latest_causal(valid, timestamps, float(event.onset_s))
        if observation is None:
            output.append(event)
            continue

        named = frozenset(observation.positions)
        supported = np.asarray(
            [
                (candidate.string_idx, candidate.fret) in named
                or candidate.fret == 0
                or candidate.fret == cfg.capo
                for candidate in candidates
            ],
            dtype=np.bool_,
        )
        if bool(np.all(supported)) or not bool(np.any(supported)):
            output.append(event)
            continue

        existing = _existing_values(event.fret_prior, candidates, cfg)
        if existing is None:
            output.append(event)
            continue
        if float(existing[supported].sum()) <= 0.0 or float(existing[~supported].sum()) <= 0.0:
            # One side is already eliminated; multiplying the survivor would
            # renormalize to the same distribution. Preserve the exact event so
            # the pipeline's "notes affected" count stays honest.
            output.append(event)
            continue

        log_bonus = min(
            MAX_CONTACT_LOG_BONUS,
            weight * MAX_CONTACT_LOG_BONUS * float(observation.confidence),
        )
        if log_bonus <= 0.0:
            output.append(event)
            continue
        likelihood = np.ones(len(candidates), dtype=np.float64)
        likelihood[supported] = math.exp(log_bonus)
        combined = existing * likelihood
        total = float(combined.sum())
        if not math.isfinite(total) or total <= 0.0:
            output.append(event)
            continue

        prior = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
        for candidate, probability in zip(candidates, combined / total, strict=True):
            prior[candidate.string_idx, candidate.fret] = float(probability)
        output.append(replace(event, fret_prior=prior))

    return output


def _finite_non_negative(value: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, weight) if math.isfinite(weight) else 0.0


def _is_valid(observation: FingerContactObservation, cfg: GuitarConfig) -> bool:
    try:
        timestamp = float(observation.timestamp_s)
        confidence = float(observation.confidence)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(timestamp) or not math.isfinite(confidence):
        return False
    if not MIN_CONTACT_OBSERVATION_CONFIDENCE <= confidence <= 1.0:
        return False
    if not observation.positions:
        return False
    for entry in observation.positions:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return False
        string_idx, fret = entry
        if isinstance(string_idx, bool) or isinstance(fret, bool):
            return False
        if not isinstance(string_idx, int) or not isinstance(fret, int):
            return False
        if not 0 <= string_idx < cfg.n_strings or not 0 <= fret <= cfg.max_fret:
            return False
    return True


def _latest_causal(
    observations: Sequence[FingerContactObservation],
    timestamps: Sequence[float],
    onset_s: float,
) -> FingerContactObservation | None:
    target = onset_s - CONTACT_OBSERVATION_LEAD_S
    earliest = target - CONTACT_OBSERVATION_LOOKBACK_S
    index = bisect.bisect_right(timestamps, target) - 1
    if index < 0 or timestamps[index] < earliest:
        return None
    return observations[index]


def _existing_values(
    prior: np.ndarray | None,
    candidates: Sequence[object],
    cfg: GuitarConfig,
) -> np.ndarray | None:
    if prior is None:
        return np.ones(len(candidates), dtype=np.float64)

    array = np.asarray(prior, dtype=np.float64)
    if array.shape == (cfg.max_fret + 1,):
        values = np.asarray([array[c.fret] for c in candidates])  # type: ignore[attr-defined]
    elif array.shape == (cfg.n_strings, cfg.max_fret + 1):
        values = np.asarray(
            [array[c.string_idx, c.fret] for c in candidates]  # type: ignore[attr-defined]
        )
    else:
        return None

    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        return None
    if float(values.sum()) <= 0.0:
        return np.ones(len(candidates), dtype=np.float64)
    return values


__all__ = [
    "CONTACT_OBSERVATION_LEAD_S",
    "CONTACT_OBSERVATION_LOOKBACK_S",
    "MAX_CONTACT_LOG_BONUS",
    "MIN_CONTACT_OBSERVATION_CONFIDENCE",
    "apply_contact_priors",
]
