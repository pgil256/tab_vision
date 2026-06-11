"""Utilities for making noisy video evidence safe for fusion.

The core fusion contract stays simple: callers still pass ``FrameFingering``
objects.  These helpers prepare those objects by choosing a canonical
orientation, voting over nearby frames, and dropping weak evidence before it
can pull a strong audio decode off course.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from tabvision.fusion.candidates import candidate_positions
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig

EPS = 1e-12
DEFAULT_N_FINGERS = 4


@dataclass(frozen=True)
class Orientation:
    """Canonical-axis flips to apply to a fingering posterior."""

    name: str
    flip_string: bool = False
    flip_fret: bool = False


ORIENTATIONS: tuple[Orientation, ...] = (
    Orientation("none"),
    Orientation("flip-fret", flip_fret=True),
    Orientation("flip-string", flip_string=True),
    Orientation("flip-both", flip_string=True, flip_fret=True),
)
ORIENTATION_BY_NAME: Mapping[str, Orientation] = {o.name: o for o in ORIENTATIONS}


def empty_fingering(
    t: float,
    cfg: GuitarConfig,
    *,
    n_fingers: int = DEFAULT_N_FINGERS,
) -> FrameFingering:
    """Evidence-free fingering at ``t``."""

    return FrameFingering(
        t=t,
        finger_pos_logits=np.zeros((n_fingers, cfg.n_strings, cfg.max_fret + 1), dtype=np.float64),
        homography_confidence=0.0,
    )


def apply_orientation(logits: np.ndarray, orientation: Orientation) -> np.ndarray:
    """Return a copy of ``logits`` with canonical axes flipped as requested."""

    out = np.asarray(logits).copy()
    if orientation.flip_string:
        out = out[:, ::-1, :]
    if orientation.flip_fret:
        out = out[:, :, ::-1]
    return out


def orient_fingering(fingering: FrameFingering, orientation: Orientation) -> FrameFingering:
    """Apply ``orientation`` to a ``FrameFingering`` without mutating it."""

    return replace(
        fingering,
        finger_pos_logits=apply_orientation(fingering.finger_pos_logits, orientation),
    )


def combine_fingerings(
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig,
    *,
    t: float,
) -> FrameFingering:
    """Average nearby frame posteriors into one onset-aligned fingering."""

    usable = [
        f
        for f in fingerings
        if f.homography_confidence > 0.0
        and f.finger_pos_logits is not None
        and f.finger_pos_logits.size > 0
        and (f.finger_pos_logits != 0).any()
    ]
    if not usable:
        return empty_fingering(t, cfg)

    marginals = np.array([f.marginal_string_fret() for f in usable], dtype=np.float64)
    avg = marginals.mean(axis=0)
    total = float(avg.sum())
    if not math.isfinite(total) or total <= 0.0:
        return empty_fingering(t, cfg, n_fingers=usable[0].finger_pos_logits.shape[0])

    avg /= total
    n_fingers = usable[0].finger_pos_logits.shape[0]
    logits = np.full((n_fingers, cfg.n_strings, cfg.max_fret + 1), -30.0, dtype=np.float64)
    logits[0] = np.log(np.maximum(avg, EPS))
    confidence = float(np.mean([f.homography_confidence for f in usable]))
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=confidence)


def candidate_support(event: AudioEvent, fingering: FrameFingering, cfg: GuitarConfig) -> float:
    """Probability mass on string/fret candidates compatible with ``event`` pitch."""

    candidates = candidate_positions(event.pitch_midi, cfg)
    if not candidates or fingering.homography_confidence <= 0.0:
        return 0.0
    marginal = fingering.marginal_string_fret()
    return float(sum(marginal[c.string_idx, c.fret] for c in candidates))


def candidate_probabilities(
    event: AudioEvent, fingering: FrameFingering, cfg: GuitarConfig
) -> list[tuple[int, int, float]]:
    """Return pitch-compatible candidate probabilities sorted high to low."""

    candidates = candidate_positions(event.pitch_midi, cfg)
    if not candidates or fingering.homography_confidence <= 0.0:
        return []
    marginal = fingering.marginal_string_fret()
    probs = [(c.string_idx, c.fret, float(marginal[c.string_idx, c.fret])) for c in candidates]
    probs.sort(key=lambda row: row[2], reverse=True)
    return probs


def choose_orientation(
    raw_by_event: Sequence[Sequence[FrameFingering]],
    events: Sequence[AudioEvent],
    cfg: GuitarConfig,
) -> tuple[Orientation, dict[str, float]]:
    """Pick the axis orientation whose video mass best agrees with audio pitch.

    This uses only audio pitch candidates, not tab labels, so it is safe for
    real clips where ground truth is unavailable.
    """

    scores: dict[str, float] = {}
    for orientation in ORIENTATIONS:
        score = 0.0
        for event, raw_fingerings in zip(events, raw_by_event, strict=False):
            oriented = [orient_fingering(f, orientation) for f in raw_fingerings]
            voted = combine_fingerings(oriented, cfg, t=event.onset_s)
            support = candidate_support(event, voted, cfg)
            score += math.log(max(support, EPS)) * max(0.05, voted.homography_confidence)
        scores[orientation.name] = score
    best = max(ORIENTATIONS, key=lambda o: scores[o.name])
    return best, scores


def gate_fingering_to_audio(
    event: AudioEvent,
    fingering: FrameFingering,
    cfg: GuitarConfig,
    *,
    min_homography_confidence: float = 0.1,
    min_candidate_support: float = 0.02,
    min_best_ratio: float = 1.2,
) -> FrameFingering:
    """Drop video evidence that is too weak to help this audio event."""

    if fingering.homography_confidence < min_homography_confidence:
        return empty_fingering(fingering.t, cfg, n_fingers=fingering.finger_pos_logits.shape[0])

    probs = candidate_probabilities(event, fingering, cfg)
    if not probs:
        return empty_fingering(fingering.t, cfg, n_fingers=fingering.finger_pos_logits.shape[0])

    support = sum(p for *_sf, p in probs)
    if support < min_candidate_support:
        return empty_fingering(fingering.t, cfg, n_fingers=fingering.finger_pos_logits.shape[0])

    default = candidate_positions(event.pitch_midi, cfg)[0]
    default_prob = next(
        (
            p
            for string_idx, fret, p in probs
            if string_idx == default.string_idx and fret == default.fret
        ),
        0.0,
    )
    best_string, best_fret, best_prob = probs[0]
    if (best_string, best_fret) != (default.string_idx, default.fret):
        if best_prob < max(default_prob * min_best_ratio, EPS):
            return empty_fingering(fingering.t, cfg, n_fingers=fingering.finger_pos_logits.shape[0])

    return fingering


__all__ = [
    "Orientation",
    "ORIENTATIONS",
    "ORIENTATION_BY_NAME",
    "apply_orientation",
    "candidate_probabilities",
    "candidate_support",
    "choose_orientation",
    "combine_fingerings",
    "empty_fingering",
    "gate_fingering_to_audio",
    "orient_fingering",
]
