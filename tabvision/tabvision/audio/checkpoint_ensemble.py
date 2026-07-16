"""Deterministic event matching for the gate-passed Phase 3 checkpoint ensemble.

The default production route remains ``guitar_gaps``. The explicit registered
ensemble and its evaluator share this implementation so matching semantics
cannot drift between measurement and inference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from tabvision.types import AudioEvent

DEFAULT_MATCH_TOLERANCE_S = 0.05

EventScore = Callable[[Literal["gaps", "fl"], int, AudioEvent], float]


@dataclass(frozen=True)
class EventMatch:
    """One deterministic cross-checkpoint match."""

    gaps_index: int
    fl_index: int
    onset_delta_s: float


@dataclass(frozen=True)
class CheckpointAlignment:
    """Same-pitch agreements plus one-to-one remaining disagreements."""

    agreements: tuple[EventMatch, ...]
    disagreements: tuple[EventMatch, ...]
    gaps_only: tuple[int, ...]
    fl_only: tuple[int, ...]


def emitted_pitch_probability(event: AudioEvent) -> float:
    """Return the real emitted-pitch posterior, or zero when unavailable."""

    logits = event.pitch_logits
    pitch = int(event.pitch_midi)
    if logits is None or not 0 <= pitch < len(logits):
        return 0.0
    logit = float(logits[pitch])
    if not math.isfinite(logit):
        return 0.0
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def match_same_pitch(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
) -> tuple[EventMatch, ...]:
    """Greedily match closest same-pitch events, inclusive at the boundary."""

    _validate_tolerance(tolerance_s)
    candidates = _candidate_pairs(
        gaps_events,
        fl_events,
        tolerance_s=tolerance_s,
        require_same_pitch=True,
    )
    return _select_pairs(candidates)


def align_checkpoints(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
) -> CheckpointAlignment:
    """Align agreements first, then pair remaining overlapping events.

    The second pass is used only by the two selector conditions.  It pairs
    closest onsets and then closest pitches, so simultaneous chord notes are
    deterministic and one-to-one rather than collapsed into one onset group.
    """

    agreements = match_same_pitch(gaps_events, fl_events, tolerance_s=tolerance_s)
    used_gaps = {match.gaps_index for match in agreements}
    used_fl = {match.fl_index for match in agreements}
    disagreement_candidates = _candidate_pairs(
        gaps_events,
        fl_events,
        tolerance_s=tolerance_s,
        require_same_pitch=False,
        excluded_gaps=used_gaps,
        excluded_fl=used_fl,
    )
    disagreements = _select_pairs(disagreement_candidates)
    used_gaps.update(match.gaps_index for match in disagreements)
    used_fl.update(match.fl_index for match in disagreements)
    return CheckpointAlignment(
        agreements=agreements,
        disagreements=disagreements,
        gaps_only=tuple(index for index in range(len(gaps_events)) if index not in used_gaps),
        fl_only=tuple(index for index in range(len(fl_events)) if index not in used_fl),
    )


def intersection_events(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
) -> tuple[AudioEvent, ...]:
    """Return GAPS events that the FL checkpoint agrees with, unchanged."""

    matches = match_same_pitch(gaps_events, fl_events, tolerance_s=tolerance_s)
    return _sort_events(gaps_events[match.gaps_index] for match in matches)


def union_events(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
) -> tuple[AudioEvent, ...]:
    """Return all GAPS events plus only non-agreeing FL events.

    Same-pitch cross-checkpoint duplicates collapse to the original GAPS
    object.  Within-checkpoint duplicates are retained because suppressing a
    production GAPS event would violate the agreement-preservation rule.
    """

    matches = match_same_pitch(gaps_events, fl_events, tolerance_s=tolerance_s)
    matched_fl = {match.fl_index for match in matches}
    return _sort_events(
        [*gaps_events, *(event for index, event in enumerate(fl_events) if index not in matched_fl)]
    )


def select_events(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    score: EventScore,
    threshold: float = 0.5,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
) -> tuple[AudioEvent, ...]:
    """Preserve agreements and select calibrated checkpoint disagreements.

    Paired disagreements contribute their higher-scoring event when it meets
    ``threshold``.  Isolated events are independently retained at the same
    threshold.  Ties deliberately prefer GAPS, the production checkpoint.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    alignment = align_checkpoints(gaps_events, fl_events, tolerance_s=tolerance_s)
    selected: list[AudioEvent] = [gaps_events[match.gaps_index] for match in alignment.agreements]
    for match in alignment.disagreements:
        gaps_score = _checked_score(score("gaps", match.gaps_index, gaps_events[match.gaps_index]))
        fl_score = _checked_score(score("fl", match.fl_index, fl_events[match.fl_index]))
        if gaps_score >= fl_score and gaps_score >= threshold:
            selected.append(gaps_events[match.gaps_index])
        elif fl_score >= threshold:
            selected.append(fl_events[match.fl_index])
    for index in alignment.gaps_only:
        if _checked_score(score("gaps", index, gaps_events[index])) >= threshold:
            selected.append(gaps_events[index])
    for index in alignment.fl_only:
        if _checked_score(score("fl", index, fl_events[index])) >= threshold:
            selected.append(fl_events[index])
    return _sort_events(selected)


def _candidate_pairs(
    gaps_events: Sequence[AudioEvent],
    fl_events: Sequence[AudioEvent],
    *,
    tolerance_s: float,
    require_same_pitch: bool,
    excluded_gaps: set[int] | None = None,
    excluded_fl: set[int] | None = None,
) -> list[tuple[float, int, float, float, int, int]]:
    excluded_gaps = excluded_gaps or set()
    excluded_fl = excluded_fl or set()
    candidates: list[tuple[float, int, float, float, int, int]] = []
    epsilon = 1.0e-9
    for gaps_index, gaps_event in enumerate(gaps_events):
        if gaps_index in excluded_gaps:
            continue
        for fl_index, fl_event in enumerate(fl_events):
            if fl_index in excluded_fl:
                continue
            if require_same_pitch and gaps_event.pitch_midi != fl_event.pitch_midi:
                continue
            if not require_same_pitch and gaps_event.pitch_midi == fl_event.pitch_midi:
                continue
            delta = abs(gaps_event.onset_s - fl_event.onset_s)
            if delta <= tolerance_s + epsilon:
                candidates.append(
                    (
                        delta,
                        abs(gaps_event.pitch_midi - fl_event.pitch_midi),
                        min(gaps_event.onset_s, fl_event.onset_s),
                        max(gaps_event.onset_s, fl_event.onset_s),
                        gaps_index,
                        fl_index,
                    )
                )
    candidates.sort()
    return candidates


def _select_pairs(
    candidates: Sequence[tuple[float, int, float, float, int, int]],
) -> tuple[EventMatch, ...]:
    used_gaps: set[int] = set()
    used_fl: set[int] = set()
    selected: list[EventMatch] = []
    for delta, _pitch_delta, _first_onset, _second_onset, gaps_index, fl_index in candidates:
        if gaps_index in used_gaps or fl_index in used_fl:
            continue
        used_gaps.add(gaps_index)
        used_fl.add(fl_index)
        selected.append(EventMatch(gaps_index, fl_index, delta))
    selected.sort(key=lambda match: (match.gaps_index, match.fl_index))
    return tuple(selected)


def _sort_events(events: Iterable[AudioEvent]) -> tuple[AudioEvent, ...]:
    materialized = list(events)
    return tuple(
        event
        for _index, event in sorted(
            enumerate(materialized),
            key=lambda item: (item[1].onset_s, item[1].pitch_midi, item[0]),
        )
    )


def _checked_score(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("event scores must be finite probabilities")
    return numeric


def _validate_tolerance(tolerance_s: float) -> None:
    if not math.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError("tolerance_s must be finite and non-negative")


__all__ = [
    "CheckpointAlignment",
    "DEFAULT_MATCH_TOLERANCE_S",
    "EventMatch",
    "align_checkpoints",
    "emitted_pitch_probability",
    "intersection_events",
    "match_same_pitch",
    "select_events",
    "union_events",
]
