"""Melodic-segment position prior for fast single-note runs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.fusion.playability import transition_cost
from tabvision.types import AudioEvent, GuitarConfig


def find_melodic_segments(
    events: Sequence[AudioEvent],
    *,
    min_events: int = 4,
    max_gap_s: float = 0.45,
    max_pitch_step: int = 3,
) -> list[list[int]]:
    """Return index groups that look like fast melodic runs."""
    if len(events) < min_events:
        return []

    ordered = sorted(enumerate(events), key=lambda item: item[1].onset_s)
    segments: list[list[int]] = []
    current = [ordered[0][0]]

    for (prev_idx, prev), (idx, ev) in zip(ordered, ordered[1:], strict=False):
        del prev_idx
        gap = ev.onset_s - prev.onset_s
        pitch_step = abs(ev.pitch_midi - prev.pitch_midi)
        if gap <= max_gap_s and pitch_step <= max_pitch_step:
            current.append(idx)
        else:
            if len(current) >= min_events:
                segments.append(current)
            current = [idx]

    if len(current) >= min_events:
        segments.append(current)

    return segments


def apply_melodic_segment_prior(
    events: Sequence[AudioEvent],
    cfg: GuitarConfig | None = None,
    *,
    min_events: int = 4,
    max_gap_s: float = 0.45,
    max_pitch_step: int = 3,
    prior_strength: float = 0.85,
) -> list[AudioEvent]:
    """Attach fret priors that keep fast melodic runs in a playable region."""
    if cfg is None:
        cfg = GuitarConfig()
    if not events:
        return []

    out = list(events)
    for segment in find_melodic_segments(
        out,
        min_events=min_events,
        max_gap_s=max_gap_s,
        max_pitch_step=max_pitch_step,
    ):
        path = _decode_segment([out[index] for index in segment], cfg)
        for index, candidate in zip(segment, path, strict=True):
            segment_prior = _prior_for_candidate(
                candidate,
                out[index].pitch_midi,
                cfg,
                prior_strength,
            )
            existing = out[index].fret_prior
            if existing is not None and getattr(existing, "shape", ()) == segment_prior.shape:
                combined = np.asarray(existing, dtype=np.float64) * 0.35 + segment_prior * 0.65
                total = float(combined.sum())
                if total > 0:
                    segment_prior = combined / total
            out[index] = replace(out[index], fret_prior=segment_prior)

    return out


def _decode_segment(events: Sequence[AudioEvent], cfg: GuitarConfig) -> list[Candidate]:
    candidate_steps = [candidate_positions(event.pitch_midi, cfg) for event in events]
    if any(not candidates for candidates in candidate_steps):
        return [
            candidates[0] if candidates else Candidate(string_idx=0, fret=0)
            for candidates in candidate_steps
        ]

    target_window = _infer_target_fret_window(events, cfg)
    costs: list[list[float]] = []
    backptrs: list[list[int]] = []
    first_costs = [_emission_cost(candidate, target_window) for candidate in candidate_steps[0]]
    costs.append(first_costs)
    backptrs.append([-1] * len(first_costs))

    for step_index in range(1, len(candidate_steps)):
        prev_candidates = candidate_steps[step_index - 1]
        candidates = candidate_steps[step_index]
        step_costs = [math.inf] * len(candidates)
        step_backptrs = [-1] * len(candidates)
        for current_index, current in enumerate(candidates):
            emit = _emission_cost(current, target_window)
            for prev_index, previous in enumerate(prev_candidates):
                total = (
                    costs[step_index - 1][prev_index]
                    + transition_cost(previous, current, cfg)
                    + _hand_span_cost(previous, current)
                    + emit
                )
                if total < step_costs[current_index]:
                    step_costs[current_index] = total
                    step_backptrs[current_index] = prev_index
        costs.append(step_costs)
        backptrs.append(step_backptrs)

    last = min(range(len(costs[-1])), key=lambda index: costs[-1][index])
    path_indexes = [0] * len(candidate_steps)
    path_indexes[-1] = last
    for step_index in range(len(candidate_steps) - 1, 0, -1):
        path_indexes[step_index - 1] = backptrs[step_index][path_indexes[step_index]]

    return [
        candidates[index] for candidates, index in zip(candidate_steps, path_indexes, strict=True)
    ]


def _infer_target_fret_window(
    events: Sequence[AudioEvent],
    cfg: GuitarConfig,
    *,
    window_width: int = 3,
) -> tuple[int, int] | None:
    """Infer the compact neck region that best covers a long scalar run."""
    if len(events) < 8:
        return None
    pitch_span = max(event.pitch_midi for event in events) - min(
        event.pitch_midi for event in events
    )
    if pitch_span < 12:
        return None

    best_window: tuple[int, int] | None = None
    best_score = math.inf
    for start in range(cfg.capo, cfg.max_fret - window_width + 1):
        end = start + window_width
        score = 0.02 * start
        for event in events:
            candidates = candidate_positions(event.pitch_midi, cfg)
            if not candidates:
                score += 100.0
                continue
            score += min(
                _distance_to_window(candidate.fret, start, end) for candidate in candidates
            )
        if score < best_score:
            best_score = score
            best_window = (start, end)

    return best_window


def _distance_to_window(fret: int, start: int, end: int) -> int:
    if fret < start:
        return start - fret
    if fret > end:
        return fret - end
    return 0


def _emission_cost(candidate: Candidate, target_window: tuple[int, int] | None) -> float:
    # Mildly discourage very high alternatives without forcing open strings.
    high_fret_penalty = max(0, candidate.fret - 12) * 0.18
    moderate_position_bonus = -0.2 if 4 <= candidate.fret <= 12 else 0.0
    window_penalty = 0.0
    if target_window is not None:
        window_penalty = 1.6 * _distance_to_window(
            candidate.fret,
            target_window[0],
            target_window[1],
        )
    return 0.03 * candidate.fret + high_fret_penalty + moderate_position_bonus + window_penalty


def _hand_span_cost(
    previous: Candidate,
    current: Candidate,
) -> float:
    delta = abs(current.fret - previous.fret)
    if delta <= 5:
        return 0.0
    return (delta - 5) * 3.0


def _prior_for_candidate(
    candidate: Candidate,
    pitch_midi: int,
    cfg: GuitarConfig,
    prior_strength: float,
) -> np.ndarray:
    prior = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    candidates = candidate_positions(pitch_midi, cfg)
    if not candidates:
        return prior

    residual = max(0.0, 1.0 - prior_strength)
    for other in candidates:
        prior[other.string_idx, other.fret] = residual / len(candidates)
    prior[candidate.string_idx, candidate.fret] += prior_strength
    total = float(prior.sum())
    return prior / total if total > 0 else prior


__all__ = [
    "apply_melodic_segment_prior",
    "find_melodic_segments",
]
