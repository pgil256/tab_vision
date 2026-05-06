"""Chord cluster grouping + chord-state machinery — Phase 5 deliverable.

A *chord cluster* is a maximal run of consecutive ``AudioEvent``s whose
adjacent onset gaps are all ≤ :data:`CHORD_MAX_GAP_S` (80 ms by default).
Within a cluster, decoding picks an ordered tuple of ``(string, fret)``
candidates — one per event — subject to two structural constraints:

- **Per-string monophony**: no two events share a string.
- **Hand-span**: ``max(pressed_fret) - min(pressed_fret) ≤ MAX_HAND_SPAN``
  (open strings are exempt — fret 0 doesn't constrain the fretting hand).

This module is pure machinery — clustering, state enumeration, anchor
selection. The cluster-level Viterbi DP that consumes these states lives
in :mod:`tabvision.fusion.viterbi`.

See ``docs/plans/2026-05-06-phase5-fusion-design.md`` §3.2 and SPEC.md §5.
"""

from __future__ import annotations

from typing import Sequence

from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.fusion.playability import MAX_HAND_SPAN
from tabvision.types import AudioEvent, GuitarConfig

CHORD_MAX_GAP_S = 0.080
"""Maximum onset gap (seconds) between consecutive events to count as one
chord cluster. SPEC §5 calls this "≤ 80 ms apart"."""


def cluster_events(
    events: Sequence[AudioEvent],
    max_gap_s: float = CHORD_MAX_GAP_S,
) -> list[list[AudioEvent]]:
    """Group events into chord clusters.

    Chain semantics: events ``i`` and ``i+1`` (sorted by onset) join the
    same cluster iff ``events[i+1].onset_s - events[i].onset_s ≤ max_gap_s``.
    A cluster therefore can span more than ``max_gap_s`` overall when the
    individual pairwise gaps remain bounded.
    """
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e.onset_s)
    clusters: list[list[AudioEvent]] = [[sorted_events[0]]]
    for ev in sorted_events[1:]:
        if ev.onset_s - clusters[-1][-1].onset_s <= max_gap_s:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])
    return clusters


def enumerate_chord_states(
    events: Sequence[AudioEvent],
    cfg: GuitarConfig,
) -> list[tuple[Candidate, ...]]:
    """All valid (monophony + hand-span) ordered tuples of candidates.

    Builds the state set incrementally to keep the worst-case bounded by
    the constraint-pruned size at each step rather than the raw cartesian
    product (``K^m``). Returns an empty list if any event has no
    candidates — the caller is expected to filter out-of-range events
    upstream so the cluster shape stays consistent with the input order.
    """
    if not events:
        return []

    per_event_candidates = [
        candidate_positions(ev.pitch_midi, cfg) for ev in events
    ]
    if any(not cands for cands in per_event_candidates):
        return []

    states: list[tuple[Candidate, ...]] = [
        (c,) for c in per_event_candidates[0]
    ]
    for k in range(1, len(events)):
        next_states: list[tuple[Candidate, ...]] = []
        for state in states:
            used_strings = {c.string_idx for c in state}
            pressed = [c.fret for c in state if c.fret > 0]
            for c in per_event_candidates[k]:
                if c.string_idx in used_strings:
                    continue
                new_pressed = pressed + ([c.fret] if c.fret > 0 else [])
                if new_pressed:
                    span = max(new_pressed) - min(new_pressed)
                    if span > MAX_HAND_SPAN:
                        continue
                next_states.append(state + (c,))
        states = next_states
        if not states:
            return []
    return states


def chord_anchor(state: tuple[Candidate, ...]) -> Candidate:
    """The 'anchor' candidate used as the state's representative for
    inter-cluster transition costs.

    Defined as the lowest-fret *pressed* note (fret > 0) — the natural
    centre of the fretting hand. If all notes are open, the first
    candidate is returned (any choice is equivalent because all pressed
    frets are 0 and transition cost depends on Δfret).
    """
    pressed = [c for c in state if c.fret > 0]
    if not pressed:
        return state[0]
    return min(pressed, key=lambda c: (c.fret, c.string_idx))


__all__ = [
    "CHORD_MAX_GAP_S",
    "cluster_events",
    "enumerate_chord_states",
    "chord_anchor",
]
