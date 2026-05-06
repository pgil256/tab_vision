"""Cluster-level Viterbi decode — Phase 5 deliverable.

Public entrypoint: ``fuse(events, fingerings, cfg, session, lambda_vision)``.

Each "step" in the DP is a chord cluster (often a singleton — an isolated
event). For each cluster, :func:`tabvision.fusion.chord.enumerate_chord_states`
produces the per-string-monophony + hand-span-feasible ordered tuples of
candidates. Emission for a state is the sum of per-event emission costs
(:func:`tabvision.fusion.playability.emission_cost`); transitions between
clusters use :func:`tabvision.fusion.chord.chord_anchor` to pick a
representative position for the playability transition cost.

The single-line Viterbi behaviour is the size-1-cluster degenerate case
of this same DP — no separate code path.

See ``docs/plans/2026-05-06-phase5-fusion-design.md`` §3 for the state
spaces and §2 for the cost decomposition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tabvision.fusion import chord, playability
from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.types import (
    AudioEvent,
    FrameFingering,
    GuitarConfig,
    SessionConfig,
    TabEvent,
)


def fuse(
    events: Sequence[AudioEvent],
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    lambda_vision: float = 1.0,
) -> list[TabEvent]:
    """Decode ``AudioEvent``s into ``TabEvent``s via cluster Viterbi.

    Parameters
    ----------
    events:
        Audio events. Out-of-range pitches (no playable candidate under
        ``cfg``) are dropped — no phantom notes emitted.
    fingerings:
        Per-frame fingerings from Phase 4. Empty / all-zero is treated
        as audio-only.
    cfg:
        Instrument config (tuning, capo, max_fret).
    session:
        Recording session metadata; reserved for future use.
    lambda_vision:
        Mixing weight for the vision-evidence term. ``0.0`` disables
        vision entirely; ``1.0`` is the default; higher values lean more
        heavily on the fingertip-to-fret posterior.

    Returns
    -------
    list[TabEvent]
        One ``TabEvent`` per surviving event, ordered by ``onset_s``.
    """
    if cfg is None:
        cfg = GuitarConfig()
    if session is None:
        session = SessionConfig()
    del session  # not consumed by Phase 5; preserves signature for callers.

    if not events:
        return []

    # Drop out-of-range pitches before clustering so the cluster shape
    # reflects what's actually decodable.
    valid_events = [ev for ev in events if candidate_positions(ev.pitch_midi, cfg)]
    if not valid_events:
        return []

    clusters = chord.cluster_events(valid_events)
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]] = []
    for cluster in clusters:
        states = chord.enumerate_chord_states(cluster, cfg)
        if states:
            cluster_data.append((cluster, states))

    if not cluster_data:
        return []

    return _viterbi_clusters(cluster_data, fingerings, cfg, lambda_vision)


def _viterbi_clusters(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig,
    lambda_vision: float,
) -> list[TabEvent]:
    """Cluster-level Viterbi DP. Worst case ``O(N · S^2)`` for ``N``
    clusters with ``S`` states each."""

    def state_emission(cluster: list[AudioEvent], state: tuple[Candidate, ...]) -> float:
        total = 0.0
        for ev, c in zip(cluster, state, strict=True):
            f = playability.find_fingering_at(ev.onset_s, fingerings)
            total += playability.emission_cost(c, ev, f, cfg, lambda_vision=lambda_vision)
        return total

    n = len(cluster_data)
    cost: list[list[float]] = [[] for _ in range(n)]
    backptr: list[list[int]] = [[] for _ in range(n)]

    cluster0, states0 = cluster_data[0]
    cost[0] = [state_emission(cluster0, st) for st in states0]
    backptr[0] = [-1] * len(states0)

    for i in range(1, n):
        cluster_i, states_i = cluster_data[i]
        prev_states = cluster_data[i - 1][1]
        cost[i] = [math.inf] * len(states_i)
        backptr[i] = [-1] * len(states_i)
        for si, state in enumerate(states_i):
            emit = state_emission(cluster_i, state)
            anchor_curr = chord.chord_anchor(state)
            for pi, prev_state in enumerate(prev_states):
                anchor_prev = chord.chord_anchor(prev_state)
                trans = playability.transition_cost(anchor_prev, anchor_curr, cfg)
                total = cost[i - 1][pi] + trans + emit
                if total < cost[i][si]:
                    cost[i][si] = total
                    backptr[i][si] = pi

    # Backtrack from the cheapest terminal state.
    final = cost[n - 1]
    last_idx = min(range(len(final)), key=lambda j: final[j])
    picks_idx = [0] * n
    picks_idx[n - 1] = last_idx
    for i in range(n - 1, 0, -1):
        picks_idx[i - 1] = backptr[i][picks_idx[i]]

    out: list[TabEvent] = []
    for i, (cluster, states) in enumerate(cluster_data):
        state = states[picks_idx[i]]
        for ev, c in zip(cluster, state, strict=True):
            out.append(
                TabEvent(
                    onset_s=ev.onset_s,
                    duration_s=max(0.0, ev.offset_s - ev.onset_s),
                    string_idx=c.string_idx,
                    fret=c.fret,
                    pitch_midi=ev.pitch_midi,
                    confidence=ev.confidence,
                    techniques=ev.tags,
                )
            )
    return out


__all__ = ["fuse"]
