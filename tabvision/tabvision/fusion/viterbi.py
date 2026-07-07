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

from tabvision.fusion import chord, chord_shapes, playability
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
    clusters with ``S`` states each.

    Runs a forward *and* backward pass so each note carries a **string-flip
    margin** (B4): the extra cost of the cheapest full decode that reassigns
    that note to a different string, vs the chosen decode — best vs next-best
    in string space, read off the trellis. Mapped to ``TabEvent.confidence``
    via :func:`playability.string_margin_to_confidence`. The decoded string /
    fret picks are identical to a forward-only Viterbi; only ``confidence``
    changes.
    """

    def state_emission(cluster: list[AudioEvent], state: tuple[Candidate, ...]) -> float:
        total = 0.0
        for ev, c in zip(cluster, state, strict=True):
            f = playability.find_fingering_at(ev.onset_s, fingerings)
            total += playability.emission_cost(c, ev, f, cfg, lambda_vision=lambda_vision)
        # A5: reward states whose positions match a recognised chord voicing.
        # No-op at the default CHORD_SHAPE_BONUS == 0.0; fires only on clusters
        # of >= CHORD_SHAPE_MIN_NOTES notes, so single-line decode is unchanged.
        total += chord_shapes.chord_shape_cost(state)
        return total

    n = len(cluster_data)
    anchors = [[chord.chord_anchor(st) for st in states] for _, states in cluster_data]
    emissions = [[state_emission(cluster, st) for st in states] for cluster, states in cluster_data]

    def single_line(i: int) -> bool:
        # The learned sequence prior (A15) models note-to-note movement;
        # chord-to-chord transitions stay on the hand-coded terms.
        return len(cluster_data[i - 1][0]) == 1 and len(cluster_data[i][0]) == 1

    def gap_s_between(i: int, j: int) -> float:
        # A4: inter-onset gap (first-event onsets) between adjacent clusters i
        # and j decays the hand-continuity terms; None when TRANSITION_GAP_TAU
        # is inf (the default) — a no-op regardless of this value.
        return cluster_data[j][0][0].onset_s - cluster_data[i][0][0].onset_s

    # Forward pass: alpha[i][si] = min cost of a path ending in state si of
    # cluster i (emission included). Factoring the (pi-invariant) emission out
    # of the argmin leaves picks bit-identical to a plain forward Viterbi.
    alpha: list[list[float]] = [[math.inf] * len(a) for a in anchors]
    backptr: list[list[int]] = [[-1] * len(a) for a in anchors]
    alpha[0] = list(emissions[0])
    for i in range(1, n):
        gap_s = gap_s_between(i - 1, i)
        for si, anchor_curr in enumerate(anchors[i]):
            best = math.inf
            best_pi = -1
            for pi, anchor_prev in enumerate(anchors[i - 1]):
                cand = alpha[i - 1][pi] + playability.transition_cost(
                    anchor_prev, anchor_curr, cfg, use_sequence_prior=single_line(i), gap_s=gap_s
                )
                if cand < best:
                    best = cand
                    best_pi = pi
            alpha[i][si] = best + emissions[i][si]
            backptr[i][si] = best_pi

    # Backward pass: beta[i][si] = min cost to finish from state si of cluster
    # i (transition into i+1 + its emission + its beta). Excludes
    # emissions[i][si] so alpha[i][si] + beta[i][si] counts it exactly once.
    beta: list[list[float]] = [[0.0] * len(a) for a in anchors]
    for i in range(n - 2, -1, -1):
        gap_s = gap_s_between(i, i + 1)
        for si, anchor_state in enumerate(anchors[i]):
            best = math.inf
            for ti, anchor_next in enumerate(anchors[i + 1]):
                cand = (
                    playability.transition_cost(
                        anchor_state,
                        anchor_next,
                        cfg,
                        use_sequence_prior=single_line(i + 1),
                        gap_s=gap_s,
                    )
                    + emissions[i + 1][ti]
                    + beta[i + 1][ti]
                )
                if cand < best:
                    best = cand
            beta[i][si] = best

    through = [[alpha[i][si] + beta[i][si] for si in range(len(a))] for i, a in enumerate(anchors)]
    global_opt = min(alpha[n - 1])

    # Backtrack the decoded path from the cheapest terminal state.
    picks_idx = [0] * n
    picks_idx[n - 1] = min(range(len(alpha[n - 1])), key=lambda j: alpha[n - 1][j])
    for i in range(n - 1, 0, -1):
        picks_idx[i - 1] = backptr[i][picks_idx[i]]

    out: list[TabEvent] = []
    for i, (cluster, states) in enumerate(cluster_data):
        state = states[picks_idx[i]]
        for j, (ev, c) in enumerate(zip(cluster, state, strict=True)):
            margin = _string_flip_margin(states, through[i], j, c.string_idx, global_opt)
            out.append(
                TabEvent(
                    onset_s=ev.onset_s,
                    duration_s=max(0.0, ev.offset_s - ev.onset_s),
                    string_idx=c.string_idx,
                    fret=c.fret,
                    pitch_midi=ev.pitch_midi,
                    confidence=playability.string_margin_to_confidence(margin),
                    techniques=ev.tags,
                )
            )
    return out


def _string_flip_margin(
    states: list[tuple[Candidate, ...]],
    through_costs: list[float],
    event_pos: int,
    chosen_string: int,
    global_opt: float,
) -> float:
    """Extra cost (nats) of the best full decode that moves ``event_pos`` to a
    different string than ``chosen_string``. ``inf`` if none exists (the pitch
    is playable on only one string given the cluster's constraints)."""
    alt = math.inf
    for si, state in enumerate(states):
        if state[event_pos].string_idx != chosen_string and through_costs[si] < alt:
            alt = through_costs[si]
    if alt == math.inf:
        return math.inf
    return max(0.0, alt - global_opt)


__all__ = ["fuse"]
