"""Bounded latent hand-position decoder for string/fret assignment.

``segment-v1`` preserves the detected MIDI pitch and the existing chord
constraints.  It augments the shipped cluster Viterbi with one latent hand
state per cluster-safe segment: a preferred adjacent-string offset relative
to the baseline path and a fret-zone hypothesis.  The decoder is pure; all
configuration is passed explicitly so concurrent jobs cannot share settings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tabvision.fusion import chord, chord_shapes, playability
from tabvision.fusion.candidates import Candidate
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, TabEvent


@dataclass(frozen=True)
class SegmentDecoderConfig:
    """Frozen scoring and segmentation settings for ``segment-v1``."""

    rest_boundary_s: float = 0.75
    max_segment_s: float = 4.0
    max_segment_notes: int = 32
    zone_centers: tuple[int, ...] = (2, 5, 7, 10, 13)
    zone_weight: float = 1.0
    offset_weight: float = 1.0
    state_change_weight: float = 1.0
    prior_weight: float = 0.5
    transition_weight: float = 1.0
    repeat_weight: float = 0.0
    relaxed_state_change_scale: float = 0.25
    register_jump_semitones: int = 7

    def __post_init__(self) -> None:
        if self.rest_boundary_s < 0.0:
            raise ValueError("rest_boundary_s must be non-negative")
        if self.max_segment_s <= 0.0:
            raise ValueError("max_segment_s must be positive")
        if self.max_segment_notes < 1:
            raise ValueError("max_segment_notes must be positive")
        if not self.zone_centers:
            raise ValueError("zone_centers cannot be empty")
        weights = (
            self.zone_weight,
            self.offset_weight,
            self.state_change_weight,
            self.prior_weight,
            self.transition_weight,
            self.repeat_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("decoder weights must be non-negative")
        if self.repeat_weight != 0.0:
            raise ValueError("repeat consistency is disabled until its independent gate passes")


DEFAULT_SEGMENT_CONFIG = SegmentDecoderConfig()


@dataclass(frozen=True)
class LatentHandState:
    """One segment-level adjacent-string and fret-zone hypothesis."""

    string_offset: int
    zone_center: int | None

    @property
    def label(self) -> str:
        zone = "open" if self.zone_center is None else str(self.zone_center)
        return f"offset={self.string_offset:+d},zone={zone}"


@dataclass(frozen=True)
class SegmentBoundary:
    start_cluster: int
    end_cluster: int
    start_onset_s: float
    end_onset_s: float
    note_count: int


@dataclass(frozen=True)
class SegmentRankedCandidate:
    string_idx: int
    fret: int
    cost_delta_from_best: float


@dataclass(frozen=True)
class SegmentDecodedPath:
    events: tuple[TabEvent, ...]
    latent_states: tuple[LatentHandState, ...]
    cost: float
    score_delta_from_best: float


@dataclass(frozen=True)
class SegmentDecodeResult:
    audio_events: tuple[AudioEvent, ...]
    paths: tuple[SegmentDecodedPath, ...]
    candidate_ranks: tuple[tuple[SegmentRankedCandidate, ...], ...]
    segments: tuple[SegmentBoundary, ...]


ClusterData = list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]]


def latent_states(config: SegmentDecoderConfig) -> tuple[LatentHandState, ...]:
    """Return deterministic latent-state ordering, closest to baseline first."""

    states = [LatentHandState(0, None)]
    for offset in (0, -1, 1):
        states.extend(LatentHandState(offset, center) for center in config.zone_centers)
    return tuple(states)


def partition_clusters(
    cluster_data: Sequence[tuple[Sequence[AudioEvent], Sequence[tuple[Candidate, ...]]]],
    config: SegmentDecoderConfig = DEFAULT_SEGMENT_CONFIG,
) -> tuple[SegmentBoundary, ...]:
    """Partition at rests or the four-second/32-note caps, never within a cluster."""

    if not cluster_data:
        return ()
    boundaries: list[SegmentBoundary] = []
    start = 0
    start_onset = cluster_data[0][0][0].onset_s
    note_count = 0
    for index, (cluster, _states) in enumerate(cluster_data):
        onset = cluster[0].onset_s
        gap = 0.0 if index == 0 else onset - cluster_data[index - 1][0][0].onset_s
        exceeds = index > start and (
            gap > config.rest_boundary_s
            or onset - start_onset > config.max_segment_s
            or note_count + len(cluster) > config.max_segment_notes
        )
        if exceeds:
            previous_onset = cluster_data[index - 1][0][0].onset_s
            boundaries.append(
                SegmentBoundary(start, index, start_onset, previous_onset, note_count)
            )
            start = index
            start_onset = onset
            note_count = 0
        note_count += len(cluster)
    boundaries.append(
        SegmentBoundary(
            start,
            len(cluster_data),
            start_onset,
            cluster_data[-1][0][0].onset_s,
            note_count,
        )
    )
    return tuple(boundaries)


def decode_segment_clusters(
    cluster_data: ClusterData,
    baseline_events: Sequence[TabEvent],
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig,
    lambda_vision: float,
    *,
    config: SegmentDecoderConfig = DEFAULT_SEGMENT_CONFIG,
    k_paths: int = 1,
    retain_analysis: bool = True,
) -> SegmentDecodeResult:
    """Run exact product-state DP over chord states and segment hand states."""

    if k_paths < 1:
        raise ValueError("k_paths must be positive")
    if not retain_analysis and k_paths != 1:
        raise ValueError("fast decode without analysis supports only the top path")
    flat_events = tuple(event for cluster, _states in cluster_data for event in cluster)
    if not cluster_data:
        return SegmentDecodeResult((), (), (), ())
    if len(baseline_events) != len(flat_events):
        raise ValueError("baseline path must contain one event per decodable audio event")
    baseline_candidates = tuple(
        Candidate(event.string_idx, event.fret) for event in baseline_events
    )
    boundaries = partition_clusters(cluster_data, config)
    segment_ids = _segment_ids(len(cluster_data), boundaries)
    hand_states = latent_states(config)
    baseline_by_cluster = _split_baseline(cluster_data, baseline_candidates)
    emissions = _combined_emissions(
        cluster_data,
        baseline_by_cluster,
        hand_states,
        fingerings,
        cfg,
        lambda_vision,
        config,
    )
    candidate_transitions, latent_transitions = _transition_tables(
        cluster_data,
        hand_states,
        segment_ids,
        cfg,
        config,
    )
    alpha, backptr = _forward(
        cluster_data,
        hand_states,
        segment_ids,
        emissions,
        candidate_transitions,
        latent_transitions,
    )
    global_opt = min(alpha[-1])
    chosen_nodes = _backtrack(backptr, alpha[-1])
    through: list[list[float]] = []
    if retain_analysis:
        beta = _backward(
            cluster_data,
            hand_states,
            segment_ids,
            emissions,
            candidate_transitions,
            latent_transitions,
        )
        through = [
            [alpha[index][node] + beta[index][node] for node in range(len(alpha[index]))]
            for index in range(len(cluster_data))
        ]
    best_events = _events_for_nodes(
        cluster_data,
        hand_states,
        chosen_nodes,
        through,
        global_opt,
        exact_confidence=retain_analysis,
    )
    ranks = (
        _rank_candidates(cluster_data, hand_states, chosen_nodes, through, global_opt)
        if retain_analysis
        else []
    )
    if k_paths == 1:
        state_paths = [(chosen_nodes, global_opt)]
    else:
        alternatives = _k_best_paths(
            cluster_data,
            hand_states,
            segment_ids,
            emissions,
            candidate_transitions,
            latent_transitions,
            k_paths + 1,
        )
        state_paths = [(chosen_nodes, global_opt)]
        for nodes, cost in alternatives:
            if nodes == chosen_nodes:
                continue
            state_paths.append((nodes, cost))
            if len(state_paths) >= k_paths:
                break
    paths: list[SegmentDecodedPath] = []
    for path_index, (nodes, cost) in enumerate(state_paths):
        path_events = (
            best_events
            if path_index == 0 and nodes == chosen_nodes
            else _events_for_nodes(
                cluster_data,
                hand_states,
                nodes,
                through,
                global_opt,
                exact_confidence=False,
            )
        )
        paths.append(
            SegmentDecodedPath(
                tuple(path_events),
                _path_latent_states(nodes, cluster_data, hand_states, boundaries),
                cost,
                max(0.0, cost - global_opt),
            )
        )
    return SegmentDecodeResult(flat_events, tuple(paths), tuple(ranks), boundaries)


def _segment_ids(n_clusters: int, boundaries: Sequence[SegmentBoundary]) -> list[int]:
    ids = [-1] * n_clusters
    for segment_id, boundary in enumerate(boundaries):
        ids[boundary.start_cluster : boundary.end_cluster] = [segment_id] * (
            boundary.end_cluster - boundary.start_cluster
        )
    return ids


def _split_baseline(
    cluster_data: ClusterData,
    baseline: Sequence[Candidate],
) -> list[tuple[Candidate, ...]]:
    out: list[tuple[Candidate, ...]] = []
    offset = 0
    for cluster, _states in cluster_data:
        end = offset + len(cluster)
        out.append(tuple(baseline[offset:end]))
        offset = end
    return out


def _combined_emissions(
    cluster_data: ClusterData,
    baseline_by_cluster: Sequence[Sequence[Candidate]],
    hand_states: Sequence[LatentHandState],
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig,
    lambda_vision: float,
    config: SegmentDecoderConfig,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for (cluster, states), baseline in zip(cluster_data, baseline_by_cluster, strict=True):
        base_costs = []
        for state in states:
            cost = 0.0
            for event, candidate in zip(cluster, state, strict=True):
                fingering = playability.find_fingering_at(event.onset_s, fingerings)
                cost += playability.emission_cost(
                    candidate,
                    event,
                    fingering,
                    cfg,
                    lambda_vision=lambda_vision,
                )
            base_costs.append(cost + chord_shapes.chord_shape_cost(state))
        row: list[float] = []
        for hand_state in hand_states:
            for candidate_state, base_cost in zip(states, base_costs, strict=True):
                extra = sum(
                    _latent_emission(candidate, baseline_candidate, event, hand_state, config)
                    for candidate, baseline_candidate, event in zip(
                        candidate_state, baseline, cluster, strict=True
                    )
                )
                row.append(base_cost + extra)
        rows.append(row)
    return rows


def _latent_emission(
    candidate: Candidate,
    baseline: Candidate,
    event: AudioEvent,
    state: LatentHandState,
    config: SegmentDecoderConfig,
) -> float:
    offset = candidate.string_idx - baseline.string_idx
    cost = config.offset_weight * abs(offset - state.string_offset)
    if candidate.fret > 0:
        if state.zone_center is None:
            zone_distance = max(0, candidate.fret - 4) / 12.0
        else:
            zone_distance = abs(candidate.fret - state.zone_center) / 12.0
        cost += config.zone_weight * zone_distance
    if config.prior_weight > 0.0 and event.fret_prior is not None:
        probability = _candidate_prior(event.fret_prior, candidate)
        cost += config.prior_weight * -math.log(max(probability, playability.EPS))
    return cost


def _candidate_prior(prior: object, candidate: Candidate) -> float:
    try:
        shape = getattr(prior, "shape", ())
        if len(shape) == 2:
            return float(prior[candidate.string_idx, candidate.fret])  # type: ignore[index]
        if len(shape) == 1:
            return float(prior[candidate.fret])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return 0.0
    return 0.0


def _forward(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    segment_ids: Sequence[int],
    emissions: Sequence[Sequence[float]],
    candidate_transitions: Sequence[Sequence[Sequence[float]]],
    latent_transitions: Sequence[Sequence[Sequence[float]]],
) -> tuple[list[list[float]], list[list[int]]]:
    alpha = [[math.inf] * len(row) for row in emissions]
    backptr = [[-1] * len(row) for row in emissions]
    alpha[0] = list(emissions[0])
    for index in range(1, len(cluster_data)):
        previous_states = cluster_data[index - 1][1]
        current_states = cluster_data[index][1]
        boundary = segment_ids[index] != segment_ids[index - 1]
        for latent_idx, _latent in enumerate(hand_states):
            previous_latents = range(len(hand_states)) if boundary else (latent_idx,)
            for candidate_idx, _current in enumerate(current_states):
                node = _node_index(latent_idx, candidate_idx, len(current_states))
                best = math.inf
                best_previous = -1
                for previous_latent_idx in previous_latents:
                    for previous_candidate_idx, _previous in enumerate(previous_states):
                        previous_node = _node_index(
                            previous_latent_idx,
                            previous_candidate_idx,
                            len(previous_states),
                        )
                        candidate_cost = (
                            alpha[index - 1][previous_node]
                            + candidate_transitions[index][previous_candidate_idx][candidate_idx]
                            + latent_transitions[index][previous_latent_idx][latent_idx]
                        )
                        if candidate_cost < best:
                            best = candidate_cost
                            best_previous = previous_node
                alpha[index][node] = best + emissions[index][node]
                backptr[index][node] = best_previous
    return alpha, backptr


def _backward(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    segment_ids: Sequence[int],
    emissions: Sequence[Sequence[float]],
    candidate_transitions: Sequence[Sequence[Sequence[float]]],
    latent_transitions: Sequence[Sequence[Sequence[float]]],
) -> list[list[float]]:
    beta = [[0.0] * len(row) for row in emissions]
    for index in range(len(cluster_data) - 2, -1, -1):
        current_states = cluster_data[index][1]
        next_states = cluster_data[index + 1][1]
        boundary = segment_ids[index] != segment_ids[index + 1]
        for latent_idx, _latent in enumerate(hand_states):
            next_latents = range(len(hand_states)) if boundary else (latent_idx,)
            for candidate_idx, _current in enumerate(current_states):
                node = _node_index(latent_idx, candidate_idx, len(current_states))
                best = math.inf
                for next_latent_idx in next_latents:
                    for next_candidate_idx, _following in enumerate(next_states):
                        next_node = _node_index(
                            next_latent_idx,
                            next_candidate_idx,
                            len(next_states),
                        )
                        candidate_cost = (
                            candidate_transitions[index + 1][candidate_idx][next_candidate_idx]
                            + latent_transitions[index + 1][latent_idx][next_latent_idx]
                            + emissions[index + 1][next_node]
                            + beta[index + 1][next_node]
                        )
                        if candidate_cost < best:
                            best = candidate_cost
                beta[index][node] = best
    return beta


def _transition_tables(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    segment_ids: Sequence[int],
    cfg: GuitarConfig,
    config: SegmentDecoderConfig,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Precompute separable candidate and latent transition matrices once."""

    candidate_tables: list[list[list[float]]] = [[]]
    latent_tables: list[list[list[float]]] = [[]]
    anchors = [[chord.chord_anchor(state) for state in states] for _cluster, states in cluster_data]
    for index in range(1, len(cluster_data)):
        gap_s = _gap(cluster_data, index - 1, index)
        candidate_tables.append(
            [
                [
                    config.transition_weight
                    * playability.transition_cost(
                        previous,
                        current,
                        cfg,
                        use_sequence_prior=_single_line(cluster_data, index),
                        gap_s=gap_s,
                    )
                    for current in anchors[index]
                ]
                for previous in anchors[index - 1]
            ]
        )
        boundary = segment_ids[index] != segment_ids[index - 1]
        if boundary:
            latent_tables.append(
                [
                    [
                        _latent_transition_cost(
                            previous,
                            current,
                            cluster_data[index - 1][0],
                            cluster_data[index][0],
                            gap_s,
                            config,
                        )
                        for current in hand_states
                    ]
                    for previous in hand_states
                ]
            )
        else:
            latent_tables.append(
                [
                    [
                        0.0 if previous == current else math.inf
                        for current in range(len(hand_states))
                    ]
                    for previous in range(len(hand_states))
                ]
            )
    return candidate_tables, latent_tables


def _latent_transition_cost(
    previous: LatentHandState,
    current: LatentHandState,
    previous_cluster: Sequence[AudioEvent],
    current_cluster: Sequence[AudioEvent],
    gap_s: float,
    config: SegmentDecoderConfig,
) -> float:
    previous_zone = 2 if previous.zone_center is None else previous.zone_center
    current_zone = 2 if current.zone_center is None else current.zone_center
    distance = float(abs(previous.string_offset - current.string_offset))
    distance += abs(previous_zone - current_zone) / 5.0
    if previous.zone_center is None and current.zone_center is not None:
        distance += 0.25
    elif previous.zone_center is not None and current.zone_center is None:
        distance += 0.25
    register_jump = abs(
        sum(event.pitch_midi for event in current_cluster) / len(current_cluster)
        - sum(event.pitch_midi for event in previous_cluster) / len(previous_cluster)
    )
    scale = 1.0
    if gap_s > config.rest_boundary_s or register_jump >= config.register_jump_semitones:
        scale = config.relaxed_state_change_scale
    return scale * config.state_change_weight * distance


def _single_line(cluster_data: ClusterData, index: int) -> bool:
    return len(cluster_data[index - 1][0]) == 1 and len(cluster_data[index][0]) == 1


def _gap(cluster_data: ClusterData, left: int, right: int) -> float:
    return cluster_data[right][0][0].onset_s - cluster_data[left][0][0].onset_s


def _node_index(latent_idx: int, candidate_idx: int, n_candidates: int) -> int:
    return latent_idx * n_candidates + candidate_idx


def _node_parts(node: int, n_candidates: int) -> tuple[int, int]:
    return divmod(node, n_candidates)


def _backtrack(backptr: Sequence[Sequence[int]], terminal: Sequence[float]) -> list[int]:
    picks = [0] * len(backptr)
    picks[-1] = min(range(len(terminal)), key=terminal.__getitem__)
    for index in range(len(backptr) - 1, 0, -1):
        picks[index - 1] = backptr[index][picks[index]]
    return picks


def _events_for_nodes(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    nodes: Sequence[int],
    through: Sequence[Sequence[float]],
    global_opt: float,
    *,
    exact_confidence: bool,
) -> list[TabEvent]:
    out: list[TabEvent] = []
    for cluster_idx, ((cluster, states), node) in enumerate(zip(cluster_data, nodes, strict=True)):
        _latent_idx, state_idx = _node_parts(node, len(states))
        state = states[state_idx]
        for event_idx, (event, candidate) in enumerate(zip(cluster, state, strict=True)):
            confidence = 1.0
            if exact_confidence:
                alternative = math.inf
                for other_node, cost in enumerate(through[cluster_idx]):
                    _other_latent, other_state_idx = _node_parts(other_node, len(states))
                    if states[other_state_idx][event_idx].string_idx != candidate.string_idx:
                        alternative = min(alternative, cost)
                margin = math.inf if alternative == math.inf else max(0.0, alternative - global_opt)
                confidence = playability.string_margin_to_confidence(margin)
            out.append(
                TabEvent(
                    onset_s=event.onset_s,
                    duration_s=max(0.0, event.offset_s - event.onset_s),
                    string_idx=candidate.string_idx,
                    fret=candidate.fret,
                    pitch_midi=event.pitch_midi,
                    confidence=confidence,
                    techniques=event.tags,
                )
            )
    return out


def _rank_candidates(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    chosen_nodes: Sequence[int],
    through: Sequence[Sequence[float]],
    global_opt: float,
) -> list[tuple[SegmentRankedCandidate, ...]]:
    ranked: list[tuple[SegmentRankedCandidate, ...]] = []
    for cluster_idx, (_cluster, states) in enumerate(cluster_data):
        _chosen_latent, chosen_state_idx = _node_parts(chosen_nodes[cluster_idx], len(states))
        chosen_state = states[chosen_state_idx]
        for event_idx, chosen in enumerate(chosen_state):
            best_by_candidate: dict[Candidate, float] = {}
            for node, cost in enumerate(through[cluster_idx]):
                _latent_idx, state_idx = _node_parts(node, len(states))
                candidate = states[state_idx][event_idx]
                best_by_candidate[candidate] = min(best_by_candidate.get(candidate, math.inf), cost)
            ordered = sorted(
                best_by_candidate.items(),
                key=lambda item: (
                    item[1],
                    0 if item[0] == chosen else 1,
                    item[0].fret,
                    item[0].string_idx,
                ),
            )
            ranked.append(
                tuple(
                    SegmentRankedCandidate(
                        candidate.string_idx,
                        candidate.fret,
                        max(0.0, cost - global_opt),
                    )
                    for candidate, cost in ordered
                )
            )
    return ranked


@dataclass(frozen=True)
class _KEntry:
    cost: float
    previous_node: int
    previous_rank: int


def _k_best_paths(
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    segment_ids: Sequence[int],
    emissions: Sequence[Sequence[float]],
    candidate_transitions: Sequence[Sequence[Sequence[float]]],
    latent_transitions: Sequence[Sequence[Sequence[float]]],
    k_paths: int,
) -> list[tuple[list[int], float]]:
    table: list[list[list[_KEntry]]] = [[[] for _node in row] for row in emissions]
    table[0] = [[_KEntry(cost, -1, -1)] for cost in emissions[0]]
    for index in range(1, len(cluster_data)):
        previous_states = cluster_data[index - 1][1]
        current_states = cluster_data[index][1]
        boundary = segment_ids[index] != segment_ids[index - 1]
        for latent_idx, _latent in enumerate(hand_states):
            previous_latents = range(len(hand_states)) if boundary else (latent_idx,)
            for candidate_idx, _current in enumerate(current_states):
                node = _node_index(latent_idx, candidate_idx, len(current_states))
                options: list[_KEntry] = []
                for previous_latent_idx in previous_latents:
                    for previous_candidate_idx, _previous in enumerate(previous_states):
                        previous_node = _node_index(
                            previous_latent_idx,
                            previous_candidate_idx,
                            len(previous_states),
                        )
                        transition = (
                            candidate_transitions[index][previous_candidate_idx][candidate_idx]
                            + latent_transitions[index][previous_latent_idx][latent_idx]
                        )
                        for previous_rank, entry in enumerate(table[index - 1][previous_node]):
                            options.append(
                                _KEntry(
                                    entry.cost + transition + emissions[index][node],
                                    previous_node,
                                    previous_rank,
                                )
                            )
                options.sort(
                    key=lambda entry: (
                        entry.cost,
                        entry.previous_node,
                        entry.previous_rank,
                    )
                )
                table[index][node] = options[:k_paths]
    terminals = sorted(
        (entry.cost, node, rank)
        for node, entries in enumerate(table[-1])
        for rank, entry in enumerate(entries)
    )
    out: list[tuple[list[int], float]] = []
    for cost, node, rank in terminals[:k_paths]:
        path = [0] * len(cluster_data)
        path[-1] = node
        current_rank = rank
        for index in range(len(cluster_data) - 1, 0, -1):
            entry = table[index][path[index]][current_rank]
            path[index - 1] = entry.previous_node
            current_rank = entry.previous_rank
        out.append((path, cost))
    return out


def _path_latent_states(
    nodes: Sequence[int],
    cluster_data: ClusterData,
    hand_states: Sequence[LatentHandState],
    boundaries: Sequence[SegmentBoundary],
) -> tuple[LatentHandState, ...]:
    out = []
    for boundary in boundaries:
        states = cluster_data[boundary.start_cluster][1]
        latent_idx, _candidate_idx = _node_parts(nodes[boundary.start_cluster], len(states))
        out.append(hand_states[latent_idx])
    return tuple(out)


__all__ = [
    "DEFAULT_SEGMENT_CONFIG",
    "LatentHandState",
    "SegmentBoundary",
    "SegmentDecodeResult",
    "SegmentDecodedPath",
    "SegmentDecoderConfig",
    "SegmentRankedCandidate",
    "decode_segment_clusters",
    "latent_states",
    "partition_clusters",
]
