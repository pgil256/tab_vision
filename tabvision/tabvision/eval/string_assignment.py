"""Leakage-free string-assignment evaluation helpers.

This module is intentionally evaluation-only.  It mirrors the shipped
cluster-level Viterbi costs so Phase 0 can inspect candidate rankings and run
constrained K-best ceiling probes without changing the immutable ``fuse``
signature or production behaviour.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tabvision.fusion import chord, chord_shapes, playability
from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


@dataclass(frozen=True)
class RankedCandidate:
    """A playable position ranked by its cheapest complete-decode cost."""

    string_idx: int
    fret: int
    cost_delta_from_best: float


@dataclass(frozen=True)
class DecodedPath:
    """One complete decode returned by the constrained K-best search."""

    events: tuple[TabEvent, ...]
    cost: float
    score_delta_from_best: float


@dataclass(frozen=True)
class DecodeAnalysis:
    """K-best paths plus per-note min-marginal candidate rankings."""

    audio_events: tuple[AudioEvent, ...]
    paths: tuple[DecodedPath, ...]
    candidate_ranks: tuple[tuple[RankedCandidate, ...], ...]


@dataclass(frozen=True)
class PredictionMatch:
    """The error-decomposition label assigned to one predicted note."""

    predicted_index: int
    label: str
    gold_index: int | None


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired clip-stratified bootstrap interval for a metric delta."""

    mean_delta: float
    lower: float
    upper: float
    n_resamples: int


@dataclass(frozen=True)
class PhraseWindow:
    """A phrase-sized slice of flattened, onset-sorted events."""

    start_index: int
    end_index: int
    anchor_index: int


@dataclass(frozen=True)
class _KEntry:
    cost: float
    prev_state: int
    prev_rank: int


def decode_with_analysis(
    events: Sequence[AudioEvent],
    *,
    cfg: GuitarConfig | None = None,
    constraints: Mapping[int, Candidate] | None = None,
    k_paths: int = 1,
    left_boundary: Candidate | None = None,
    right_boundary: Candidate | None = None,
) -> DecodeAnalysis:
    """Decode with the shipped costs and expose rankings needed by Phase 0.

    ``constraints`` are keyed by the flattened index after unplayable events
    are removed and chord clusters are onset-sorted.  They are exact hard
    constraints: infeasible states are removed rather than assigned a large
    penalty.
    """

    if k_paths < 1:
        raise ValueError("k_paths must be positive")
    cfg = cfg or GuitarConfig()
    constraint_map = dict(constraints or {})

    valid_events = [event for event in events if candidate_positions(event.pitch_midi, cfg)]
    clusters = chord.cluster_events(valid_events)
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]] = []
    cluster_offsets: list[int] = []
    flat_events: list[AudioEvent] = []
    offset = 0
    for cluster in clusters:
        states = chord.enumerate_chord_states(cluster, cfg)
        filtered: list[tuple[Candidate, ...]] = []
        for state in states:
            if all(
                offset + local_idx not in constraint_map
                or state[local_idx] == constraint_map[offset + local_idx]
                for local_idx in range(len(cluster))
            ):
                filtered.append(state)
        if not filtered:
            return DecodeAnalysis(tuple(flat_events + cluster), (), ())
        cluster_offsets.append(offset)
        cluster_data.append((cluster, filtered))
        flat_events.extend(cluster)
        offset += len(cluster)

    if not cluster_data:
        return DecodeAnalysis((), (), ())

    anchors = [[chord.chord_anchor(state) for state in states] for _, states in cluster_data]
    emissions = [
        [_state_emission(cluster, state, cfg) for state in states]
        for cluster, states in cluster_data
    ]
    alpha, backptr = _forward_costs(
        cluster_data,
        anchors,
        emissions,
        cfg,
        left_boundary=left_boundary,
    )
    beta = _backward_costs(
        cluster_data,
        anchors,
        emissions,
        cfg,
        right_boundary=right_boundary,
    )
    terminal_costs = [
        alpha[-1][state_idx]
        + _right_boundary_cost(
            anchors[-1][state_idx],
            right_boundary,
            cfg,
        )
        for state_idx in range(len(anchors[-1]))
    ]
    global_opt = min(terminal_costs)
    best_state_indices = _backtrack_best(backptr, terminal_costs)
    through = [
        [alpha[i][state_idx] + beta[i][state_idx] for state_idx in range(len(states))]
        for i, (_cluster, states) in enumerate(cluster_data)
    ]

    ranked = _rank_candidates(
        cluster_data,
        cluster_offsets,
        through,
        best_state_indices,
        global_opt,
    )
    best_events = _events_for_state_path(
        cluster_data,
        best_state_indices,
        through,
        global_opt,
        exact_confidence=True,
    )
    best_key = _path_key(best_events)

    k_state_paths = _k_best_state_paths(
        cluster_data,
        anchors,
        emissions,
        cfg,
        k_paths=max(k_paths, 1),
        left_boundary=left_boundary,
        right_boundary=right_boundary,
    )
    decoded_paths: list[DecodedPath] = [DecodedPath(tuple(best_events), global_opt, 0.0)]
    seen = {best_key}
    for state_path, cost in k_state_paths:
        candidate_events = _events_for_state_path(
            cluster_data,
            state_path,
            through,
            global_opt,
            exact_confidence=False,
        )
        key = _path_key(candidate_events)
        if key in seen:
            continue
        seen.add(key)
        decoded_paths.append(
            DecodedPath(
                tuple(candidate_events),
                cost,
                max(0.0, cost - global_opt),
            )
        )
        if len(decoded_paths) >= k_paths:
            break

    return DecodeAnalysis(tuple(flat_events), tuple(decoded_paths), tuple(ranked))


def _state_emission(
    cluster: Sequence[AudioEvent],
    state: tuple[Candidate, ...],
    cfg: GuitarConfig,
) -> float:
    total = sum(
        playability.emission_cost(candidate, event, None, cfg, lambda_vision=0.0)
        for event, candidate in zip(cluster, state, strict=True)
    )
    return total + chord_shapes.chord_shape_cost(state)


def _uses_sequence_prior(
    cluster_data: Sequence[tuple[Sequence[AudioEvent], Sequence[tuple[Candidate, ...]]]],
    index: int,
) -> bool:
    return len(cluster_data[index - 1][0]) == 1 and len(cluster_data[index][0]) == 1


def _gap_s(
    cluster_data: Sequence[tuple[Sequence[AudioEvent], Sequence[tuple[Candidate, ...]]]],
    left: int,
    right: int,
) -> float:
    return cluster_data[right][0][0].onset_s - cluster_data[left][0][0].onset_s


def _forward_costs(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    anchors: list[list[Candidate]],
    emissions: list[list[float]],
    cfg: GuitarConfig,
    *,
    left_boundary: Candidate | None,
) -> tuple[list[list[float]], list[list[int]]]:
    alpha = [[math.inf] * len(row) for row in anchors]
    backptr = [[-1] * len(row) for row in anchors]
    for state_idx, anchor in enumerate(anchors[0]):
        alpha[0][state_idx] = emissions[0][state_idx] + _left_boundary_cost(
            left_boundary, anchor, cfg
        )
    for i in range(1, len(cluster_data)):
        gap = _gap_s(cluster_data, i - 1, i)
        for state_idx, current in enumerate(anchors[i]):
            best = math.inf
            best_prev = -1
            for prev_idx, previous in enumerate(anchors[i - 1]):
                candidate_cost = alpha[i - 1][prev_idx] + playability.transition_cost(
                    previous,
                    current,
                    cfg,
                    use_sequence_prior=_uses_sequence_prior(cluster_data, i),
                    gap_s=gap,
                )
                if candidate_cost < best:
                    best = candidate_cost
                    best_prev = prev_idx
            alpha[i][state_idx] = best + emissions[i][state_idx]
            backptr[i][state_idx] = best_prev
    return alpha, backptr


def _backward_costs(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    anchors: list[list[Candidate]],
    emissions: list[list[float]],
    cfg: GuitarConfig,
    *,
    right_boundary: Candidate | None,
) -> list[list[float]]:
    beta = [[0.0] * len(row) for row in anchors]
    for state_idx, anchor in enumerate(anchors[-1]):
        beta[-1][state_idx] = _right_boundary_cost(anchor, right_boundary, cfg)
    for i in range(len(cluster_data) - 2, -1, -1):
        gap = _gap_s(cluster_data, i, i + 1)
        for state_idx, current in enumerate(anchors[i]):
            best = math.inf
            for next_idx, following in enumerate(anchors[i + 1]):
                candidate_cost = (
                    playability.transition_cost(
                        current,
                        following,
                        cfg,
                        use_sequence_prior=_uses_sequence_prior(cluster_data, i + 1),
                        gap_s=gap,
                    )
                    + emissions[i + 1][next_idx]
                    + beta[i + 1][next_idx]
                )
                if candidate_cost < best:
                    best = candidate_cost
            beta[i][state_idx] = best
    return beta


def _left_boundary_cost(
    boundary: Candidate | None,
    current: Candidate,
    cfg: GuitarConfig,
) -> float:
    if boundary is None:
        return 0.0
    return playability.transition_cost(boundary, current, cfg, use_sequence_prior=False)


def _right_boundary_cost(
    current: Candidate,
    boundary: Candidate | None,
    cfg: GuitarConfig,
) -> float:
    if boundary is None:
        return 0.0
    return playability.transition_cost(current, boundary, cfg, use_sequence_prior=False)


def _backtrack_best(backptr: list[list[int]], terminal_costs: list[float]) -> list[int]:
    picks = [0] * len(backptr)
    picks[-1] = min(range(len(terminal_costs)), key=terminal_costs.__getitem__)
    for i in range(len(backptr) - 1, 0, -1):
        picks[i - 1] = backptr[i][picks[i]]
    return picks


def _rank_candidates(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    offsets: list[int],
    through: list[list[float]],
    picks: list[int],
    global_opt: float,
) -> list[tuple[RankedCandidate, ...]]:
    ranked: list[tuple[RankedCandidate, ...]] = [()] * sum(
        len(cluster) for cluster, _states in cluster_data
    )
    for cluster_idx, (cluster, states) in enumerate(cluster_data):
        chosen_state = states[picks[cluster_idx]]
        for local_idx in range(len(cluster)):
            best_by_candidate: dict[Candidate, float] = {}
            for state_idx, state in enumerate(states):
                candidate = state[local_idx]
                best_by_candidate[candidate] = min(
                    best_by_candidate.get(candidate, math.inf), through[cluster_idx][state_idx]
                )
            chosen = chosen_state[local_idx]
            ordered = sorted(
                best_by_candidate.items(),
                key=lambda item: (
                    item[1],
                    0 if item[0] == chosen else 1,
                    item[0].fret,
                    item[0].string_idx,
                ),
            )
            ranked[offsets[cluster_idx] + local_idx] = tuple(
                RankedCandidate(
                    candidate.string_idx,
                    candidate.fret,
                    max(0.0, cost - global_opt),
                )
                for candidate, cost in ordered
            )
    return ranked


def _events_for_state_path(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    state_path: Sequence[int],
    through: list[list[float]],
    global_opt: float,
    *,
    exact_confidence: bool,
) -> list[TabEvent]:
    out: list[TabEvent] = []
    for cluster_idx, ((cluster, states), state_idx) in enumerate(
        zip(cluster_data, state_path, strict=True)
    ):
        state = states[state_idx]
        for event_idx, (event, candidate) in enumerate(zip(cluster, state, strict=True)):
            confidence = 1.0
            if exact_confidence:
                alternative = min(
                    (
                        through[cluster_idx][other_idx]
                        for other_idx, other_state in enumerate(states)
                        if other_state[event_idx].string_idx != candidate.string_idx
                    ),
                    default=math.inf,
                )
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


def _k_best_state_paths(
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]],
    anchors: list[list[Candidate]],
    emissions: list[list[float]],
    cfg: GuitarConfig,
    *,
    k_paths: int,
    left_boundary: Candidate | None,
    right_boundary: Candidate | None,
) -> list[tuple[list[int], float]]:
    table: list[list[list[_KEntry]]] = [
        [[] for _state in states] for _cluster, states in cluster_data
    ]
    for state_idx, anchor in enumerate(anchors[0]):
        table[0][state_idx] = [
            _KEntry(
                emissions[0][state_idx] + _left_boundary_cost(left_boundary, anchor, cfg),
                -1,
                -1,
            )
        ]
    for i in range(1, len(cluster_data)):
        gap = _gap_s(cluster_data, i - 1, i)
        for state_idx, current in enumerate(anchors[i]):
            options: list[_KEntry] = []
            for prev_idx, previous in enumerate(anchors[i - 1]):
                transition = playability.transition_cost(
                    previous,
                    current,
                    cfg,
                    use_sequence_prior=_uses_sequence_prior(cluster_data, i),
                    gap_s=gap,
                )
                for prev_rank, entry in enumerate(table[i - 1][prev_idx]):
                    options.append(
                        _KEntry(
                            entry.cost + transition + emissions[i][state_idx],
                            prev_idx,
                            prev_rank,
                        )
                    )
            options.sort(key=lambda entry: (entry.cost, entry.prev_state, entry.prev_rank))
            table[i][state_idx] = options[:k_paths]

    terminals: list[tuple[float, int, int]] = []
    for state_idx, entries in enumerate(table[-1]):
        boundary_cost = _right_boundary_cost(anchors[-1][state_idx], right_boundary, cfg)
        terminals.extend(
            (entry.cost + boundary_cost, state_idx, rank) for rank, entry in enumerate(entries)
        )
    terminals.sort()

    out: list[tuple[list[int], float]] = []
    for cost, state_idx, rank in terminals[:k_paths]:
        path = [0] * len(cluster_data)
        path[-1] = state_idx
        current_rank = rank
        for i in range(len(cluster_data) - 1, 0, -1):
            entry = table[i][path[i]][current_rank]
            path[i - 1] = entry.prev_state
            current_rank = entry.prev_rank
        out.append((path, cost))
    return out


def _path_key(events: Sequence[TabEvent]) -> tuple[tuple[int, int], ...]:
    return tuple((event.string_idx, event.fret) for event in events)


def label_prediction_matches(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = 0.05,
    timing_extended_tolerance_s: float = 0.15,
) -> list[PredictionMatch]:
    """Apply the six-bucket matcher and retain the matched gold index."""

    labels = ["extra_detection"] * len(predicted)
    gold_indices: list[int | None] = [None] * len(predicted)
    used = [False] * len(predicted)
    ordered_gold = sorted(enumerate(gold), key=lambda item: item[1].onset_s)
    for gold_idx, reference in ordered_gold:
        reference_pitch = reference.pitch_midi
        reference_position = (reference.string_idx, reference.fret)
        best_pos = _best_prediction(
            predicted,
            used,
            reference,
            onset_tolerance_s,
            _matches_position(reference_position),
        )
        best_pitch = _best_prediction(
            predicted,
            used,
            reference,
            onset_tolerance_s,
            _matches_pitch(reference_pitch),
        )
        best_any = _best_prediction(
            predicted,
            used,
            reference,
            onset_tolerance_s,
            lambda _event: True,
        )
        chosen = best_pos if best_pos is not None else best_pitch
        label = "correct" if best_pos is not None else "wrong_position_same_pitch"
        if chosen is None:
            chosen = best_any
            label = "pitch_off"
        if chosen is None:
            chosen = _best_prediction(
                predicted,
                used,
                reference,
                timing_extended_tolerance_s,
                _matches_pitch_or_position(reference_pitch, reference_position),
            )
            label = "timing_only"
        if chosen is not None:
            used[chosen] = True
            labels[chosen] = label
            gold_indices[chosen] = gold_idx
    return [
        PredictionMatch(index, label, gold_indices[index]) for index, label in enumerate(labels)
    ]


def _matches_position(position: tuple[int, int]) -> Callable[[TabEvent], bool]:
    return lambda event: (event.string_idx, event.fret) == position


def _matches_pitch(pitch: int) -> Callable[[TabEvent], bool]:
    return lambda event: event.pitch_midi == pitch


def _matches_pitch_or_position(
    pitch: int,
    position: tuple[int, int],
) -> Callable[[TabEvent], bool]:
    return lambda event: event.pitch_midi == pitch or (event.string_idx, event.fret) == position


def _best_prediction(
    predicted: Sequence[TabEvent],
    used: Sequence[bool],
    reference: TabEvent,
    tolerance_s: float,
    predicate: Callable[[TabEvent], bool],
) -> int | None:
    best_idx: int | None = None
    best_delta = tolerance_s + 1e-9
    for index, event in enumerate(predicted):
        if used[index]:
            continue
        delta = abs(event.onset_s - reference.onset_s)
        if delta > tolerance_s:
            continue
        if not predicate(event):
            continue
        if delta < best_delta:
            best_idx = index
            best_delta = delta
    return best_idx


def wrong_below_correct_auc(confidence_and_correct: Sequence[tuple[float, bool]]) -> float:
    """AUC where a wrong-position note should have lower confidence."""

    wrong = [confidence for confidence, correct in confidence_and_correct if not correct]
    correct = sorted(confidence for confidence, is_correct in confidence_and_correct if is_correct)
    if not wrong or not correct:
        return float("nan")
    total = 0.0
    for confidence in wrong:
        left = bisect.bisect_left(correct, confidence)
        right = bisect.bisect_right(correct, confidence)
        higher = len(correct) - right
        ties = right - left
        total += higher + 0.5 * ties
    return total / (len(wrong) * len(correct))


def expected_calibration_error(
    confidence_and_correct: Sequence[tuple[float, bool]],
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for string-assignment confidence."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not confidence_and_correct:
        return float("nan")
    total = len(confidence_and_correct)
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        bucket = [
            (confidence, correct)
            for confidence, correct in confidence_and_correct
            if lower <= confidence < upper or (bin_idx == n_bins - 1 and confidence == 1.0)
        ]
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / total * abs(accuracy - mean_confidence)
    return ece


def paired_stratified_bootstrap(
    baseline_by_clip: Mapping[str, float],
    candidate_by_clip: Mapping[str, float],
    stratum_by_clip: Mapping[str, str],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Bootstrap paired clip deltas while preserving each stratum's size."""

    common = sorted(set(baseline_by_clip) & set(candidate_by_clip))
    if not common:
        raise ValueError("no paired clips")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    strata: dict[str, list[float]] = {}
    for clip_id in common:
        stratum = stratum_by_clip[clip_id]
        strata.setdefault(stratum, []).append(
            candidate_by_clip[clip_id] - baseline_by_clip[clip_id]
        )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    total_clips = len(common)
    for draw_idx in range(n_resamples):
        total = 0.0
        for values in strata.values():
            indices = rng.integers(0, len(values), size=len(values))
            total += sum(values[int(index)] for index in indices)
        draws[draw_idx] = total / total_clips
    point = (
        sum(candidate_by_clip[clip_id] - baseline_by_clip[clip_id] for clip_id in common)
        / total_clips
    )
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return PairedBootstrapResult(point, float(lower), float(upper), n_resamples)


def phrase_windows(
    events: Sequence[TabEvent],
    ambiguous_indices: set[int],
    *,
    cluster_gap_s: float = 0.080,
    phrase_gap_s: float = 0.75,
    max_duration_s: float = 8.0,
    max_notes: int = 32,
) -> list[PhraseWindow]:
    """Build anchor-centred phrase windows without splitting simultaneous notes."""

    if not events or not ambiguous_indices:
        return []
    indexed = sorted(enumerate(events), key=lambda item: item[1].onset_s)
    clusters: list[list[tuple[int, TabEvent]]] = [[indexed[0]]]
    for item in indexed[1:]:
        if item[1].onset_s - clusters[-1][-1][1].onset_s <= cluster_gap_s:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    groups: list[list[list[tuple[int, TabEvent]]]] = [[clusters[0]]]
    for cluster in clusters[1:]:
        gap = cluster[0][1].onset_s - groups[-1][-1][-1][1].onset_s
        if gap <= phrase_gap_s:
            groups[-1].append(cluster)
        else:
            groups.append([cluster])

    windows: list[PhraseWindow] = []
    for group in groups:
        _partition_phrase_group(
            group,
            ambiguous_indices,
            windows,
            max_duration_s=max_duration_s,
            max_notes=max_notes,
        )
    return sorted(windows, key=lambda window: window.start_index)


def _partition_phrase_group(
    group: list[list[tuple[int, TabEvent]]],
    ambiguous_indices: set[int],
    out: list[PhraseWindow],
    *,
    max_duration_s: float,
    max_notes: int,
) -> None:
    ambiguous = sorted(
        index for cluster in group for index, _event in cluster if index in ambiguous_indices
    )
    if not ambiguous:
        return
    anchor = ambiguous[0]
    anchor_cluster = next(
        idx for idx, cluster in enumerate(group) if any(index == anchor for index, _ in cluster)
    )
    left = right = anchor_cluster
    while True:
        choices: list[tuple[float, int, int]] = []
        if left > 0:
            gap = group[left][0][1].onset_s - group[left - 1][-1][1].onset_s
            choices.append((gap, left - 1, right))
        if right + 1 < len(group):
            gap = group[right + 1][0][1].onset_s - group[right][-1][1].onset_s
            choices.append((gap, left, right + 1))
        expanded = False
        for _gap, candidate_left, candidate_right in sorted(choices):
            selected = group[candidate_left : candidate_right + 1]
            note_count = sum(len(cluster) for cluster in selected)
            duration = selected[-1][-1][1].onset_s - selected[0][0][1].onset_s
            if note_count <= max_notes and duration <= max_duration_s:
                left, right = candidate_left, candidate_right
                expanded = True
                break
        if not expanded:
            break

    selected_items = [item for cluster in group[left : right + 1] for item in cluster]
    selected_indices = sorted(index for index, _event in selected_items)
    out.append(PhraseWindow(selected_indices[0], selected_indices[-1] + 1, anchor))
    if left > 0:
        _partition_phrase_group(
            group[:left],
            ambiguous_indices,
            out,
            max_duration_s=max_duration_s,
            max_notes=max_notes,
        )
    if right + 1 < len(group):
        _partition_phrase_group(
            group[right + 1 :],
            ambiguous_indices,
            out,
            max_duration_s=max_duration_s,
            max_notes=max_notes,
        )


__all__ = [
    "DecodeAnalysis",
    "DecodedPath",
    "PairedBootstrapResult",
    "PhraseWindow",
    "PredictionMatch",
    "RankedCandidate",
    "decode_with_analysis",
    "expected_calibration_error",
    "label_prediction_matches",
    "paired_stratified_bootstrap",
    "phrase_windows",
    "wrong_below_correct_auc",
]
