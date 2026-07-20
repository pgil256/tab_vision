"""Gold-only segment ceilings for the string-assignment benchmark.

This module lives under ``scripts.eval`` deliberately.  Production code has
no gold labels and must never import these selectors.  Each oracle chooses one
small latent state per track/window and applies it only through real playable
candidates for the already-detected MIDI pitch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from tabvision.eval.string_assignment import RankedCandidate
from tabvision.fusion.candidates import Candidate
from tabvision.types import GuitarConfig, TabEvent

OracleStrategy = Literal["offset", "fret_zone", "joint"]
FretZone = tuple[int, int]

STRING_OFFSETS = (-1, 0, 1)
FRET_ZONES: tuple[FretZone, ...] = ((0, 4), (3, 7), (5, 9), (7, 12), (10, 15))


@dataclass(frozen=True)
class OracleState:
    """One gold-selectable state shared by every target note in a group."""

    label: str
    string_offset: int | None = None
    fret_zone: FretZone | None = None
    neutral: bool = False


@dataclass(frozen=True)
class OracleApplication:
    """Pitch-preserving events and diagnostics from one oracle application."""

    events: tuple[TabEvent, ...]
    ambiguous_correct: int
    ambiguous_total: int
    dropped_impossible: int
    state_counts: tuple[tuple[str, int], ...]

    @property
    def ambiguous_accuracy(self) -> float:
        return self.ambiguous_correct / self.ambiguous_total


def track_groups(target_indices: set[int]) -> tuple[tuple[int, ...], ...]:
    """Return one group containing every target note in a track."""

    if not target_indices:
        return ()
    return (tuple(sorted(target_indices)),)


def fixed_window_groups(
    events: Sequence[TabEvent],
    target_indices: set[int],
    window_s: float,
    *,
    cluster_gap_s: float = 0.080,
) -> tuple[tuple[int, ...], ...]:
    """Assign onset clusters to fixed windows without splitting a cluster.

    The first onset in a simultaneous cluster chooses the fixed window.  Notes
    later in that cluster remain in the same window even when their timestamp
    crosses the nominal boundary.
    """

    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if cluster_gap_s < 0:
        raise ValueError("cluster_gap_s cannot be negative")
    if not target_indices:
        return ()
    if min(target_indices) < 0 or max(target_indices) >= len(events):
        raise IndexError("target index outside event sequence")

    ordered = sorted(range(len(events)), key=lambda index: (events[index].onset_s, index))
    clusters: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        previous = clusters[-1][-1]
        if events[index].onset_s - events[previous].onset_s <= cluster_gap_s:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    buckets: dict[int, list[int]] = {}
    for cluster in clusters:
        selected = [index for index in cluster if index in target_indices]
        if not selected:
            continue
        bucket = math.floor(events[cluster[0]].onset_s / window_s)
        buckets.setdefault(bucket, []).extend(selected)
    return tuple(tuple(sorted(buckets[key])) for key in sorted(buckets))


def apply_gold_oracle(
    predicted: Sequence[TabEvent],
    candidate_ranks: Sequence[Sequence[RankedCandidate]],
    gold_by_index: Mapping[int, TabEvent],
    groups: Sequence[Sequence[int]],
    *,
    strategy: OracleStrategy,
    cfg: GuitarConfig | None = None,
) -> OracleApplication:
    """Choose the best shared state per group using gold positions.

    A missing shifted/zone candidate is discarded from the diagnostic output;
    the event's MIDI pitch is never changed or replaced by a nearby pitch.  A
    baseline-equivalent state is always available, so aggregate ambiguous-note
    correctness cannot be worse than the unmodified decode.
    """

    cfg = cfg or GuitarConfig()
    if len(candidate_ranks) != len(predicted):
        raise ValueError("candidate ranks must align one-to-one with predicted events")
    if not gold_by_index:
        raise ValueError("gold oracle requires at least one target note")
    if set(index for group in groups for index in group) != set(gold_by_index):
        raise ValueError("oracle groups must cover every gold target exactly once")

    states = _states(strategy)
    replacements: dict[int, Candidate | None] = {}
    state_counts: Counter[str] = Counter()
    baseline_correct = 0
    oracle_correct = 0

    for group in groups:
        if len(set(group)) != len(group):
            raise ValueError("oracle group contains duplicate indices")
        baseline_correct += sum(
            _matches(
                Candidate(predicted[index].string_idx, predicted[index].fret),
                gold_by_index[index],
            )
            for index in group
        )
        choices: list[tuple[int, int, OracleState, dict[int, Candidate | None]]] = []
        for order, state in enumerate(states):
            positions = {
                index: _candidate_for_state(predicted[index], candidate_ranks[index], state, cfg)
                for index in group
            }
            correct = sum(_matches(positions[index], gold_by_index[index]) for index in group)
            choices.append((correct, -order, state, positions))
        correct, _tie_break, state, positions = max(choices, key=lambda item: item[:2])
        oracle_correct += correct
        state_counts[state.label] += 1
        replacements.update(positions)

    if oracle_correct < baseline_correct:
        raise AssertionError("gold oracle scored below the available baseline state")

    out: list[TabEvent] = []
    dropped = 0
    for index, event in enumerate(predicted):
        if index not in replacements:
            out.append(event)
            continue
        candidate = replacements[index]
        if candidate is None:
            dropped += 1
            continue
        if candidate not in _playable_candidates(candidate_ranks[index]):
            raise AssertionError("oracle selected a candidate absent from the playable ranking")
        updated = replace(event, string_idx=candidate.string_idx, fret=candidate.fret)
        if updated.pitch_midi != event.pitch_midi:
            raise AssertionError("oracle changed MIDI pitch")
        if updated.pitch_midi != cfg.tuning_midi[updated.string_idx] + updated.fret:
            raise AssertionError("oracle selected a pitch-inconsistent candidate")
        out.append(updated)

    return OracleApplication(
        tuple(out),
        oracle_correct,
        len(gold_by_index),
        dropped,
        tuple(sorted(state_counts.items())),
    )


def _states(strategy: OracleStrategy) -> tuple[OracleState, ...]:
    neutral = OracleState("neutral", neutral=True)
    if strategy == "offset":
        # Prefer the baseline-equivalent zero state on exact ties.
        return tuple(
            OracleState(f"offset_{offset:+d}", string_offset=offset) for offset in (0, -1, 1)
        )
    if strategy == "fret_zone":
        return (
            neutral,
            *(OracleState(f"zone_{low}_{high}", fret_zone=(low, high)) for low, high in FRET_ZONES),
        )
    if strategy == "joint":
        return (
            neutral,
            *(
                OracleState(
                    f"offset_{offset:+d}_zone_{low}_{high}",
                    string_offset=offset,
                    fret_zone=(low, high),
                )
                for offset in STRING_OFFSETS
                for low, high in FRET_ZONES
            ),
        )
    raise ValueError(f"unknown oracle strategy: {strategy}")


def _candidate_for_state(
    baseline: TabEvent,
    ranked: Sequence[RankedCandidate],
    state: OracleState,
    cfg: GuitarConfig,
) -> Candidate | None:
    candidates = _playable_candidates(ranked)
    if state.neutral:
        return Candidate(baseline.string_idx, baseline.fret)

    selected = list(candidates)
    if state.string_offset is not None:
        target_string = baseline.string_idx + state.string_offset
        selected = [candidate for candidate in selected if candidate.string_idx == target_string]
    if state.fret_zone is not None:
        low, high = state.fret_zone
        selected = [candidate for candidate in selected if low <= candidate.fret <= high]
    if not selected:
        return None

    rank_order = {
        (candidate.string_idx, candidate.fret): index for index, candidate in enumerate(ranked)
    }
    center = sum(state.fret_zone) / 2.0 if state.fret_zone is not None else float(baseline.fret)
    choice = min(
        selected,
        key=lambda candidate: (
            rank_order[(candidate.string_idx, candidate.fret)],
            abs(candidate.fret - center),
            candidate.fret,
            candidate.string_idx,
        ),
    )
    if choice.fret < cfg.capo or choice.fret > cfg.max_fret:
        return None
    return choice


def _playable_candidates(ranked: Sequence[RankedCandidate]) -> tuple[Candidate, ...]:
    return tuple(Candidate(candidate.string_idx, candidate.fret) for candidate in ranked)


def _matches(candidate: Candidate | None, gold: TabEvent) -> int:
    return int(
        candidate is not None
        and (candidate.string_idx, candidate.fret) == (gold.string_idx, gold.fret)
    )


__all__ = [
    "FRET_ZONES",
    "STRING_OFFSETS",
    "OracleApplication",
    "OracleState",
    "apply_gold_oracle",
    "fixed_window_groups",
    "track_groups",
]
