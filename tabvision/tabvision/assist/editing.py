"""Atomic, pitch-preserving editing helpers for assisted tablature review."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from tabvision.eval.string_assignment import DecodedPath
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig, TabEvent


class PositionLike(Protocol):
    string_idx: int
    fret: int


@dataclass(frozen=True)
class PositionEdit:
    event_index: int
    before_string: int
    before_fret: int
    after_string: int
    after_fret: int


@dataclass(frozen=True)
class BatchEdit:
    edits: tuple[PositionEdit, ...]
    reason: str

    def __post_init__(self) -> None:
        indices = [edit.event_index for edit in self.edits]
        if not self.edits:
            raise ValueError("a batch edit must change at least one event")
        if len(set(indices)) != len(indices):
            raise ValueError("a batch edit cannot change an event twice")


@dataclass(frozen=True)
class MotifPreview:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


@dataclass(frozen=True)
class AssistOptions:
    """Optional side information; every mode is disabled by default."""

    calibration_mode: Literal["six_open_strings", "known_chord"] | None = None
    starting_hand_position: int | None = None
    score_reference_format: Literal["midi", "musicxml", "guitarpro", "chord_chart"] | None = None
    licensed_song_reference: bool = False
    private_prior_opt_in: bool = False

    def __post_init__(self) -> None:
        if self.starting_hand_position is not None and self.starting_hand_position < 0:
            raise ValueError("starting hand position must be non-negative")


class EditSession:
    """In-memory atomic editor with one-step snapshots and deterministic undo."""

    def __init__(self, events: Sequence[TabEvent], cfg: GuitarConfig | None = None) -> None:
        self._events = tuple(events)
        self._history: list[tuple[TabEvent, ...]] = []
        self._cfg = cfg or GuitarConfig()

    @property
    def events(self) -> tuple[TabEvent, ...]:
        return self._events

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def apply(self, batch: BatchEdit) -> tuple[TabEvent, ...]:
        output = list(self._events)
        for edit in batch.edits:
            if not 0 <= edit.event_index < len(output):
                raise IndexError(f"event index is out of range: {edit.event_index}")
            event = output[edit.event_index]
            if (event.string_idx, event.fret) != (edit.before_string, edit.before_fret):
                raise ValueError("batch precondition does not match current phrase")
            playable = {
                (candidate.string_idx, candidate.fret)
                for candidate in candidate_positions(event.pitch_midi, self._cfg)
            }
            if (edit.after_string, edit.after_fret) not in playable:
                raise ValueError("batch edit changes pitch or creates an unplayable position")
        for edit in batch.edits:
            event = output[edit.event_index]
            output[edit.event_index] = replace(
                event,
                string_idx=edit.after_string,
                fret=edit.after_fret,
            )
        self._history.append(self._events)
        self._events = tuple(output)
        return self._events

    def reject(self, _batch: BatchEdit) -> tuple[TabEvent, ...]:
        return self._events

    def undo(self) -> tuple[TabEvent, ...]:
        if not self._history:
            raise RuntimeError("there is no accepted edit to undo")
        self._events = self._history.pop()
        return self._events


def cycle_candidate_edit(
    events: Sequence[TabEvent],
    event_index: int,
    rankings: Sequence[PositionLike],
    *,
    direction: int = 1,
    cfg: GuitarConfig | None = None,
) -> BatchEdit:
    """Create one pitch-preserving candidate-cycle edit from decoder rankings."""

    if direction not in {-1, 1}:
        raise ValueError("candidate-cycle direction must be -1 or 1")
    if not 0 <= event_index < len(events):
        raise IndexError("event index is out of range")
    event = events[event_index]
    playable = {
        (candidate.string_idx, candidate.fret)
        for candidate in candidate_positions(event.pitch_midi, cfg or GuitarConfig())
    }
    ordered = [
        (int(item.string_idx), int(item.fret))
        for item in rankings
        if (int(item.string_idx), int(item.fret)) in playable
    ]
    ordered = list(dict.fromkeys(ordered))
    current = (event.string_idx, event.fret)
    if current not in ordered or len(ordered) < 2:
        raise ValueError("decoder rankings do not contain a distinct current alternative")
    selected = ordered[(ordered.index(current) + direction) % len(ordered)]
    return BatchEdit(
        (
            PositionEdit(
                event_index,
                event.string_idx,
                event.fret,
                selected[0],
                selected[1],
            ),
        ),
        "cycle_candidate",
    )


def move_phrase_edit(
    events: Sequence[TabEvent],
    event_indices: Sequence[int],
    *,
    string_delta: int,
    cfg: GuitarConfig | None = None,
) -> BatchEdit:
    """Move a complete phrase one string while preserving every MIDI pitch."""

    if string_delta not in {-1, 1}:
        raise ValueError("phrase movement must be exactly one string up or down")
    config = cfg or GuitarConfig()
    edits: list[PositionEdit] = []
    for index in sorted(set(event_indices)):
        if not 0 <= index < len(events):
            raise IndexError("event index is out of range")
        event = events[index]
        target_string = event.string_idx + string_delta
        target = next(
            (
                candidate
                for candidate in candidate_positions(event.pitch_midi, config)
                if candidate.string_idx == target_string
            ),
            None,
        )
        if target is None:
            raise ValueError("the complete phrase cannot move one string and remain playable")
        edits.append(
            PositionEdit(
                index,
                event.string_idx,
                event.fret,
                target.string_idx,
                target.fret,
            )
        )
    return BatchEdit(tuple(edits), f"move_phrase_{string_delta:+d}")


def phrase_alternatives(
    paths: Sequence[DecodedPath],
    start_index: int,
    end_index: int,
    *,
    cfg: GuitarConfig | None = None,
    limit: int = 3,
) -> tuple[tuple[TabEvent, ...], ...]:
    """Slice unique, playable, pitch-identical alternatives from decoder K-best paths."""

    if limit < 1 or not paths or not 0 <= start_index < end_index:
        return ()
    config = cfg or GuitarConfig()
    baseline = paths[0].events[start_index:end_index]
    if len(baseline) != end_index - start_index:
        return ()
    baseline_key = _position_key(baseline)
    seen = {baseline_key}
    output: list[tuple[TabEvent, ...]] = []
    for path in paths[1:]:
        candidate = tuple(path.events[start_index:end_index])
        if len(candidate) != len(baseline):
            continue
        if any(
            event.pitch_midi != reference.pitch_midi
            or abs(event.onset_s - reference.onset_s) > 1.0e-9
            for event, reference in zip(candidate, baseline, strict=True)
        ):
            continue
        if any(
            (event.string_idx, event.fret)
            not in {
                (position.string_idx, position.fret)
                for position in candidate_positions(event.pitch_midi, config)
            }
            for event in candidate
        ):
            continue
        key = _position_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= limit:
            break
    return tuple(output)


def matched_motif_previews(
    events: Sequence[TabEvent],
    source_indices: Sequence[int],
    *,
    onset_quantum_s: float = 0.01,
) -> tuple[MotifPreview, ...]:
    """Return exact repeated-motif previews; never apply them automatically."""

    source = tuple(sorted(set(source_indices)))
    if not source or source != tuple(range(source[0], source[-1] + 1)):
        raise ValueError("motif source must be one non-empty contiguous event range")
    if source[-1] >= len(events) or onset_quantum_s <= 0.0:
        raise ValueError("invalid motif source or onset quantum")
    length = len(source)
    signature = _motif_signature([events[index] for index in source], onset_quantum_s)
    output: list[MotifPreview] = []
    source_set = set(source)
    for start in range(0, len(events) - length + 1):
        target = tuple(range(start, start + length))
        if source_set & set(target):
            continue
        if _motif_signature([events[index] for index in target], onset_quantum_s) == signature:
            output.append(MotifPreview(source, target))
    return tuple(output)


def _position_key(events: Sequence[TabEvent]) -> tuple[tuple[int, int], ...]:
    return tuple((event.string_idx, event.fret) for event in events)


def _motif_signature(
    events: Sequence[TabEvent], onset_quantum_s: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pitches = tuple(event.pitch_midi for event in events)
    intervals = tuple(
        int(round((right.onset_s - left.onset_s) / onset_quantum_s))
        for left, right in zip(events, events[1:], strict=False)
    )
    return pitches, intervals


__all__ = [
    "AssistOptions",
    "BatchEdit",
    "EditSession",
    "MotifPreview",
    "PositionEdit",
    "cycle_candidate_edit",
    "matched_motif_previews",
    "move_phrase_edit",
    "phrase_alternatives",
]
