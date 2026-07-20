from __future__ import annotations

from dataclasses import replace

import pytest

from tabvision.assist.editing import (
    AssistOptions,
    EditSession,
    cycle_candidate_edit,
    matched_motif_previews,
    move_phrase_edit,
    phrase_alternatives,
)
from tabvision.eval.string_assignment import DecodedPath, RankedCandidate
from tabvision.types import TabEvent


def _event(pitch: int, onset: float, string: int, fret: int) -> TabEvent:
    return TabEvent(onset, 0.25, string, fret, pitch, 0.8)


def test_candidate_cycle_is_pitch_preserving_and_undoable() -> None:
    events = (_event(64, 0.0, 5, 0),)
    rankings = (
        RankedCandidate(5, 0, 0.0),
        RankedCandidate(4, 5, 1.0),
        RankedCandidate(3, 9, 2.0),
    )
    batch = cycle_candidate_edit(events, 0, rankings)
    session = EditSession(events)

    changed = session.apply(batch)

    assert (changed[0].string_idx, changed[0].fret, changed[0].pitch_midi) == (4, 5, 64)
    assert session.undo() == events


def test_reject_changes_nothing_and_atomic_failure_changes_nothing() -> None:
    events = (_event(64, 0.0, 5, 0), _event(65, 0.5, 5, 1))
    batch = move_phrase_edit(events, (0, 1), string_delta=-1)
    session = EditSession(events)
    assert session.reject(batch) == events

    stale = replace(
        batch,
        edits=(replace(batch.edits[0], before_fret=99), *batch.edits[1:]),
    )
    with pytest.raises(ValueError, match="precondition"):
        session.apply(stale)
    assert session.events == events


def test_move_phrase_fails_closed_if_any_note_is_unplayable() -> None:
    events = (_event(40, 0.0, 0, 0), _event(64, 0.5, 5, 0))
    with pytest.raises(ValueError, match="complete phrase"):
        move_phrase_edit(events, (0, 1), string_delta=-1)


def test_phrase_alternatives_are_unique_playable_and_pitch_identical() -> None:
    baseline = (_event(64, 0.0, 5, 0), _event(65, 0.5, 5, 1))
    alternative = (_event(64, 0.0, 4, 5), _event(65, 0.5, 4, 6))
    invalid_pitch = (_event(63, 0.0, 4, 4), _event(65, 0.5, 4, 6))
    paths = (
        DecodedPath(baseline, 0.0, 0.0),
        DecodedPath(alternative, 1.0, 1.0),
        DecodedPath(alternative, 2.0, 2.0),
        DecodedPath(invalid_pitch, 3.0, 3.0),
    )

    observed = phrase_alternatives(paths, 0, 2)

    assert observed == (alternative,)


def test_motif_propagation_returns_preview_only_for_exact_repeat() -> None:
    source = [_event(60, 0.0, 3, 5), _event(62, 0.5, 3, 7), _event(64, 1.0, 3, 9)]
    repeat = [_event(60, 2.0, 4, 1), _event(62, 2.5, 4, 3), _event(64, 3.0, 4, 5)]
    near_miss = [_event(60, 4.0, 4, 1), _event(62, 4.4, 4, 3), _event(64, 5.0, 4, 5)]

    previews = matched_motif_previews((*source, *repeat, *near_miss), (0, 1, 2))

    assert [preview.target_indices for preview in previews] == [(3, 4, 5)]


def test_optional_modes_are_explicit_and_default_off() -> None:
    options = AssistOptions()
    assert options.calibration_mode is None
    assert options.starting_hand_position is None
    assert options.score_reference_format is None
    assert options.private_prior_opt_in is False
