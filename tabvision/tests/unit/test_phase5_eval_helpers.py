"""Unit tests for Phase 5 acceptance-eval timing/alignment helpers."""

from __future__ import annotations

import pytest

from tabvision.eval.metrics import tab_f1
from tabvision.types import TabEvent
from tests.eval.test_phase5_eval import (
    _align_gold_to_audio_only,
    _find_best_pitch_offset,
    _gold_notes_to_tab_events,
    _parse_legacy_tab_text,
)


def _event(t: float, string_idx: int, fret: int, pitch: int) -> TabEvent:
    return TabEvent(
        onset_s=t,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def test_gold_conversion_uses_bpm_for_seconds_and_computes_midi():
    notes = [
        {"string": 1, "fret": 0, "beat": 1.0},
        {"string": 6, "fret": 3, "beat": 2.0},
    ]

    events = _gold_notes_to_tab_events(notes, bpm=120, video_duration_s=99.0)

    assert [e.onset_s for e in events] == [0.5, 1.0]
    assert [(e.string_idx, e.fret, e.pitch_midi) for e in events] == [
        (5, 0, 64),  # high E open
        (0, 3, 43),  # low E fret 3
    ]


def test_gold_conversion_falls_back_to_duration_when_bpm_missing():
    notes = [
        {"string": 1, "fret": 0, "beat": 2.0},
        {"string": 2, "fret": 1, "beat": 8.0},
    ]

    events = _gold_notes_to_tab_events(notes, bpm=None, video_duration_s=4.0)

    assert [e.onset_s for e in events] == [1.0, 4.0]


def test_gold_conversion_skips_muted_notes():
    notes = [
        {"string": 1, "fret": "X", "beat": 0.0},
        {"string": 1, "fret": 3, "beat": 1.0},
    ]

    events = _gold_notes_to_tab_events(notes, bpm=60, video_duration_s=10.0)

    assert len(events) == 1
    assert events[0].fret == 3
    assert events[0].pitch_midi == 67


def test_legacy_tab_parser_fallback_handles_tabs_and_muted_notes():
    notes = _parse_legacy_tab_text(
        """
e|--12-x-|
B|--3----|
"""
    )

    assert notes == [
        {"string": 1, "fret": 12, "beat": 0.5},
        {"string": 2, "fret": 3, "beat": 0.5},
        {"string": 1, "fret": "X", "beat": 1.0},
    ]


def test_find_best_pitch_offset_recovers_known_offset():
    gold = [_event(0.0, 5, 0, 64), _event(1.0, 5, 2, 66)]
    predicted = [_event(2.0, 5, 0, 64), _event(3.0, 5, 2, 66)]

    offset, matches = _find_best_pitch_offset(
        predicted=predicted,
        gold=gold,
        video_duration_s=5.0,
        tolerance_s=0.01,
        step_s=0.05,
    )

    assert offset == pytest.approx(2.0)
    assert matches == 2


def test_alignment_from_audio_only_is_reused_for_audio_video_scoring():
    gold = [_event(0.0, 5, 0, 64)]
    audio_only = [_event(1.25, 5, 0, 64)]
    audio_video = [_event(1.25, 5, 0, 64)]

    aligned_gold, offset, matches = _align_gold_to_audio_only(
        audio_only=audio_only,
        gold=gold,
        video_duration_s=3.0,
    )

    assert offset == pytest.approx(1.25)
    assert matches == 1
    assert tab_f1(audio_only, aligned_gold).f1 == 1.0
    assert tab_f1(audio_video, aligned_gold).f1 == 1.0
