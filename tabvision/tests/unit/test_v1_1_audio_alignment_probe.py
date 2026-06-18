"""Tests for the v1.1 UT-Austin audio alignment probe helpers."""

from __future__ import annotations

import pytest

from scripts.eval.v1_1_audio_alignment_probe import (
    count_pitch_time_matches,
    estimate_audio_alignment,
    estimate_global_alignment,
    events_from_json,
    events_to_json,
    score_events,
    shift_audio_events,
)
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


def _audio(t: float, pitch: int) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.2,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=0.9,
    )


def _tab(t: float, string_idx: int, fret: int, pitch: int) -> TabEvent:
    return TabEvent(
        onset_s=t,
        duration_s=0.2,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def test_estimate_audio_alignment_finds_pitch_and_time_shift() -> None:
    gold = [_tab(1.50, 5, 0, 64), _tab(2.00, 5, 1, 65)]
    events = [_audio(0.25, 65), _audio(0.75, 66)]

    choice, scores = estimate_audio_alignment(
        events,
        gold,
        tolerance_s=0.05,
        max_abs_pitch_shift=2,
        max_abs_time_shift_s=2.0,
        time_step_s=0.25,
    )

    assert choice.pitch_shift == -1
    assert choice.time_shift_s == pytest.approx(1.25)
    assert choice.matches == 2
    assert scores[(-1, 1.25)] == 2


def test_count_pitch_time_matches_is_greedy_one_to_one() -> None:
    gold = [_tab(1.0, 5, 0, 64)]
    events = [_audio(1.0, 64), _audio(1.01, 64)]

    matches = count_pitch_time_matches(
        events,
        gold,
        pitch_shift=0,
        time_shift_s=0.0,
        tolerance_s=0.05,
    )

    assert matches == 1


def test_estimate_global_alignment_sums_clip_score_maps() -> None:
    gold_a = [_tab(1.50, 5, 0, 64)]
    gold_b = [_tab(2.50, 5, 1, 65)]
    events_a = [_audio(0.25, 65)]
    events_b = [_audio(1.25, 66)]

    _choice_a, scores_a = estimate_audio_alignment(
        events_a,
        gold_a,
        tolerance_s=0.05,
        max_abs_pitch_shift=2,
        max_abs_time_shift_s=2.0,
        time_step_s=0.25,
    )
    _choice_b, scores_b = estimate_audio_alignment(
        events_b,
        gold_b,
        tolerance_s=0.05,
        max_abs_pitch_shift=2,
        max_abs_time_shift_s=2.0,
        time_step_s=0.25,
    )

    choice = estimate_global_alignment([scores_a, scores_b])

    assert choice.pitch_shift == -1
    assert choice.time_shift_s == pytest.approx(1.25)
    assert choice.matches == 2


def test_event_cache_json_round_trip_keeps_score_fields() -> None:
    events = [_audio(0.25, 65)]

    restored = events_from_json(events_to_json(events))

    assert restored == events


def test_score_events_reports_perfect_when_audio_matches_gold() -> None:
    cfg = GuitarConfig()
    gold = [_tab(1.0, 5, 0, 64)]
    events = [_audio(1.0, 64)]

    score = score_events(events, gold, cfg=cfg, onset_tolerance_s=0.05)

    assert score.onset_f1 == pytest.approx(1.0)
    assert score.pitch_f1 == pytest.approx(1.0)
    assert score.tab_f1 == pytest.approx(1.0)
    assert score.oracle_tab_f1 == pytest.approx(1.0)


def test_shift_audio_events_does_not_mutate_original() -> None:
    original = [_audio(0.25, 65)]

    shifted = shift_audio_events(original, pitch_shift=-1, time_shift_s=1.25)

    assert shifted[0].pitch_midi == 64
    assert shifted[0].onset_s == pytest.approx(1.50)
    assert original[0].pitch_midi == 65
    assert original[0].onset_s == pytest.approx(0.25)
