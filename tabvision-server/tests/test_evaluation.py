"""Tests for evaluation helpers."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate_transcription import (
    parse_ground_truth_tabs,
    evaluate_accuracy,
    EvalMetrics,
)
from app.fusion_engine import TabNote


class TestParseGroundTruth:
    def test_parses_simple_tab(self):
        tabs = (
            "e|---5---|\n"
            "B|-------|\n"
            "G|-------|\n"
            "D|-------|\n"
            "A|-------|\n"
            "E|-------|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 1
        assert notes[0]['string'] == 1
        assert notes[0]['fret'] == 5

    def test_parses_chord(self):
        tabs = (
            "e|0--|\n"
            "B|1--|\n"
            "G|0--|\n"
            "D|2--|\n"
            "A|3--|\n"
            "E|---|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 5
        # All notes at same beat position
        beats = {n['beat'] for n in notes}
        assert len(beats) == 1

    def test_parses_two_digit_fret(self):
        tabs = (
            "e|--12--|\n"
            "B|------|\n"
            "G|------|\n"
            "D|------|\n"
            "A|------|\n"
            "E|------|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 1
        assert notes[0]['fret'] == 12

    def test_bar_lines_dont_add_time(self):
        """Bar lines are visual separators, not time markers."""
        tabs = (
            "e|--5--|--7--|\n"
            "B|-----|-----|\n"
            "G|-----|-----|\n"
            "D|-----|-----|\n"
            "A|-----|-----|\n"
            "E|-----|-----|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 2
        # 5 dashes between notes (2 before bar + 2 after bar + 1 for fret width) = 1.25 beats
        # Bar line itself adds no time
        assert notes[1]['beat'] - notes[0]['beat'] == pytest.approx(1.25, abs=0.01)


def _make_note(timestamp, string, fret, confidence=0.9, midi_note=69):
    """Helper to create a TabNote with all required fields."""
    return TabNote(
        id=f"test-{timestamp}-{string}-{fret}",
        timestamp=timestamp,
        string=string,
        fret=fret,
        confidence=confidence,
        confidence_level="high" if confidence > 0.8 else "medium",
        midi_note=midi_note,
    )


class TestEvaluateAccuracy:
    def test_perfect_match(self):
        # beat=1.0, video_duration=1.0 → gt_time = 1.0s, det at 1.0s = exact match
        detected = [_make_note(1.0, string=1, fret=5, midi_note=69)]
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        metrics = evaluate_accuracy(detected, ground_truth, time_tolerance=0.5, video_duration=1.0)
        assert metrics.exact_f1 == 1.0

    def test_no_detections(self):
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        metrics = evaluate_accuracy([], ground_truth, time_tolerance=0.5, video_duration=1.0)
        assert metrics.exact_recall == 0.0
        assert metrics.exact_fn == 1

    def test_false_positive(self):
        detected = [_make_note(1.0, string=1, fret=5, midi_note=69)]
        metrics = evaluate_accuracy(detected, [], time_tolerance=0.5, video_duration=1.0)
        assert metrics.exact_precision == 0.0
        assert metrics.exact_fp == 1

    def test_wrong_position_same_pitch(self):
        """Same MIDI note but different string/fret should be pitch match, not exact."""
        detected = [_make_note(1.0, string=2, fret=10, midi_note=69)]
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        metrics = evaluate_accuracy(detected, ground_truth, time_tolerance=0.5, video_duration=1.0)
        assert metrics.exact_tp == 0
        assert metrics.pitch_tp == 1
