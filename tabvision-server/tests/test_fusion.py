"""Tests for fusion_engine module."""
import pytest
from app.fusion_engine import fuse_audio_only, get_confidence_level, TabNote
from app.audio_pipeline import DetectedNote


class TestGetConfidenceLevel:
    """Tests for confidence level mapping."""

    def test_high_confidence(self):
        """Confidence > 0.8 should return 'high'."""
        assert get_confidence_level(0.85) == "high"
        assert get_confidence_level(0.9) == "high"
        assert get_confidence_level(1.0) == "high"

    def test_medium_confidence(self):
        """Confidence 0.5-0.8 should return 'medium'."""
        assert get_confidence_level(0.5) == "medium"
        assert get_confidence_level(0.65) == "medium"
        assert get_confidence_level(0.8) == "medium"

    def test_low_confidence(self):
        """Confidence < 0.5 should return 'low'."""
        assert get_confidence_level(0.0) == "low"
        assert get_confidence_level(0.3) == "low"
        assert get_confidence_level(0.49) == "low"


class TestFuseAudioOnly:
    """Tests for audio-only fusion."""

    def test_fuse_single_note(self):
        """Single detected note should produce single TabNote."""
        detected_notes = [
            DetectedNote(
                start_time=1.0,
                end_time=1.5,
                midi_note=69,  # A4
                confidence=0.85,
            )
        ]

        result = fuse_audio_only(detected_notes, capo_fret=0)

        assert len(result) == 1
        tab_note = result[0]
        assert tab_note.timestamp == 1.0
        assert tab_note.string == 1  # Lowest fret is string 1, fret 5
        assert tab_note.fret == 5
        assert tab_note.confidence == 0.85
        assert tab_note.confidence_level == "high"

    def test_fuse_multiple_notes(self):
        """Multiple detected notes should produce multiple TabNotes."""
        detected_notes = [
            DetectedNote(start_time=0.0, end_time=0.5, midi_note=64, confidence=0.9),  # E4
            DetectedNote(start_time=0.5, end_time=1.0, midi_note=67, confidence=0.7),  # G4
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.4),  # A4
        ]

        result = fuse_audio_only(detected_notes, capo_fret=0)

        assert len(result) == 3
        # First note: E4 (MIDI 64) = open string 1
        assert result[0].fret == 0
        assert result[0].string == 1
        assert result[0].confidence_level == "high"
        # Second note: G4 (MIDI 67) = string 1 fret 3
        assert result[1].fret == 3
        assert result[1].string == 1
        assert result[1].confidence_level == "medium"
        # Third note: A4 (MIDI 69) = string 1 fret 5
        assert result[2].fret == 5
        assert result[2].string == 1
        assert result[2].confidence_level == "low"

    def test_fuse_with_capo(self):
        """Capo should affect fret positions."""
        # E4 (MIDI 64) with capo at fret 2
        # Open string 1 is no longer available, so it goes to string 2 fret 5
        detected_notes = [
            DetectedNote(start_time=0.0, end_time=0.5, midi_note=64, confidence=0.9),
        ]

        result = fuse_audio_only(detected_notes, capo_fret=2)

        assert len(result) == 1
        # With capo at 2, fret 0 is excluded, next lowest is string 2 fret 5
        assert result[0].fret == 5
        assert result[0].string == 2

    def test_fuse_skips_out_of_range(self):
        """Notes outside guitar range should be skipped."""
        detected_notes = [
            DetectedNote(start_time=0.0, end_time=0.5, midi_note=30, confidence=0.9),  # Too low
            DetectedNote(start_time=0.5, end_time=1.0, midi_note=69, confidence=0.8),  # Valid
        ]

        result = fuse_audio_only(detected_notes, capo_fret=0)

        assert len(result) == 1
        assert result[0].midi_note == 69

    def test_fuse_empty_input(self):
        """Empty input should return empty output."""
        result = fuse_audio_only([], capo_fret=0)
        assert result == []

    def test_tab_note_has_id(self):
        """Each TabNote should have a unique ID."""
        detected_notes = [
            DetectedNote(start_time=0.0, end_time=0.5, midi_note=69, confidence=0.9),
            DetectedNote(start_time=0.5, end_time=1.0, midi_note=69, confidence=0.9),
        ]

        result = fuse_audio_only(detected_notes, capo_fret=0)

        assert len(result) == 2
        assert result[0].id != result[1].id  # IDs should be unique
        assert result[0].id is not None
