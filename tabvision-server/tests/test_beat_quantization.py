"""Tests for beat quantization module."""
import pytest
from unittest.mock import patch, MagicMock

from app.beat_quantization import (
    QuantizationConfig,
    build_grid,
    quantize_notes,
)
from app.fusion_engine import TabNote


def _make_note(timestamp, confidence_level="medium", end_time=None, note_id=None):
    """Helper to create a TabNote with minimal required fields."""
    return TabNote(
        id=note_id or f"note-{timestamp}",
        timestamp=timestamp,
        string=1,
        fret=5,
        confidence=0.8 if confidence_level == "high" else 0.5,
        confidence_level=confidence_level,
        midi_note=69,
        end_time=end_time,
    )


class TestQuantizationConfig:
    """Tests for QuantizationConfig defaults."""

    def test_defaults(self):
        config = QuantizationConfig()
        assert config.enabled is True
        assert config.subdivision == 16
        assert config.max_snap_distance == 0.08
        assert config.snap_strength_low == 1.0
        assert config.snap_strength_high == 0.3

    def test_custom_values(self):
        config = QuantizationConfig(
            enabled=False, subdivision=8, max_snap_distance=0.1
        )
        assert config.enabled is False
        assert config.subdivision == 8
        assert config.max_snap_distance == 0.1


class TestBuildGrid:
    """Tests for grid construction from beat positions."""

    def test_uniform_grid_from_tempo(self):
        """When no beats provided, build uniform grid from tempo."""
        grid = build_grid(tempo=120.0, beat_times=[], duration=2.0, subdivision=4)
        # At 120 BPM, quarter notes are 0.5s apart
        assert grid[0] == 0.0
        assert abs(grid[1] - 0.5) < 0.001
        assert len(grid) >= 4

    def test_grid_from_beats_subdivision_4(self):
        """Quarter-note grid: one point per beat."""
        beats = [0.0, 0.5, 1.0, 1.5]
        grid = build_grid(tempo=120.0, beat_times=beats, duration=2.0, subdivision=4)
        # Should include at least the beat positions
        for b in beats:
            assert any(abs(g - b) < 0.001 for g in grid), f"Beat {b} not in grid"

    def test_grid_from_beats_subdivision_16(self):
        """16th-note grid: 4 subdivisions per beat."""
        beats = [0.0, 0.5, 1.0]
        grid = build_grid(tempo=120.0, beat_times=beats, duration=1.5, subdivision=16)
        # Between 0.0 and 0.5, should have points at 0.0, 0.125, 0.25, 0.375
        expected_subs = [0.0, 0.125, 0.25, 0.375]
        for t in expected_subs:
            assert any(abs(g - t) < 0.001 for g in grid), f"Subdivision {t} not in grid"

    def test_grid_extends_beyond_beats(self):
        """Grid should extend before first and after last beat."""
        beats = [1.0, 1.5, 2.0]
        grid = build_grid(tempo=120.0, beat_times=beats, duration=3.0, subdivision=4)
        # Should have grid points before 1.0
        assert any(g < 1.0 for g in grid)
        # Should have grid points after 2.0
        assert any(g > 2.0 for g in grid)

    def test_single_beat_falls_back_to_tempo(self):
        """With <2 beats, should fall back to tempo-based grid."""
        grid = build_grid(tempo=120.0, beat_times=[1.0], duration=3.0, subdivision=4)
        # Should still generate a valid grid
        assert len(grid) > 0
        assert grid[0] == 0.0

    def test_grid_is_sorted(self):
        """Grid should always be sorted."""
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        grid = build_grid(tempo=120.0, beat_times=beats, duration=2.5, subdivision=16)
        assert grid == sorted(grid)


class TestQuantizeNotes:
    """Tests for the note quantization function."""

    @patch('app.beat_quantization.detect_tempo')
    def test_empty_notes_returns_empty(self, mock_detect):
        """Quantizing empty list returns empty list."""
        result = quantize_notes([], "audio.wav")
        assert result == []
        mock_detect.assert_not_called()

    @patch('app.beat_quantization.detect_tempo')
    def test_disabled_returns_original(self, mock_detect):
        """When disabled, notes are returned unchanged."""
        notes = [_make_note(1.0)]
        config = QuantizationConfig(enabled=False)
        result = quantize_notes(notes, "audio.wav", config)
        assert result[0].timestamp == 1.0
        mock_detect.assert_not_called()

    @patch('app.beat_quantization.detect_tempo')
    def test_snaps_note_to_grid(self, mock_detect):
        """Note near a grid point should snap towards it."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0, 1.5])
        # At 120 BPM with subdivision=16, grid points at 0, 0.125, 0.25, 0.375, 0.5...
        # Note at 0.13 should snap toward 0.125
        notes = [_make_note(0.13, confidence_level="low")]
        config = QuantizationConfig(snap_strength_low=1.0)
        result = quantize_notes(notes, "audio.wav", config)
        # Low confidence, full snap strength → should be very close to 0.125
        assert abs(result[0].timestamp - 0.125) < 0.01

    @patch('app.beat_quantization.detect_tempo')
    def test_high_confidence_snaps_less(self, mock_detect):
        """High-confidence notes should snap less aggressively."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0])
        notes = [_make_note(0.13, confidence_level="high")]
        config = QuantizationConfig(snap_strength_high=0.3)
        result = quantize_notes(notes, "audio.wav", config)
        # Should move toward 0.125 but only 30%
        # Distance = 0.13 - 0.125 = 0.005, move 30% = 0.0015
        # New = 0.13 - 0.0015 = 0.1285
        assert abs(result[0].timestamp - 0.1285) < 0.005

    @patch('app.beat_quantization.detect_tempo')
    def test_far_notes_not_snapped(self, mock_detect):
        """Notes too far from any grid point should not be snapped."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0])
        # Grid at 16th notes: 0, 0.125, 0.25, ...
        # Note at 0.0625 is exactly between two grid points = 0.0625 away
        notes = [_make_note(0.0625, confidence_level="low")]
        config = QuantizationConfig(max_snap_distance=0.05)
        result = quantize_notes(notes, "audio.wav", config)
        # 0.0625 > max_snap_distance of 0.05, should not snap
        assert result[0].timestamp == 0.0625

    @patch('app.beat_quantization.detect_tempo')
    def test_preserves_note_duration(self, mock_detect):
        """Quantization should preserve note duration (end_time - timestamp)."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0])
        notes = [_make_note(0.13, confidence_level="low", end_time=0.63)]
        config = QuantizationConfig(snap_strength_low=1.0)
        result = quantize_notes(notes, "audio.wav", config)
        original_duration = 0.63 - 0.13  # 0.5s
        new_duration = result[0].end_time - result[0].timestamp
        assert abs(new_duration - original_duration) < 0.001

    @patch('app.beat_quantization.detect_tempo')
    def test_preserves_note_attributes(self, mock_detect):
        """Quantization should preserve all non-timing attributes."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0])
        note = TabNote(
            id="note-1",
            timestamp=0.13,
            string=3,
            fret=7,
            confidence=0.85,
            confidence_level="high",
            midi_note=64,
            technique="bend",
            is_part_of_chord=True,
            chord_id="chord-1",
            video_matched=True,
            audio_confidence=0.9,
            video_confidence=0.7,
            pitch_bend=1.5,
        )
        result = quantize_notes([note], "audio.wav")
        r = result[0]
        assert r.id == "note-1"
        assert r.string == 3
        assert r.fret == 7
        assert r.confidence == 0.85
        assert r.confidence_level == "high"
        assert r.midi_note == 64
        assert r.technique == "bend"
        assert r.is_part_of_chord is True
        assert r.chord_id == "chord-1"
        assert r.video_matched is True
        assert r.audio_confidence == 0.9
        assert r.video_confidence == 0.7
        assert r.pitch_bend == 1.5

    @patch('app.beat_quantization.detect_tempo')
    def test_beat_detection_failure_returns_original(self, mock_detect):
        """If beat detection fails, return notes unchanged."""
        mock_detect.side_effect = RuntimeError("librosa failed")
        notes = [_make_note(1.0), _make_note(2.0)]
        result = quantize_notes(notes, "audio.wav")
        assert len(result) == 2
        assert result[0].timestamp == 1.0
        assert result[1].timestamp == 2.0

    @patch('app.beat_quantization.detect_tempo')
    def test_medium_confidence_uses_average_strength(self, mock_detect):
        """Medium confidence notes use average of high and low strength."""
        mock_detect.return_value = (120.0, [0.0, 0.5, 1.0])
        notes = [_make_note(0.13, confidence_level="medium")]
        config = QuantizationConfig(snap_strength_low=1.0, snap_strength_high=0.3)
        result = quantize_notes(notes, "audio.wav", config)
        # Medium strength = (1.0 + 0.3) / 2 = 0.65
        # Nearest grid = 0.125, distance = 0.005
        # New = 0.13 + (0.125 - 0.13) * 0.65 = 0.13 - 0.00325 = 0.12675
        assert abs(result[0].timestamp - 0.12675) < 0.005
