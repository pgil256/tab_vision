"""Tests for enhanced accuracy features."""
import pytest
from app.audio_pipeline import (
    DetectedNote,
    AudioAnalysisConfig,
    group_notes_into_chords,
    detect_note_onsets,
    _filter_notes,
    _merge_consecutive_notes,
    _filter_harmonics,
)
from app.fusion_engine import (
    fuse_audio_only,
    fuse_audio_video,
    FusionConfig,
    TabNote,
    Chord,
    get_confidence_level,
    _select_best_position,
    _optimize_chord_positions,
    _detect_techniques,
)
from app.guitar_mapping import Position
from app.video_pipeline import (
    FingerPosition,
    HandObservation,
    VideoAnalysisConfig,
)
from app.fretboard_detection import (
    FretboardGeometry,
    FretboardDetectionConfig,
    map_finger_to_position,
    _calculate_detection_confidence,
)


class TestAudioAnalysisConfig:
    """Tests for AudioAnalysisConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = AudioAnalysisConfig()
        assert config.min_confidence == 0.3
        assert config.min_note_duration == 0.03
        assert config.filter_harmonics is True

    def test_custom_config(self):
        """Can create custom config."""
        config = AudioAnalysisConfig(
            min_confidence=0.5,
            min_note_duration=0.05,
            filter_harmonics=False
        )
        assert config.min_confidence == 0.5
        assert config.filter_harmonics is False


class TestDetectedNoteDuration:
    """Tests for DetectedNote.duration property."""

    def test_duration_calculation(self):
        """Duration is correctly calculated."""
        note = DetectedNote(
            start_time=1.0,
            end_time=2.5,
            midi_note=60,
            confidence=0.8,
            amplitude=0.7,
            pitch_bend=0.0
        )
        assert note.duration == 1.5


class TestGroupNotesIntoChords:
    """Tests for chord grouping."""

    def test_single_note_is_own_chord(self):
        """Single note forms its own chord."""
        notes = [
            DetectedNote(0.5, 1.0, 60, 0.8, 0.7, 0.0)
        ]
        chords = group_notes_into_chords(notes)
        assert len(chords) == 1
        assert len(chords[0]) == 1

    def test_simultaneous_notes_grouped(self):
        """Notes within tolerance are grouped."""
        notes = [
            DetectedNote(1.0, 1.5, 60, 0.8, 0.7, 0.0),
            DetectedNote(1.02, 1.5, 64, 0.8, 0.7, 0.0),
            DetectedNote(1.04, 1.5, 67, 0.8, 0.7, 0.0),
        ]
        chords = group_notes_into_chords(notes, tolerance=0.05)
        assert len(chords) == 1
        assert len(chords[0]) == 3

    def test_sequential_notes_separate_chords(self):
        """Notes far apart form separate chords."""
        notes = [
            DetectedNote(1.0, 1.5, 60, 0.8, 0.7, 0.0),
            DetectedNote(2.0, 2.5, 64, 0.8, 0.7, 0.0),
        ]
        chords = group_notes_into_chords(notes, tolerance=0.05)
        assert len(chords) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert group_notes_into_chords([]) == []


class TestFilterNotes:
    """Tests for note filtering."""

    def test_filters_low_confidence(self):
        """Low confidence notes are filtered."""
        config = AudioAnalysisConfig(min_confidence=0.5)
        notes = [
            DetectedNote(0.0, 1.0, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.0, 2.0, 60, 0.3, 0.3, 0.0),
        ]
        filtered = _filter_notes(notes, config)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.8

    def test_filters_short_notes(self):
        """Very short notes are filtered."""
        config = AudioAnalysisConfig(min_note_duration=0.05)
        notes = [
            DetectedNote(0.0, 1.0, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.0, 1.02, 60, 0.8, 0.8, 0.0),  # 20ms, too short
        ]
        filtered = _filter_notes(notes, config)
        assert len(filtered) == 1


class TestMergeConsecutiveNotes:
    """Tests for merging consecutive notes."""

    def test_merges_same_pitch(self):
        """Same pitch notes close together are merged."""
        config = AudioAnalysisConfig(merge_gap_threshold=0.05)
        notes = [
            DetectedNote(0.0, 1.0, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.02, 2.0, 60, 0.7, 0.7, 0.0),
        ]
        merged = _merge_consecutive_notes(notes, config)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 2.0

    def test_does_not_merge_different_pitch(self):
        """Different pitches are not merged."""
        config = AudioAnalysisConfig(merge_gap_threshold=0.05)
        notes = [
            DetectedNote(0.0, 1.0, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.02, 2.0, 64, 0.8, 0.8, 0.0),
        ]
        merged = _merge_consecutive_notes(notes, config)
        assert len(merged) == 2


class TestFilterHarmonics:
    """Tests for harmonic filtering."""

    def test_filters_octave_harmonic(self):
        """Octave above with much lower amplitude is filtered.

        Octaves use a stricter threshold (0.35) because octave intervals
        are common in real guitar music (same note on different strings).
        """
        notes = [
            DetectedNote(1.0, 2.0, 60, 0.9, 0.9, 0.0),  # C4
            DetectedNote(1.0, 2.0, 72, 0.3, 0.3, 0.0),  # C5 - octave, very low amp
        ]
        filtered = _filter_harmonics(notes)
        assert len(filtered) == 1
        assert filtered[0].midi_note == 60

    def test_keeps_high_confidence_harmonic(self):
        """High confidence 'harmonic' is kept (likely intentional)."""
        notes = [
            DetectedNote(1.0, 2.0, 60, 0.7, 0.7, 0.0),
            DetectedNote(1.0, 2.0, 72, 0.8, 0.8, 0.0),  # Higher confidence
        ]
        filtered = _filter_harmonics(notes)
        assert len(filtered) == 2


class TestDetectNoteOnsets:
    """Tests for onset detection."""

    def test_single_note_onset(self):
        """Single note has one onset."""
        notes = [
            DetectedNote(1.0, 2.0, 60, 0.8, 0.8, 0.0)
        ]
        onsets = detect_note_onsets(notes)
        assert len(onsets) == 1
        assert onsets[0] == pytest.approx(1.0)

    def test_chord_has_one_onset(self):
        """Chord has single combined onset."""
        notes = [
            DetectedNote(1.0, 2.0, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.02, 2.0, 64, 0.9, 0.9, 0.0),
        ]
        onsets = detect_note_onsets(notes)
        assert len(onsets) == 1

    def test_sequential_notes_multiple_onsets(self):
        """Sequential notes have separate onsets."""
        notes = [
            DetectedNote(1.0, 1.5, 60, 0.8, 0.8, 0.0),
            DetectedNote(2.0, 2.5, 64, 0.8, 0.8, 0.0),
        ]
        onsets = detect_note_onsets(notes)
        assert len(onsets) == 2


class TestFusionConfig:
    """Tests for FusionConfig."""

    def test_default_values(self):
        """Default values are sensible."""
        config = FusionConfig()
        assert config.video_match_boost == 0.3
        assert config.open_string_confidence == 0.7
        assert config.chord_time_tolerance == 0.05


class TestSelectBestPosition:
    """Tests for position selection heuristics."""

    def test_prefers_lower_fret(self):
        """Lower fret is preferred when enabled."""
        config = FusionConfig(prefer_lower_frets=True)
        candidates = [
            Position(string=1, fret=5),
            Position(string=2, fret=0),
            Position(string=3, fret=10),
        ]
        best = _select_best_position(candidates, None, config)
        assert best.fret == 0

    def test_prefers_position_continuity(self):
        """Position near previous is preferred."""
        config = FusionConfig(prefer_same_position=True, prefer_lower_frets=False)
        candidates = [
            Position(string=1, fret=1),
            Position(string=2, fret=7),
        ]
        previous = Position(string=2, fret=5)
        best = _select_best_position(candidates, previous, config)
        # Should prefer fret 7 (closer to previous fret 5)
        assert best.fret == 7

    def test_single_candidate_returns_it(self):
        """Single candidate is returned."""
        config = FusionConfig()
        candidates = [Position(string=3, fret=7)]
        best = _select_best_position(candidates, None, config)
        assert best == candidates[0]


class TestOptimizeChordPositions:
    """Tests for chord position optimization."""

    def test_no_string_conflicts(self):
        """Selected positions don't share strings."""
        config = FusionConfig()
        # C major chord: C4, E4, G4
        chord_candidates = [
            (DetectedNote(0, 1, 60, 0.8, 0.8, 0.0), [Position(1, 0), Position(2, 5)]),
            (DetectedNote(0, 1, 64, 0.8, 0.8, 0.0), [Position(1, 5), Position(2, 0)]),
        ]
        positions = _optimize_chord_positions(chord_candidates, None, config)
        strings = [p.string for p in positions if p]
        assert len(strings) == len(set(strings))  # No duplicates


class TestDetectTechniques:
    """Tests for technique detection."""

    def test_detects_hammer_on(self):
        """Ascending notes on same string detected as hammer-on."""
        config = FusionConfig(hammer_on_max_gap=0.2)
        notes = [
            TabNote("1", 1.0, 3, 5, 0.8, "high", 60),
            TabNote("2", 1.1, 3, 7, 0.8, "high", 62),  # Same string, higher fret
        ]
        detected = _detect_techniques(notes, config)
        assert detected[1].technique == "hammer-on"

    def test_detects_pull_off(self):
        """Descending notes on same string detected as pull-off."""
        config = FusionConfig(hammer_on_max_gap=0.2)
        notes = [
            TabNote("1", 1.0, 3, 7, 0.8, "high", 62),
            TabNote("2", 1.1, 3, 5, 0.8, "high", 60),  # Same string, lower fret
        ]
        detected = _detect_techniques(notes, config)
        assert detected[1].technique == "pull-off"


class TestFretboardDetectionConfig:
    """Tests for FretboardDetectionConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = FretboardDetectionConfig()
        assert config.use_adaptive_threshold is True
        assert config.use_multi_scale is True
        assert len(config.scale_factors) > 0


class TestCalculateDetectionConfidence:
    """Tests for fretboard detection confidence calculation."""

    def test_more_frets_higher_confidence(self):
        """More detected frets increases confidence."""
        few_frets = [0.0, 0.5, 1.0]
        many_frets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        conf_few = _calculate_detection_confidence(few_frets, 500, 100, 3)
        conf_many = _calculate_detection_confidence(many_frets, 500, 100, 11)

        assert conf_many > conf_few

    def test_good_aspect_ratio_bonus(self):
        """Good aspect ratio increases confidence."""
        # Good aspect ratio (5:1 to 12:1)
        conf_good = _calculate_detection_confidence([0, 0.5, 1], 500, 50, 3)
        # Bad aspect ratio (1:1)
        conf_bad = _calculate_detection_confidence([0, 0.5, 1], 100, 100, 3)

        assert conf_good > conf_bad


class TestVideoAnalysisConfig:
    """Tests for VideoAnalysisConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = VideoAnalysisConfig()
        assert config.min_detection_confidence == 0.5
        assert config.frames_per_onset == 5
        assert config.max_num_hands == 2


class TestFingerPositionEnhancements:
    """Tests for enhanced FingerPosition."""

    def test_finger_position_with_state(self):
        """FingerPosition includes state attributes."""
        finger = FingerPosition(
            finger_id=1,
            x=0.5,
            y=0.5,
            z=-0.03,
            is_extended=True,
            angle=45.0,
            confidence=0.9
        )
        assert finger.is_extended is True
        assert finger.angle == 45.0


class TestHandObservationEnhancements:
    """Tests for enhanced HandObservation."""

    def test_get_pressing_fingers(self):
        """Can get only pressing fingers."""
        fingers = [
            FingerPosition(0, 0.3, 0.4, -0.01),
            FingerPosition(1, 0.4, 0.4, -0.03),  # pressing
            FingerPosition(2, 0.5, 0.4, 0.01),
        ]
        obs = HandObservation(
            timestamp=1.0,
            fingers=fingers,
            is_left_hand=True,
            pressing_fingers=[1]
        )
        pressing = obs.get_pressing_finger_positions()
        assert len(pressing) == 1
        assert pressing[0].finger_id == 1


class TestMapFingerToPositionEnhancements:
    """Tests for enhanced finger mapping."""

    def test_includes_finger_id(self):
        """Mapped position includes finger ID."""
        geometry = FretboardGeometry(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 300),
            bottom_right=(500, 300),
            fret_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
            string_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        pos = map_finger_to_position(
            finger_x=300,
            finger_y=200,
            geometry=geometry,
            finger_id=2
        )
        assert pos is not None
        assert pos.finger_id == 2

    def test_z_depth_affects_pressing_detection(self):
        """Z-depth affects is_pressing flag."""
        geometry = FretboardGeometry(
            top_left=(100, 100),
            top_right=(500, 100),
            bottom_left=(100, 300),
            bottom_right=(500, 300),
            fret_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
            string_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        # Pressing finger (negative z)
        pressing = map_finger_to_position(300, 200, geometry, finger_z=-0.05)
        assert pressing is not None
        assert pressing.is_pressing is True

        # Hovering finger (positive z)
        hovering = map_finger_to_position(300, 200, geometry, finger_z=0.05)
        assert hovering is not None
        assert hovering.is_pressing is False


class TestFuseAudioOnlyEnhancements:
    """Tests for enhanced audio-only fusion."""

    def test_creates_chord_ids(self):
        """Simultaneous notes get chord IDs."""
        notes = [
            DetectedNote(1.0, 1.5, 60, 0.8, 0.8, 0.0),
            DetectedNote(1.02, 1.5, 64, 0.8, 0.8, 0.0),
        ]
        config = FusionConfig(chord_time_tolerance=0.05)
        result = fuse_audio_only(notes, capo_fret=0, config=config)

        # Both notes should have same chord_id
        assert result[0].chord_id == result[1].chord_id
        assert result[0].is_part_of_chord is True


class TestTabNoteEnhancements:
    """Tests for enhanced TabNote."""

    def test_tab_note_has_enhanced_attrs(self):
        """TabNote has all enhanced attributes."""
        note = TabNote(
            id="test",
            timestamp=1.0,
            string=3,
            fret=5,
            confidence=0.8,
            confidence_level="high",
            midi_note=60,
            end_time=1.5,
            technique="hammer-on",
            is_part_of_chord=True,
            chord_id="chord1",
            video_matched=True,
            audio_confidence=0.75,
            video_confidence=0.85
        )
        assert note.technique == "hammer-on"
        assert note.video_matched is True


class TestFretboardGeometryEnhancements:
    """Tests for enhanced FretboardGeometry."""

    def test_width_and_height_properties(self):
        """Width and height calculated correctly."""
        geometry = FretboardGeometry(
            top_left=(100, 100),
            top_right=(600, 100),
            bottom_left=(100, 200),
            bottom_right=(600, 200),
            fret_positions=[0, 0.5, 1.0],
            string_positions=[0, 0.5, 1.0]
        )
        assert geometry.width == 500
        assert geometry.height == 100

    def test_is_valid_checks_aspect_ratio(self):
        """is_valid checks aspect ratio."""
        # Valid aspect ratio
        valid = FretboardGeometry(
            top_left=(0, 0),
            top_right=(500, 0),
            bottom_left=(0, 50),
            bottom_right=(500, 50),
            fret_positions=[0, 0.3, 0.6, 1.0],
            string_positions=[0, 0.5, 1.0]
        )
        assert valid.is_valid() is True

        # Invalid aspect ratio (square)
        invalid = FretboardGeometry(
            top_left=(0, 0),
            top_right=(100, 0),
            bottom_left=(0, 100),
            bottom_right=(100, 100),
            fret_positions=[0, 0.5, 1.0],
            string_positions=[0, 0.5, 1.0]
        )
        assert invalid.is_valid() is False
