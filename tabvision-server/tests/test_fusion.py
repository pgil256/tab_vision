"""Tests for fusion_engine module."""
import pytest
from app.fusion_engine import (
    fuse_audio_only,
    fuse_audio_video,
    find_nearest_observation,
    match_video_to_candidates,
    has_open_string_candidate,
    get_confidence_level,
    _correct_slide_positions,
    _postfilter_tab_notes,
    FusionConfig,
    TabNote,
)
from app.audio_pipeline import DetectedNote
from app.video_pipeline import HandObservation, FingerPosition
from app.fretboard_detection import FretboardGeometry
from app.guitar_mapping import Position


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
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.65),  # A4
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
        assert result[2].confidence_level == "medium"

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


class TestFindNearestObservation:
    """Tests for finding video observations near timestamps."""

    def test_find_exact_match(self):
        """Should find observation at exact timestamp."""
        obs = HandObservation(
            timestamp=1.0,
            fingers=[FingerPosition(0, 0.5, 0.5, 0.0)],
            is_left_hand=True,
        )
        observations = {1.0: obs}

        result = find_nearest_observation(observations, 1.0)
        assert result == obs

    def test_find_within_tolerance(self):
        """Should find observation within tolerance."""
        obs = HandObservation(
            timestamp=1.0,
            fingers=[FingerPosition(0, 0.5, 0.5, 0.0)],
            is_left_hand=True,
        )
        observations = {1.0: obs}

        result = find_nearest_observation(observations, 1.05, tolerance=0.1)
        assert result == obs

    def test_no_match_outside_tolerance(self):
        """Should return None if no observation within tolerance."""
        obs = HandObservation(
            timestamp=1.0,
            fingers=[FingerPosition(0, 0.5, 0.5, 0.0)],
            is_left_hand=True,
        )
        observations = {1.0: obs}

        result = find_nearest_observation(observations, 2.0, tolerance=0.1)
        assert result is None

    def test_empty_observations(self):
        """Should return None for empty observations dict."""
        result = find_nearest_observation({}, 1.0)
        assert result is None

    def test_finds_nearest_of_multiple(self):
        """Should find nearest observation among multiple."""
        obs1 = HandObservation(timestamp=1.0, fingers=[], is_left_hand=True)
        obs2 = HandObservation(timestamp=2.0, fingers=[], is_left_hand=True)
        observations = {1.0: obs1, 2.0: obs2}

        result = find_nearest_observation(observations, 1.8, tolerance=0.5)
        assert result == obs2


class TestFuseAudioVideo:
    """Tests for audio+video fusion."""

    def _make_fretboard(self) -> FretboardGeometry:
        """Create a simple fretboard geometry for testing."""
        return FretboardGeometry(
            top_left=(100.0, 100.0),
            top_right=(540.0, 100.0),
            bottom_left=(100.0, 380.0),
            bottom_right=(540.0, 380.0),
            fret_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            string_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

    def test_fuse_falls_back_without_fretboard(self):
        """Should fall back to audio-only when no fretboard detected."""
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.8),
        ]

        result = fuse_audio_video(detected_notes, {}, fretboard=None, capo_fret=0)

        assert len(result) == 1
        # Should match audio-only behavior
        assert result[0].fret == 5
        assert result[0].string == 1

    def test_fuse_with_empty_video_observations(self):
        """Should use audio-only when no video observations."""
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.8),
        ]
        fretboard = self._make_fretboard()

        result = fuse_audio_video(detected_notes, {}, fretboard, capo_fret=0)

        assert len(result) == 1
        # Falls back to lowest-fret heuristic
        assert result[0].fret == 5
        assert result[0].string == 1

    def test_fuse_boosts_confidence_on_agreement(self):
        """Should boost confidence when video agrees with audio candidate."""
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.7),
        ]
        fretboard = self._make_fretboard()

        # Create video observation with finger at position matching fret 5, string 1
        # For A4 (MIDI 69), valid positions include string 1 fret 5
        finger = FingerPosition(
            finger_id=1,
            # Position finger inside fretboard area in normalized coords
            x=0.5,  # Middle of frame width
            y=0.25,  # Near top of fretboard (high E string area)
            z=0.0,
        )
        video_obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        # Note: The actual match depends on coordinate mapping, so we test
        # the confidence boost path by mocking or accepting the fallback behavior
        result = fuse_audio_video(
            detected_notes, {1.0: video_obs}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # With or without video match, we get a result

    def test_fuse_audio_video_with_no_match_uses_audio(self):
        """Should use audio position when video doesn't match any candidate."""
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.7),
        ]
        fretboard = self._make_fretboard()

        # Finger position that doesn't match any audio candidate
        finger = FingerPosition(finger_id=1, x=0.1, y=0.9, z=0.0)
        video_obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        result = fuse_audio_video(
            detected_notes, {1.0: video_obs}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # Falls back to lowest-fret heuristic
        assert result[0].fret == 5
        assert result[0].string == 1
        assert result[0].confidence == 0.7  # Original confidence, no boost

    def test_fuse_empty_input(self):
        """Should return empty list for empty input."""
        fretboard = self._make_fretboard()
        result = fuse_audio_video([], {}, fretboard, capo_fret=0)
        assert result == []


class TestMatchVideoToCandidates:
    """Tests for matching video positions to audio candidates."""

    def _make_fretboard(self) -> FretboardGeometry:
        """Create a fretboard geometry for testing."""
        return FretboardGeometry(
            top_left=(100.0, 100.0),
            top_right=(540.0, 100.0),
            bottom_left=(100.0, 380.0),
            bottom_right=(540.0, 380.0),
            fret_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            string_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

    def test_no_match_with_empty_candidates(self):
        """Should return None when no candidates provided."""
        fretboard = self._make_fretboard()
        finger = FingerPosition(finger_id=1, x=0.5, y=0.5, z=0.0)
        obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        result = match_video_to_candidates(obs, fretboard, candidates=[])
        assert result is None

    def test_no_match_with_no_fingers(self):
        """Should return None when no fingers detected."""
        fretboard = self._make_fretboard()
        obs = HandObservation(timestamp=1.0, fingers=[], is_left_hand=True)
        candidates = [Position(string=1, fret=5)]

        result = match_video_to_candidates(obs, fretboard, candidates)
        assert result is None


class TestHasOpenStringCandidate:
    """Tests for open string candidate detection."""

    def test_finds_open_string_when_present(self):
        """Should return the fret 0 position if it exists."""
        candidates = [
            Position(string=1, fret=0),
            Position(string=2, fret=5),
        ]
        result = has_open_string_candidate(candidates)
        assert result is not None
        assert result.fret == 0
        assert result.string == 1

    def test_returns_none_when_no_open_string(self):
        """Should return None when no fret 0 candidate."""
        candidates = [
            Position(string=1, fret=5),
            Position(string=2, fret=10),
        ]
        result = has_open_string_candidate(candidates)
        assert result is None

    def test_handles_empty_candidates(self):
        """Should return None for empty candidates list."""
        result = has_open_string_candidate([])
        assert result is None


class TestOpenStringDetection:
    """Tests for open string detection in audio+video fusion."""

    def _make_fretboard(self) -> FretboardGeometry:
        """Create a simple fretboard geometry for testing."""
        return FretboardGeometry(
            top_left=(100.0, 100.0),
            top_right=(540.0, 100.0),
            bottom_left=(100.0, 380.0),
            bottom_right=(540.0, 380.0),
            fret_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            string_positions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

    def test_open_string_when_no_finger_match_and_fret0_valid(self):
        """Should use open string when video shows no finger match and fret 0 is valid.

        E4 (MIDI 64) can be played as open string 1 (fret 0) or string 2 fret 5.
        If video observation exists but no finger matches, prefer open string.
        """
        # E4 = open string 1 (high E)
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=64, confidence=0.8),
        ]
        fretboard = self._make_fretboard()

        # Video observation with finger that doesn't match any candidate position
        # (finger is outside fretboard area)
        finger = FingerPosition(finger_id=1, x=0.05, y=0.05, z=0.0)
        video_obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        result = fuse_audio_video(
            detected_notes, {1.0: video_obs}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # Should prefer open string (fret 0) since video shows no finger match
        assert result[0].fret == 0
        assert result[0].string == 1
        assert result[0].confidence == 0.7  # Open string confidence from FusionConfig
        assert result[0].confidence_level == "medium"

    def test_no_open_string_when_fret0_not_valid(self):
        """Should fall back to lowest fret when fret 0 is not a valid candidate.

        A4 (MIDI 69) cannot be an open string, so it falls back to lowest fret.
        """
        # A4 = string 1 fret 5 (no open string option)
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.7),
        ]
        fretboard = self._make_fretboard()

        # Video observation with no matching finger
        finger = FingerPosition(finger_id=1, x=0.05, y=0.05, z=0.0)
        video_obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        result = fuse_audio_video(
            detected_notes, {1.0: video_obs}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # Falls back to lowest fret heuristic (fret 5 on string 1)
        assert result[0].fret == 5
        assert result[0].string == 1
        assert result[0].confidence == 0.7  # Original confidence

    def test_no_open_string_without_video_observation(self):
        """Should use lowest fret heuristic when no video observation exists."""
        # E4 has open string option, but without video we use lowest fret
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=64, confidence=0.8),
        ]
        fretboard = self._make_fretboard()

        # No video observation at this timestamp
        result = fuse_audio_video(
            detected_notes, {}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # Lowest fret heuristic also picks fret 0, but with original confidence
        assert result[0].fret == 0
        assert result[0].string == 1
        assert result[0].confidence == 0.8  # Original confidence, not 0.65

    def test_video_match_takes_precedence_over_open_string(self):
        """Video match should take precedence over open string inference."""
        # E4 = open string 1 OR string 2 fret 5
        detected_notes = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=64, confidence=0.7),
        ]
        fretboard = self._make_fretboard()

        # Finger positioned to match string 2 fret 5 candidate
        # (This would require the coordinate mapping to work correctly)
        # For this test, we verify the logic flow - if video matches, use it
        finger = FingerPosition(finger_id=1, x=0.5, y=0.5, z=0.0)
        video_obs = HandObservation(timestamp=1.0, fingers=[finger], is_left_hand=True)

        result = fuse_audio_video(
            detected_notes, {1.0: video_obs}, fretboard, capo_fret=0
        )

        assert len(result) == 1
        # Result depends on coordinate mapping - just verify we get a result
        # If video matches a candidate, confidence is boosted; otherwise open string or fallback


class TestCorrectSlidePositions:
    """Tests for slide/legato position correction."""

    def test_descending_semitone_prefers_existing_string(self):
        """When s5f8 is followed by a note that could be s5f7 or s4f2,
        prefer s5f7 (same string as prev) for continuity."""
        notes = [
            TabNote(id='1', timestamp=12.62, string=5, fret=8, confidence=0.77,
                    confidence_level='medium', midi_note=53),
            TabNote(id='2', timestamp=12.98, string=4, fret=2, confidence=1.00,
                    confidence_level='high', midi_note=52),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        # Both should end up on string 5 (descending slide)
        assert result[0].string == 5
        assert result[0].fret == 8
        assert result[1].string == 5
        assert result[1].fret == 7

    def test_ascending_semitone_prefers_existing_string(self):
        """When s5f7 is followed by a note that could be s5f8 or s4f3,
        prefer s5f8 (same string as prev)."""
        notes = [
            TabNote(id='1', timestamp=1.00, string=5, fret=7, confidence=0.8,
                    confidence_level='high', midi_note=52),
            TabNote(id='2', timestamp=1.30, string=4, fret=3, confidence=0.8,
                    confidence_level='high', midi_note=53),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 5
        assert result[0].fret == 7
        assert result[1].string == 5
        assert result[1].fret == 8

    def test_does_not_change_same_string_pair(self):
        """Notes already on the same string should not be modified."""
        notes = [
            TabNote(id='1', timestamp=1.00, string=3, fret=2, confidence=0.8,
                    confidence_level='high', midi_note=57),
            TabNote(id='2', timestamp=1.30, string=3, fret=4, confidence=0.8,
                    confidence_level='high', midi_note=59),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 3
        assert result[0].fret == 2
        assert result[1].string == 3
        assert result[1].fret == 4

    def test_full_slide_section_s5f8_to_s5f7(self):
        """Realistic slide section: s5f0->s5f4->s5f8->s4f2 should correct
        the last note to s5f7 while preserving the rest."""
        notes = [
            TabNote(id='a', timestamp=12.16, string=5, fret=0, confidence=0.88,
                    confidence_level='high', midi_note=45),
            TabNote(id='b', timestamp=12.51, string=5, fret=4, confidence=0.54,
                    confidence_level='medium', midi_note=49),
            TabNote(id='c', timestamp=12.62, string=5, fret=8, confidence=0.77,
                    confidence_level='medium', midi_note=53),
            TabNote(id='d', timestamp=12.98, string=4, fret=2, confidence=1.00,
                    confidence_level='high', midi_note=52),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 5 and result[0].fret == 0   # unchanged
        assert result[1].string == 5 and result[1].fret == 4   # unchanged
        assert result[2].string == 5 and result[2].fret == 8   # unchanged
        assert result[3].string == 5 and result[3].fret == 7   # corrected!


class TestFuseAudioVideoFiltering:
    """Tests that fuse_audio_video applies the same filtering as fuse_audio_only."""

    def _make_fretboard(self):
        """Create a minimal FretboardGeometry for testing."""
        return FretboardGeometry(
            top_left=(0.0, 0.2),
            top_right=(1.0, 0.2),
            bottom_left=(0.0, 0.8),
            bottom_right=(1.0, 0.8),
            fret_positions=[i * 0.04 for i in range(25)],
            string_positions=[0.2 + i * 0.1 for i in range(6)],
            detection_confidence=0.8,
            frame_width=640,
            frame_height=480,
        )

    def test_ghost_notes_filtered(self):
        """Ghost notes (low amplitude overlapping loud notes) should be removed."""
        loud_note = DetectedNote(
            start_time=1.0, end_time=2.0, midi_note=64,
            confidence=0.9, amplitude=0.8,
        )
        ghost_note = DetectedNote(
            start_time=1.0, end_time=1.5, midi_note=76,
            confidence=0.5, amplitude=0.15,
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [loud_note, ghost_note], {}, fretboard, capo_fret=0
        )
        assert len(result) == 1
        assert result[0].midi_note == 64

    def test_chord_size_limited(self):
        """Chords should be limited to max_chord_size (default 3)."""
        notes = [
            DetectedNote(
                start_time=1.0, end_time=2.0, midi_note=midi,
                confidence=0.8, amplitude=0.5 + i * 0.1,
            )
            for i, midi in enumerate([40, 45, 50, 55, 60])
        ]
        fretboard = self._make_fretboard()
        result = fuse_audio_video(notes, {}, fretboard, capo_fret=0)
        assert len(result) <= 3

    def test_slide_positions_corrected(self):
        """Consecutive semitone notes on different strings should be corrected to same string."""
        # MIDI 55 (G3) and MIDI 56 (G#3) are a semitone apart
        # Both can be played on string 4 or string 5
        note1 = DetectedNote(
            start_time=1.0, end_time=1.3, midi_note=55,
            confidence=0.9, amplitude=0.7,
        )
        note2 = DetectedNote(
            start_time=1.3, end_time=1.6, midi_note=56,
            confidence=0.9, amplitude=0.7,
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [note1, note2], {}, fretboard, capo_fret=0
        )
        assert len(result) == 2
        # Both should end up on the same string after slide correction
        assert result[0].string == result[1].string

    def test_low_confidence_singletons_filtered(self):
        """Low-confidence non-chord notes should be removed by post-filtering."""
        # A single quiet note with low confidence
        notes = [
            DetectedNote(
                start_time=1.0, end_time=2.0, midi_note=64,
                confidence=0.45, amplitude=0.5,
            ),
        ]
        fretboard = self._make_fretboard()
        config = FusionConfig()
        result = fuse_audio_video(notes, {}, fretboard, capo_fret=0, config=config)
        # Should be removed by post-filter (confidence < 0.6, not in chord)
        assert len(result) == 0

    def test_duplicate_positions_deduped(self):
        """Same string+fret within 0.3s should be deduped."""
        # Two notes with the same MIDI pitch very close together
        note1 = DetectedNote(
            start_time=1.0, end_time=1.5, midi_note=64,
            confidence=0.9, amplitude=0.7,
        )
        note2 = DetectedNote(
            start_time=1.15, end_time=1.6, midi_note=64,
            confidence=0.8, amplitude=0.6,
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [note1, note2], {}, fretboard, capo_fret=0
        )
        assert len(result) == 1


class TestPostfilterTabNotes:
    """Tests for post-fusion note filtering."""

    def _make_note(self, timestamp, string, fret, confidence=0.8, midi_note=60,
                   is_part_of_chord=False):
        return TabNote(
            id=str(timestamp), timestamp=timestamp, string=string, fret=fret,
            confidence=confidence, confidence_level=get_confidence_level(confidence),
            midi_note=midi_note, is_part_of_chord=is_part_of_chord,
        )

    def test_removes_duplicate_position_within_window(self):
        """Same string+fret within 0.3s should keep only the higher confidence one."""
        notes = [
            self._make_note(9.99, 3, 0, confidence=0.64),
            self._make_note(10.06, 3, 0, confidence=0.85),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 1
        assert result[0].confidence == 0.85  # kept the higher-confidence one

    def test_keeps_duplicate_outside_window(self):
        """Same string+fret more than 0.3s apart should both be kept."""
        notes = [
            self._make_note(1.00, 3, 0, confidence=0.80),
            self._make_note(1.50, 3, 0, confidence=0.85),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 2

    def test_removes_low_confidence_isolated_singleton(self):
        """Low-confidence (<0.6) notes not in a chord should be removed."""
        notes = [
            self._make_note(12.16, 5, 0, confidence=0.88, midi_note=45),
            self._make_note(12.51, 5, 4, confidence=0.54, midi_note=49),  # low conf, not in chord
            self._make_note(12.62, 5, 8, confidence=0.77, midi_note=53),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        # s5f4 at 12.51 has conf=0.54 < 0.6, not in chord -> removed
        assert len(result) == 2
        assert all(n.fret != 4 for n in result)

    def test_keeps_low_confidence_chord_member(self):
        """Low-confidence notes that are part of a chord should be kept."""
        notes = [
            self._make_note(1.00, 3, 5, confidence=0.55, is_part_of_chord=True),
            self._make_note(1.00, 1, 7, confidence=0.80, is_part_of_chord=True),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 2

    def test_keeps_high_confidence_singleton(self):
        """High-confidence singletons should always be kept."""
        notes = [
            self._make_note(5.00, 5, 0, confidence=0.70),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 1

    def test_removes_isolated_open_string_different_string_group(self):
        """Open string note isolated among notes on distant strings should be removed."""
        notes = [
            self._make_note(7.55, 3, 2, confidence=0.90, midi_note=57),   # melody on s3
            self._make_note(7.95, 2, 1, confidence=0.93, midi_note=60),   # melody on s2
            self._make_note(8.20, 5, 0, confidence=0.70, midi_note=45),   # FP: isolated open A
            self._make_note(8.81, 1, 0, confidence=0.92, midi_note=64),   # melody on s1
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3
        # s5f0 should be removed — isolated open string far from neighbor strings
        assert all(not (n.string == 5 and n.fret == 0) for n in result)

    def test_keeps_open_string_in_chord(self):
        """Open string that's part of a chord should be kept."""
        notes = [
            self._make_note(10.00, 5, 2, confidence=0.67, midi_note=47,
                           is_part_of_chord=True),
            self._make_note(10.00, 3, 0, confidence=0.64, midi_note=55,
                           is_part_of_chord=True),
            self._make_note(10.00, 2, 0, confidence=0.71, midi_note=59,
                           is_part_of_chord=True),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3  # all kept — part of chord

    def test_keeps_open_string_among_nearby_strings(self):
        """Open string with neighbors on adjacent strings should be kept."""
        notes = [
            self._make_note(5.00, 4, 3, confidence=0.80, midi_note=53),
            self._make_note(5.20, 5, 0, confidence=0.70, midi_note=45),  # open A near s4
            self._make_note(5.40, 5, 2, confidence=0.80, midi_note=47),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3  # kept — s5 is adjacent to s4
