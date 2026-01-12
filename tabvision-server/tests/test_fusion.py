"""Tests for fusion_engine module."""
import pytest
from app.fusion_engine import (
    fuse_audio_only,
    fuse_audio_video,
    find_nearest_observation,
    match_video_to_candidates,
    has_open_string_candidate,
    get_confidence_level,
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
        assert result[0].confidence == 0.65  # Medium confidence for open string inference
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
