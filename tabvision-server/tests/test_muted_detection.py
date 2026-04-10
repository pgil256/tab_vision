"""Tests for muted string detection across audio, video, and fusion."""
import pytest
from unittest.mock import patch, MagicMock

from app.audio_pipeline import MutedNote, DetectedNote, detect_muted_notes
from app.video_pipeline import (
    HandObservation, FingerPosition, VideoAnalysisConfig,
)
from app.fusion_engine import (
    TabNote, FusionConfig, _create_muted_tab_notes, fuse_audio_only,
)


class TestMutedNote:
    """Tests for MutedNote dataclass."""

    def test_creation(self):
        note = MutedNote(timestamp=1.0, onset_strength=0.8, confidence=0.7)
        assert note.timestamp == 1.0
        assert note.onset_strength == 0.8
        assert note.confidence == 0.7


class TestDetectMutedNotes:
    """Tests for audio-based muted note detection."""

    def test_detect_missing_file_returns_empty(self):
        """Missing file should return empty list."""
        result = detect_muted_notes("nonexistent.wav", [])
        assert result == []

    def test_detect_with_unpitched_onsets(self):
        """Should detect onsets that don't match any pitched note."""
        import numpy as np
        import types

        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (np.zeros(22050), 22050)
        onset_frames = np.array([0, 10, 20])
        mock_librosa.onset.onset_detect.return_value = onset_frames
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])
        onset_env = np.ones(21) * 0.8
        mock_librosa.onset.onset_strength.return_value = onset_env

        pitched = [DetectedNote(start_time=0.5, end_time=1.0, midi_note=69, confidence=0.9)]

        with patch('os.path.exists', return_value=True), \
             patch.dict('sys.modules', {'librosa': mock_librosa}):
            # Need to reimport to pick up mock
            import importlib
            import app.audio_pipeline as ap
            # Directly call with the mock in place
            result = detect_muted_notes("audio.wav", pitched, min_onset_strength=0.3)

        # Onsets at 0.0 and 1.0 don't match pitched note at 0.5
        assert len(result) >= 1

    def test_all_onsets_have_pitch(self):
        """When all onsets match pitched notes, no muted notes returned."""
        import numpy as np

        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (np.zeros(22050), 22050)
        mock_librosa.onset.onset_detect.return_value = np.array([10])
        mock_librosa.frames_to_time.return_value = np.array([0.5])
        mock_librosa.onset.onset_strength.return_value = np.ones(11) * 0.8

        pitched = [DetectedNote(start_time=0.48, end_time=1.0, midi_note=69, confidence=0.9)]

        with patch('os.path.exists', return_value=True), \
             patch.dict('sys.modules', {'librosa': mock_librosa}):
            result = detect_muted_notes("audio.wav", pitched, min_onset_strength=0.3)

        assert len(result) == 0


class TestVideoMutingDetection:
    """Tests for video-based muting finger detection."""

    def test_muting_z_threshold_config(self):
        """VideoAnalysisConfig should have muting_z_threshold."""
        config = VideoAnalysisConfig()
        assert hasattr(config, 'muting_z_threshold')
        assert config.muting_z_threshold == -0.005

    def test_hand_observation_has_muting_fingers(self):
        """HandObservation should track muting fingers."""
        obs = HandObservation(
            timestamp=1.0,
            fingers=[],
            is_left_hand=True,
            muting_fingers=[1, 2],
        )
        assert obs.muting_fingers == [1, 2]

    def test_get_muting_finger_positions(self):
        """get_muting_finger_positions returns only muting fingers."""
        fingers = [
            FingerPosition(finger_id=0, x=0.1, y=0.5, z=0.0),
            FingerPosition(finger_id=1, x=0.2, y=0.5, z=-0.01),
            FingerPosition(finger_id=2, x=0.3, y=0.5, z=-0.03),
        ]
        obs = HandObservation(
            timestamp=1.0,
            fingers=fingers,
            is_left_hand=True,
            muting_fingers=[1],
            pressing_fingers=[2],
        )
        muting = obs.get_muting_finger_positions()
        assert len(muting) == 1
        assert muting[0].finger_id == 1


class TestCreateMutedTabNotes:
    """Tests for fusion of muted notes."""

    def test_creates_muted_tab_note(self):
        """Should create TabNote with fret='X' from high-confidence muted note."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.8, confidence=0.7)]
        config = FusionConfig(muted_note_min_confidence=0.4)
        result = _create_muted_tab_notes(muted, config=config)
        assert len(result) == 1
        assert result[0].fret == "X"
        assert result[0].technique == "muted"
        assert result[0].midi_note == 0

    def test_filters_low_confidence(self):
        """Should filter muted notes below confidence threshold."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.3, confidence=0.3)]
        config = FusionConfig(muted_note_min_confidence=0.4)
        result = _create_muted_tab_notes(muted, config=config)
        assert len(result) == 0

    def test_audio_only_requires_higher_confidence(self):
        """Without video, muted notes need confidence >= 0.6."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.5, confidence=0.5)]
        config = FusionConfig(muted_note_min_confidence=0.4)
        result = _create_muted_tab_notes(muted, video_observations=None, config=config)
        # 0.5 >= 0.4 min but < 0.6 audio-only threshold
        assert len(result) == 0

    def test_video_confirmation_lowers_threshold(self):
        """With video muting confirmation, lower confidence is accepted."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.5, confidence=0.5)]
        video_obs = {
            1.0: HandObservation(
                timestamp=1.0,
                fingers=[],
                is_left_hand=True,
                muting_fingers=[1, 2],
            )
        }
        config = FusionConfig(muted_note_min_confidence=0.4)
        result = _create_muted_tab_notes(muted, video_observations=video_obs, config=config)
        assert len(result) == 1
        assert result[0].video_matched is True

    def test_video_confirmation_boosts_confidence(self):
        """Video-confirmed muted notes should get confidence boost."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.7, confidence=0.7)]
        video_obs = {
            1.0: HandObservation(
                timestamp=1.0,
                fingers=[],
                is_left_hand=True,
                muting_fingers=[1],
            )
        }
        result = _create_muted_tab_notes(muted, video_observations=video_obs)
        assert abs(result[0].confidence - 0.9) < 0.01  # 0.7 + 0.2 boost

    def test_no_video_muting_at_timestamp(self):
        """Video observations without muting fingers don't confirm."""
        muted = [MutedNote(timestamp=1.0, onset_strength=0.5, confidence=0.5)]
        video_obs = {
            1.0: HandObservation(
                timestamp=1.0,
                fingers=[],
                is_left_hand=True,
                muting_fingers=[],  # No muting
                pressing_fingers=[1],
            )
        }
        config = FusionConfig(muted_note_min_confidence=0.4)
        result = _create_muted_tab_notes(muted, video_observations=video_obs, config=config)
        # 0.5 < 0.6 audio-only threshold, no video confirmation
        assert len(result) == 0


class TestFuseAudioOnlyWithMutedNotes:
    """Tests for muted note integration in fuse_audio_only."""

    def test_muted_notes_included_in_output(self):
        """fuse_audio_only should include muted notes in output."""
        pitched = [
            DetectedNote(start_time=0.5, end_time=1.0, midi_note=69, confidence=0.9)
        ]
        muted = [MutedNote(timestamp=1.5, onset_strength=0.8, confidence=0.8)]
        result = fuse_audio_only(pitched, capo_fret=0, muted_notes=muted)

        muted_notes = [n for n in result if n.fret == "X"]
        assert len(muted_notes) == 1
        assert muted_notes[0].timestamp == 1.5
        assert muted_notes[0].technique == "muted"

    def test_no_muted_notes_when_none(self):
        """fuse_audio_only with no muted notes should work normally."""
        pitched = [
            DetectedNote(start_time=0.5, end_time=1.0, midi_note=69, confidence=0.9)
        ]
        result = fuse_audio_only(pitched, capo_fret=0, muted_notes=None)

        muted_notes = [n for n in result if n.fret == "X"]
        assert len(muted_notes) == 0

    def test_muted_notes_sorted_by_timestamp(self):
        """Muted notes should be interleaved with pitched notes by timestamp."""
        pitched = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.9)
        ]
        muted = [MutedNote(timestamp=0.5, onset_strength=0.8, confidence=0.8)]
        result = fuse_audio_only(pitched, capo_fret=0, muted_notes=muted)

        # Muted note at 0.5 should come before pitched note at 1.0
        timestamps = [n.timestamp for n in result]
        assert timestamps == sorted(timestamps)
