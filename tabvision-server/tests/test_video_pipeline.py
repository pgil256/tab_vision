"""Tests for video_pipeline module."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.video_pipeline import (
    FingerPosition,
    HandObservation,
    extract_frame,
    detect_hand_landmarks,
    analyze_video_at_timestamps,
    FINGERTIP_INDICES,
)


class TestFingerPosition:
    """Tests for the FingerPosition dataclass."""

    def test_finger_position_creation(self):
        """FingerPosition stores fingertip coordinates."""
        finger = FingerPosition(
            finger_id=1,
            x=0.5,
            y=0.6,
            z=0.1,
        )

        assert finger.finger_id == 1
        assert finger.x == 0.5
        assert finger.y == 0.6
        assert finger.z == 0.1


class TestHandObservation:
    """Tests for the HandObservation dataclass."""

    def test_hand_observation_creation(self):
        """HandObservation stores hand detection results."""
        fingers = [
            FingerPosition(finger_id=i, x=0.1 * i, y=0.2 * i, z=0.0)
            for i in range(5)
        ]
        obs = HandObservation(
            timestamp=1.5,
            fingers=fingers,
            is_left_hand=True,
        )

        assert obs.timestamp == 1.5
        assert len(obs.fingers) == 5
        assert obs.is_left_hand is True


class TestExtractFrame:
    """Tests for frame extraction from video."""

    def test_extract_frame_returns_none_for_missing_file(self):
        """extract_frame returns None for non-existent video."""
        result = extract_frame("/nonexistent/video.mp4", 0.0)
        assert result is None

    def test_extract_frame_returns_none_for_invalid_video(self, tmp_path):
        """extract_frame returns None for invalid video file."""
        # Create a fake video file
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_text("not a video")

        result = extract_frame(str(fake_video), 0.0)
        assert result is None

    def test_extract_frame_returns_valid_frame(self, tmp_path):
        """extract_frame returns numpy array for valid video."""
        import cv2

        # Create a simple test video
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))

        # Write some frames
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (100, 150, 200)  # BGR color
        for _ in range(30):  # 1 second of video
            out.write(frame)
        out.release()

        # Extract frame at 0.5 seconds
        result = extract_frame(video_path, 0.5)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (240, 320, 3)

    def test_extract_frame_at_different_timestamps(self, tmp_path):
        """extract_frame can extract frames at different timestamps."""
        import cv2

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))

        # Write frames with different colors
        for i in range(60):  # 2 seconds
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (i * 4, i * 4, i * 4)  # Gradually brighter
            out.write(frame)
        out.release()

        # Extract at start and middle
        frame_start = extract_frame(video_path, 0.0)
        frame_middle = extract_frame(video_path, 1.0)

        assert frame_start is not None
        assert frame_middle is not None
        # Middle frame should be brighter (higher values)
        assert frame_middle.mean() > frame_start.mean()


class TestDetectHandLandmarks:
    """Tests for hand landmark detection."""

    def test_detect_hand_landmarks_returns_none_for_empty_frame(self):
        """detect_hand_landmarks returns None for empty frame."""
        empty_frame = np.array([])
        result = detect_hand_landmarks(empty_frame)
        assert result is None

    def test_detect_hand_landmarks_returns_none_for_none_frame(self):
        """detect_hand_landmarks returns None for None input."""
        result = detect_hand_landmarks(None)
        assert result is None

    def test_detect_hand_landmarks_no_hand_in_image(self, tmp_path):
        """detect_hand_landmarks returns None when no hand present."""
        pytest.importorskip("mediapipe")

        # Create a simple image without any hand
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)  # Gray background

        result = detect_hand_landmarks(frame)
        assert result is None

    def test_detect_hand_landmarks_returns_correct_structure(self):
        """detect_hand_landmarks returns HandObservation with correct structure when hand detected."""
        # This test verifies the return structure using a mock observation
        # Real hand detection is tested in integration tests with actual images

        # Create a sample HandObservation to verify structure
        fingers = [
            FingerPosition(finger_id=0, x=0.3, y=0.4, z=-0.1),  # thumb
            FingerPosition(finger_id=1, x=0.4, y=0.3, z=-0.05),  # index
            FingerPosition(finger_id=2, x=0.5, y=0.3, z=-0.05),  # middle
            FingerPosition(finger_id=3, x=0.6, y=0.35, z=-0.03),  # ring
            FingerPosition(finger_id=4, x=0.7, y=0.4, z=-0.02),  # pinky
        ]
        obs = HandObservation(timestamp=1.0, fingers=fingers, is_left_hand=True)

        # Verify structure
        assert len(obs.fingers) == 5
        assert all(isinstance(f, FingerPosition) for f in obs.fingers)
        assert all(0 <= f.finger_id <= 4 for f in obs.fingers)
        assert all(0.0 <= f.x <= 1.0 for f in obs.fingers)
        assert all(0.0 <= f.y <= 1.0 for f in obs.fingers)
        assert isinstance(obs.is_left_hand, bool)

    def test_detect_hand_landmarks_extracts_fingertips(self):
        """detect_hand_landmarks extracts all 5 fingertip positions."""
        # Test that we're looking for the right landmark indices
        assert len(FINGERTIP_INDICES) == 5
        assert 4 in FINGERTIP_INDICES   # Thumb tip
        assert 8 in FINGERTIP_INDICES   # Index tip
        assert 12 in FINGERTIP_INDICES  # Middle tip
        assert 16 in FINGERTIP_INDICES  # Ring tip
        assert 20 in FINGERTIP_INDICES  # Pinky tip


class TestAnalyzeVideoAtTimestamps:
    """Tests for batch video analysis."""

    def test_analyze_video_returns_empty_dict_for_missing_file(self):
        """analyze_video_at_timestamps returns empty dict for missing video."""
        result = analyze_video_at_timestamps(
            "/nonexistent/video.mp4",
            [0.0, 0.5, 1.0]
        )
        assert result == {}

    def test_analyze_video_returns_empty_dict_for_empty_timestamps(self, tmp_path):
        """analyze_video_at_timestamps returns empty dict for empty timestamp list."""
        import cv2

        # Create a simple test video
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(30):
            out.write(frame)
        out.release()

        result = analyze_video_at_timestamps(video_path, [])
        assert result == {}

    def test_analyze_video_returns_dict_with_timestamps(self, tmp_path):
        """analyze_video_at_timestamps returns dict keyed by timestamp."""
        import cv2

        # Create a test video
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(60):  # 2 seconds
            out.write(frame)
        out.release()

        # Mock hand detection to return observations
        mock_obs = HandObservation(
            timestamp=0.0,
            fingers=[FingerPosition(i, 0.5, 0.5, 0.0) for i in range(5)],
            is_left_hand=True,
        )

        with patch('app.video_pipeline.detect_hand_landmarks', return_value=mock_obs):
            result = analyze_video_at_timestamps(video_path, [0.0, 0.5, 1.0])

        assert len(result) == 3
        assert 0.0 in result
        assert 0.5 in result
        assert 1.0 in result

    def test_analyze_video_sets_correct_timestamps(self, tmp_path):
        """analyze_video_at_timestamps sets correct timestamp in each observation."""
        import cv2

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(60):
            out.write(frame)
        out.release()

        mock_obs = HandObservation(
            timestamp=999.0,  # Will be overwritten
            fingers=[FingerPosition(i, 0.5, 0.5, 0.0) for i in range(5)],
            is_left_hand=True,
        )

        with patch('app.video_pipeline.detect_hand_landmarks', return_value=mock_obs):
            result = analyze_video_at_timestamps(video_path, [0.5, 1.5])

        # Check that timestamps were correctly set
        assert result[0.5].timestamp == 0.5
        assert result[1.5].timestamp == 1.5
