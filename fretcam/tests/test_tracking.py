from __future__ import annotations

import cv2
import numpy as np

from fretcam.tracking import OpticalBoardTracker, align_detection_homography
from tabvision.types import Homography


FRAME_HEIGHT = 240
FRAME_WIDTH = 320
BOARD_LEFT = 50.0
BOARD_TOP = 60.0
BOARD_WIDTH = 140.0
BOARD_HEIGHT = 70.0


def _board_homography(
    *,
    left: float = BOARD_LEFT,
    top: float = BOARD_TOP,
    confidence: float = 0.9,
) -> Homography:
    return Homography(
        H=np.asarray(
            [
                [BOARD_WIDTH, 0.0, left],
                [0.0, BOARD_HEIGHT, top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        confidence=confidence,
        method="fixture",
    )


def _textured_frame() -> np.ndarray:
    """Create a deterministic neck texture with many unambiguous corners."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for row, y in enumerate(range(66, 126, 8)):
        for column, x in enumerate(range(56, 186, 8)):
            value = 80 + ((row * 37 + column * 53) % 176)
            cv2.circle(frame, (x, y), 2, (value, value, value), -1)
            inverse = 255 - value
            cv2.line(
                frame,
                (x - 2, y),
                (x + 2, y),
                (inverse, inverse, inverse),
                1,
            )
    return frame


def _translate(
    frame: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> np.ndarray:
    return cv2.warpAffine(
        frame,
        np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64),
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _project_quad(homography: Homography) -> np.ndarray:
    canonical = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(
        canonical.reshape(-1, 1, 2),
        np.asarray(homography.H, dtype=np.float64),
    ).reshape(-1, 2)


def test_tracks_a_smooth_board_translation_between_detector_passes() -> None:
    source = _textured_frame()
    target = _translate(source, dx=7.0, dy=4.0)
    initial = _board_homography()
    tracker = OpticalBoardTracker()

    assert tracker.accept_detection(source, initial, timestamp_s=10.0)
    snapshot = tracker.advance(target, timestamp_s=10.1)

    expected_quad = _project_quad(initial) + np.asarray([7.0, 4.0])
    np.testing.assert_allclose(
        _project_quad(snapshot.homography),
        expected_quad,
        atol=0.05,
    )
    assert snapshot.status == "tracked"
    assert snapshot.homography.method == "optical_flow"
    assert snapshot.flow_inliers >= 8
    assert snapshot.flow_inlier_ratio >= 0.95
    assert snapshot.flow_error_px < 0.05
    assert snapshot.stability > 0.95
    assert snapshot.geometry_age_s == 0.0
    assert np.isclose(snapshot.detector_age_s, 0.1)


def test_repeated_good_flow_does_not_decay_detector_confidence_with_fps() -> None:
    source = _textured_frame()
    initial = _board_homography(confidence=0.87)
    tracker = OpticalBoardTracker()
    assert tracker.accept_detection(source, initial, timestamp_s=0.0)

    snapshot = tracker.snapshot(0.0)
    for index in range(1, 6):
        snapshot = tracker.advance(
            _translate(source, dx=2.0 * index, dy=float(index)),
            timestamp_s=index / 30.0,
        )
        assert snapshot.status == "tracked"

    assert snapshot.homography.confidence == initial.confidence
    assert snapshot.stability > 0.95


def test_optical_flow_rejects_a_canonical_axis_flip(monkeypatch) -> None:
    source = _textured_frame()
    initial = _board_homography()
    tracker = OpticalBoardTracker()
    assert tracker.accept_detection(source, initial, timestamp_s=0.0)
    points = tracker._previous_points
    assert points is not None
    reflected = points.copy()
    reflected[..., 0] = 2.0 * BOARD_LEFT + BOARD_WIDTH - reflected[..., 0]
    statuses = np.ones((len(points), 1), dtype=np.uint8)
    calls = 0

    def fake_lk(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return reflected.copy(), statuses.copy(), None
        return points.copy(), statuses.copy(), None

    reflection = np.asarray(
        [
            [-1.0, 0.0, 2.0 * BOARD_LEFT + BOARD_WIDTH],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *_args, **_kwargs: (
            reflection.copy(),
            np.ones((len(points), 1), dtype=np.uint8),
        ),
    )

    snapshot = tracker.advance(source.copy(), timestamp_s=0.1)

    assert snapshot.status == "held"
    np.testing.assert_array_equal(snapshot.homography.H, initial.H)


def test_failed_flow_is_held_then_stale_then_hard_expires() -> None:
    frame = _textured_frame()
    tracker = OpticalBoardTracker(
        stale_after_s=0.35,
        hard_expire_s=0.90,
        max_features=100,
        min_inliers=101,
    )
    assert tracker.accept_detection(frame, _board_homography(), timestamp_s=0.0)

    held = tracker.advance(frame, timestamp_s=0.20)
    stale = tracker.advance(frame, timestamp_s=0.35)
    expired = tracker.advance(frame, timestamp_s=0.90)

    assert held.status == "held"
    assert held.homography.confidence > 0.0
    assert np.isclose(held.geometry_age_s, 0.20)
    assert stale.status == "stale"
    assert stale.homography.confidence > 0.0
    assert np.isclose(stale.geometry_age_s, 0.35)
    assert expired.status == "missing"
    assert expired.homography.confidence == 0.0
    assert expired.homography.method == "missing"
    assert expired.stability == 0.0
    assert np.isclose(expired.detector_age_s, 0.90)


def test_failed_flow_discards_points_from_the_previous_source_frame() -> None:
    source = _textured_frame()
    tracker = OpticalBoardTracker()
    assert tracker.accept_detection(source, _board_homography(), timestamp_s=0.0)
    assert tracker._previous_points is not None
    assert len(tracker._previous_points) >= 24

    tracker.advance(np.zeros_like(source), timestamp_s=0.1)

    assert tracker._previous_points is None


def test_rejects_an_implausible_fresh_detection_without_mutating_state() -> None:
    frame = _textured_frame()
    tracker = OpticalBoardTracker()
    initial = _board_homography()
    assert tracker.accept_detection(frame, initial, timestamp_s=1.0)

    far_away = _board_homography(left=220.0, top=140.0)
    assert not tracker.accept_detection(frame, far_away, timestamp_s=1.2)

    snapshot = tracker.snapshot(1.2)
    np.testing.assert_array_equal(snapshot.homography.H, initial.H)
    assert snapshot.homography.confidence == initial.confidence
    assert snapshot.status == "detected"
    assert np.isclose(snapshot.detector_age_s, 0.2)
    assert np.isclose(snapshot.geometry_age_s, 0.2)


def test_rejects_same_quad_canonical_axis_flips() -> None:
    frame = _textured_frame()
    initial = _board_homography()

    x_flip = Homography(
        H=np.asarray(
            [
                [-BOARD_WIDTH, 0.0, BOARD_LEFT + BOARD_WIDTH],
                [0.0, BOARD_HEIGHT, BOARD_TOP],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        confidence=0.9,
        method="fixture_x_flip",
    )
    rotation_180 = Homography(
        H=np.asarray(
            [
                [-BOARD_WIDTH, 0.0, BOARD_LEFT + BOARD_WIDTH],
                [0.0, -BOARD_HEIGHT, BOARD_TOP + BOARD_HEIGHT],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        confidence=0.9,
        method="fixture_180",
    )

    for flipped in (x_flip, rotation_180):
        tracker = OpticalBoardTracker()
        assert tracker.accept_detection(frame, initial, timestamp_s=1.0)
        assert not tracker.accept_detection(frame, flipped, timestamp_s=1.1)
        np.testing.assert_array_equal(tracker.snapshot(1.1).homography.H, initial.H)


def test_accepts_a_nearby_fresh_detection_and_resets_geometry_age() -> None:
    frame = _textured_frame()
    tracker = OpticalBoardTracker()
    assert tracker.accept_detection(frame, _board_homography(), timestamp_s=1.0)

    nearby = _board_homography(left=54.0, top=63.0, confidence=0.82)
    assert tracker.accept_detection(frame, nearby, timestamp_s=1.2)

    snapshot = tracker.snapshot(1.2)
    np.testing.assert_array_equal(snapshot.homography.H, nearby.H)
    assert snapshot.homography.confidence == nearby.confidence
    assert snapshot.status == "detected"
    assert snapshot.geometry_age_s == 0.0
    assert snapshot.detector_age_s == 0.0
    assert snapshot.stability == 1.0


def test_frame_resize_rescales_geometry_without_refreshing_its_age() -> None:
    source = _textured_frame()
    tracker = OpticalBoardTracker(
        max_features=100,
        min_inliers=101,
    )
    initial = _board_homography()
    assert tracker.accept_detection(source, initial, timestamp_s=2.0)
    doubled = cv2.resize(
        source,
        (FRAME_WIDTH * 2, FRAME_HEIGHT * 2),
        interpolation=cv2.INTER_NEAREST,
    )

    snapshot = tracker.advance(doubled, timestamp_s=2.1)

    np.testing.assert_allclose(
        _project_quad(snapshot.homography),
        _project_quad(initial) * 2.0,
        atol=1e-6,
    )
    assert snapshot.status == "held"
    assert np.isclose(snapshot.geometry_age_s, 0.1)
    assert np.isclose(snapshot.detector_age_s, 0.1)


def test_aligns_a_delayed_detection_to_the_current_frame() -> None:
    source = _textured_frame()
    target = _translate(source, dx=-5.0, dy=6.0)
    initial = _board_homography()

    aligned_result = align_detection_homography(source, target, initial)

    assert aligned_result is not None
    aligned, inlier_ratio, error_px = aligned_result
    expected_quad = _project_quad(initial) + np.asarray([-5.0, 6.0])
    np.testing.assert_allclose(_project_quad(aligned), expected_quad, atol=0.05)
    assert aligned.method == initial.method
    assert 0.95 <= inlier_ratio <= 1.0
    assert np.isclose(aligned.confidence, initial.confidence * inlier_ratio)
    assert error_px < 0.05


def test_delayed_detection_alignment_rejects_mismatched_frame_sizes() -> None:
    source = _textured_frame()
    target = cv2.resize(source, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))

    assert align_detection_homography(source, target, _board_homography()) is None
