from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fretcam.hand_tracking import (
    FINGER_LANDMARKS,
    LandmarkObservation,
    OpticalFlowResult,
    TemporalHandLandmarkTracker,
    TemporalHandTrackerConfig,
    VideoModeTimestampClock,
)


FRAME_SHAPE = (240, 320, 3)


def _frame(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _hand_points(*, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    points = np.asarray(
        [
            (100, 180),  # wrist
            (90, 160),
            (80, 145),
            (70, 135),
            (60, 125),  # thumb
            (80, 140),
            (78, 115),
            (78, 95),
            (78, 78),  # index
            (95, 135),
            (95, 105),
            (95, 82),
            (95, 62),  # middle
            (110, 138),
            (112, 110),
            (112, 90),
            (112, 73),  # ring
            (125, 145),
            (130, 122),
            (132, 105),
            (134, 90),  # pinky
        ],
        dtype=np.float64,
    )
    points[:, 0] += dx
    points[:, 1] += dy
    return points


def _observation(
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    confidence: float = 0.9,
    joint_quality: np.ndarray | None = None,
    points: np.ndarray | None = None,
) -> LandmarkObservation:
    return LandmarkObservation(
        landmarks_xy=(
            _hand_points(dx=dx, dy=dy)
            if points is None
            else np.asarray(points, dtype=np.float64)
        ),
        landmarks_z=np.linspace(0.0, -0.08, 21),
        confidence=confidence,
        joint_quality=joint_quality,
        is_left_hand=True,
    )


def test_video_mode_clock_supplies_strict_millisecond_timestamps() -> None:
    clock = VideoModeTimestampClock()

    assert clock.next_ms(10.0) == 10_000
    assert clock.next_ms(10.0001) == 10_001
    with pytest.raises(ValueError, match="strictly increasing"):
        clock.next_ms(10.0001)

    clock.reset()
    assert clock.next_ms(0.0) == 0


def test_observation_converts_mediapipe_normalised_landmarks() -> None:
    landmarks = [
        SimpleNamespace(x=index / 20.0, y=0.25, z=-index / 100.0) for index in range(21)
    ]

    observation = LandmarkObservation.from_mediapipe(
        landmarks,
        frame_width=200,
        frame_height=100,
        confidence=0.8,
        is_left_hand=False,
    )

    assert observation.landmarks_xy[20] == pytest.approx((200.0, 25.0))
    assert observation.landmarks_z[20] == pytest.approx(-0.2)
    assert observation.confidence == pytest.approx(0.8)
    assert observation.is_left_hand is False
    assert not observation.landmarks_xy.flags.writeable


def test_one_euro_filter_smooths_a_valid_whole_hand_jump() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )
    assert tracker.next_video_timestamp_ms(0.0) == 0
    first = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    assert tracker.next_video_timestamp_ms(1.0 / 30.0) == 33
    second = tracker.update(
        _frame(),
        timestamp_s=1.0 / 30.0,
        observation=_observation(dx=30.0),
    )

    assert first is not None
    assert second is not None
    assert second.detector_timestamp_ms == 33
    raw_tip_x = _hand_points(dx=30.0)[8, 0]
    assert first.landmarks_xy[8, 0] < second.landmarks_xy[8, 0] < raw_tip_x
    assert second.finger_quality["index"].source == "detector"
    assert second.finger_quality["index"].detector_observed
    assert second.fretting_finger_quality["index"] > 0.5


def test_lk_flow_propagates_joints_between_detector_observations() -> None:
    calls: list[np.ndarray] = []

    def translated_flow(
        _previous: np.ndarray,
        _current: np.ndarray,
        points: np.ndarray,
    ) -> OpticalFlowResult:
        calls.append(points.copy())
        return OpticalFlowResult(
            points_xy=points + np.asarray([4.0, 2.0]),
            status=np.ones(len(points), dtype=bool),
            error_px=np.zeros(len(points), dtype=np.float64),
        )

    tracker = TemporalHandLandmarkTracker(optical_flow=translated_flow)
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    propagated = tracker.update(
        _frame(10),
        timestamp_s=0.05,
        observation=None,
    )

    assert detected is not None
    assert propagated is not None
    assert calls
    assert propagated.used_optical_flow
    assert propagated.landmarks_xy[8, 0] > detected.landmarks_xy[8, 0]
    index_quality = propagated.finger_quality["index"]
    assert index_quality.source == "optical_flow"
    assert index_quality.flow_inlier_ratio == pytest.approx(1.0)
    assert not index_quality.detector_observed
    assert index_quality.age_ms == pytest.approx(50.0)


def test_lk_rejects_one_finger_jump_and_holds_only_that_finger() -> None:
    def isolated_index_jump(
        _previous: np.ndarray,
        _current: np.ndarray,
        points: np.ndarray,
    ) -> OpticalFlowResult:
        propagated = points + np.asarray([3.0, 1.0])
        propagated[np.asarray(FINGER_LANDMARKS["index"]), 0] += 60.0
        return OpticalFlowResult(
            points_xy=propagated,
            status=np.ones(len(points), dtype=bool),
            error_px=np.zeros(len(points), dtype=np.float64),
        )

    tracker = TemporalHandLandmarkTracker(optical_flow=isolated_index_jump)
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    tracked = tracker.update(
        _frame(10),
        timestamp_s=0.05,
        observation=None,
    )

    assert detected is not None
    assert tracked is not None
    assert tracked.used_optical_flow
    assert tracked.finger_quality["index"].source == "held_rejected"
    assert tracked.landmarks_xy[8] == pytest.approx(detected.landmarks_xy[8])
    for name in ("middle", "ring", "pinky"):
        assert tracked.finger_quality[name].source == "optical_flow"


def test_lk_accepts_a_coherent_fast_whole_hand_shift() -> None:
    def coherent_shift(
        _previous: np.ndarray,
        _current: np.ndarray,
        points: np.ndarray,
    ) -> OpticalFlowResult:
        return OpticalFlowResult(
            points_xy=points + np.asarray([70.0, 0.0]),
            status=np.ones(len(points), dtype=bool),
            error_px=np.zeros(len(points), dtype=np.float64),
        )

    tracker = TemporalHandLandmarkTracker(optical_flow=coherent_shift)
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    tracked = tracker.update(
        _frame(10),
        timestamp_s=0.05,
        observation=None,
    )

    assert detected is not None
    assert tracked is not None
    assert tracked.used_optical_flow
    assert tracked.landmarks_xy[8, 0] > detected.landmarks_xy[8, 0]
    for name in ("index", "middle", "ring", "pinky"):
        assert tracked.finger_quality[name].source == "optical_flow"


def test_flow_requires_a_valid_tracked_fingertip() -> None:
    def missing_index_tip(
        _previous: np.ndarray,
        _current: np.ndarray,
        points: np.ndarray,
    ) -> OpticalFlowResult:
        status = np.ones(len(points), dtype=bool)
        # Every point is finite in the fixture, so LK input order is landmark order.
        status[FINGER_LANDMARKS["index"][-1]] = False
        return OpticalFlowResult(
            points_xy=points + 2.0,
            status=status,
            error_px=np.zeros(len(points), dtype=np.float64),
        )

    tracker = TemporalHandLandmarkTracker(optical_flow=missing_index_tip)
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    propagated = tracker.update(
        _frame(),
        timestamp_s=0.04,
        observation=None,
    )

    assert detected is not None
    assert propagated is not None
    assert propagated.finger_quality["index"].source == "held"
    assert propagated.finger_quality["middle"].source == "optical_flow"
    assert propagated.landmarks_xy[8] == pytest.approx(detected.landmarks_xy[8])


def test_impossible_segment_length_rejects_only_the_bad_finger() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    corrupted = _hand_points(dx=2.0)
    corrupted[7] = (290.0, 20.0)
    rejected = tracker.update(
        _frame(),
        timestamp_s=0.04,
        observation=_observation(points=corrupted),
    )

    assert detected is not None
    assert rejected is not None
    index_quality = rejected.finger_quality["index"]
    assert index_quality.source == "held_rejected"
    assert not index_quality.detector_observed
    assert index_quality.quality < detected.finger_quality["index"].quality
    assert rejected.landmarks_xy[7] == pytest.approx(detected.landmarks_xy[7])
    assert rejected.finger_quality["middle"].source == "detector"
    assert rejected.finger_quality["middle"].detector_observed


def test_isolated_detector_finger_jump_is_held_then_recovers() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    jumped = _hand_points()
    jumped[np.asarray(FINGER_LANDMARKS["index"]), 0] += 80.0
    rejected = tracker.update(
        _frame(),
        timestamp_s=0.10,
        observation=_observation(points=jumped),
    )
    recovered = tracker.update(
        _frame(),
        timestamp_s=0.20,
        observation=_observation(dx=2.0),
    )

    assert detected is not None
    assert rejected is not None
    assert recovered is not None
    assert rejected.finger_quality["index"].source == "held_rejected"
    assert not rejected.finger_quality["index"].detector_observed
    assert rejected.landmarks_xy[8] == pytest.approx(detected.landmarks_xy[8])
    assert rejected.finger_quality["middle"].source == "detector"
    assert recovered.finger_quality["index"].source == "detector"
    assert recovered.finger_quality["index"].detector_observed


def test_coherent_fast_whole_hand_shift_passes_innovation_gate() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )
    before = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    shifted = tracker.update(
        _frame(),
        timestamp_s=0.10,
        observation=_observation(dx=80.0),
    )

    assert before is not None
    assert shifted is not None
    assert shifted.landmarks_xy[8, 0] > before.landmarks_xy[8, 0]
    for name in ("index", "middle", "ring", "pinky"):
        assert shifted.finger_quality[name].source == "detector"
        assert shifted.finger_quality[name].detector_observed


def test_impossible_joint_angle_is_rejected() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )
    detected = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    folded = _hand_points()
    # MCP -> PIP -> DIP doubles directly back with realistic segment lengths.
    folded[5] = (78.0, 140.0)
    folded[6] = (78.0, 115.0)
    folded[7] = (78.0, 139.0)
    folded[8] = (78.0, 157.0)
    rejected = tracker.update(
        _frame(),
        timestamp_s=0.03,
        observation=_observation(points=folded),
    )

    assert detected is not None
    assert rejected is not None
    assert rejected.finger_quality["index"].source == "held_rejected"
    assert rejected.landmarks_xy[8] == pytest.approx(detected.landmarks_xy[8])


def test_strong_fingertips_survive_short_occlusion_then_expire() -> None:
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(
            use_optical_flow=False,
            max_occlusion_s=0.18,
        )
    )
    initial = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(),
    )
    at_120_ms = tracker.update(
        _frame(),
        timestamp_s=0.12,
        observation=None,
    )
    at_179_ms = tracker.update(
        _frame(),
        timestamp_s=0.179,
        observation=None,
    )
    expired = tracker.update(
        _frame(),
        timestamp_s=0.181,
        observation=None,
    )

    assert initial is not None
    assert at_120_ms is not None
    assert at_179_ms is not None
    assert np.all(np.isfinite(at_179_ms.landmarks_xy[8]))
    assert at_120_ms.finger_quality["index"].source == "held"
    assert (
        0.0
        < at_120_ms.finger_quality["index"].quality
        < initial.finger_quality["index"].quality
    )
    assert expired is None


def test_per_finger_quality_distinguishes_missing_finger() -> None:
    joint_quality = np.ones(21, dtype=np.float64)
    joint_quality[np.asarray(FINGER_LANDMARKS["pinky"])] = 0.0
    tracker = TemporalHandLandmarkTracker(
        TemporalHandTrackerConfig(use_optical_flow=False)
    )

    result = tracker.update(
        _frame(),
        timestamp_s=0.0,
        observation=_observation(joint_quality=joint_quality),
    )

    assert result is not None
    assert result.finger_quality["index"].detector_observed
    assert result.finger_quality["index"].quality > 0.5
    assert result.finger_quality["pinky"].quality == 0.0
    assert not result.finger_quality["pinky"].retained
    assert not result.finger_quality["pinky"].detector_observed
    assert np.all(np.isnan(result.landmarks_xy[list(FINGER_LANDMARKS["pinky"])]))


def test_track_update_and_video_clock_reject_non_finite_or_old_time() -> None:
    tracker = TemporalHandLandmarkTracker()
    tracker.update(
        _frame(),
        timestamp_s=1.0,
        observation=_observation(),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update(
            _frame(),
            timestamp_s=1.0,
            observation=None,
        )
    with pytest.raises(ValueError, match="finite"):
        tracker.next_video_timestamp_ms(float("nan"))
