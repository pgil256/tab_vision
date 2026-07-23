from __future__ import annotations

import cv2
import numpy as np
import pytest

from fretcam.detection import FrameDetection, FretTick, HandPoint, StageLatency
from fretcam.position import EstimatorConfig, PositionEstimator
from fretcam.processing import HudFrameProcessor, PositionCalibration
from tabvision.video.hand.neck_anchor import HandNeckAnchor


class FakeChain:
    def __init__(
        self,
        *,
        position_fret: float | None = 5.0,
        observation_confidence: float = 0.8,
        composite_available: bool = True,
    ) -> None:
        self.closed = False
        self.frame_shape: tuple[int, ...] | None = None
        self.position_fret = position_fret
        self.observation_confidence = observation_confidence
        self.composite_available = composite_available
        self.reset_count = 0
        self.player_handedness = "right"

    def process_frame(self, frame: np.ndarray, *, timestamp_s: float) -> FrameDetection:
        self.frame_shape = frame.shape
        return FrameDetection(
            timestamp_s=timestamp_s,
            detector_ran=True,
            neck_locked=True,
            fret_map_locked=True,
            homography_confidence=0.8,
            homography_method="fixture",
            neck_quad=(
                (10.0, 10.0),
                (90.0, 10.0),
                (90.0, 40.0),
                (10.0, 40.0),
            ),
            fret_ticks=(FretTick(5, (50.0, 10.0), (50.0, 40.0)),),
            hand_points=(HandPoint("index", 50.0, 25.0),),
            index_fret=5.0,
            anchor=HandNeckAnchor(5.2, 4.0, 8.0, 0.72, "fixture"),
            stage_latency=StageLatency(1.0, 1.0, 1.0, 1.0, 4.0),
            index_fret_raw=5.2,
            composite_available=self.composite_available,
            position_fret=self.position_fret,
            observation_confidence=self.observation_confidence,
            geometry_status="detected",
            geometry_stability=0.9,
        )

    def reset(self) -> None:
        self.reset_count += 1

    def set_player_handedness(self, value: str) -> None:
        self.player_handedness = value

    def close(self) -> None:
        self.closed = True


def _jpeg(*, width: int = 100, height: int = 50) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    assert ok
    return encoded.tobytes()


def test_processor_emits_complete_json_hud_contract_and_closes() -> None:
    chain = FakeChain()
    processor = HudFrameProcessor(
        chain=chain,  # type: ignore[arg-type]
        estimator=PositionEstimator(
            EstimatorConfig(acquisition_duration_s=0.0, shift_duration_s=0.0)
        ),
    )

    result = processor.process_jpeg(_jpeg())
    processor.close()

    assert result["type"] == "hud"
    assert result["version"] == 2
    assert result["frame"] == {"width": 100, "height": 50}
    assert result["detection"]["neck_locked"] is True  # type: ignore[index]
    assert result["detection"]["index_fret"] == 5.0  # type: ignore[index]
    assert result["detection"]["index_fret_raw"] == 5.2  # type: ignore[index]
    assert result["detection"]["geometry_status"] == "detected"  # type: ignore[index]
    assert result["position"]["label"] == "Position V"  # type: ignore[index]
    assert result["position"]["observation_confidence"] == 0.8  # type: ignore[index]
    assert result["guidance"]["code"] == "locked"  # type: ignore[index]
    assert result["calibration"]["status"] == "idle"  # type: ignore[index]
    assert result["server_ms"] > 0
    assert chain.frame_shape == (50, 100, 3)
    assert chain.closed


def test_processor_prefers_composite_position_observation_over_index_finger() -> None:
    chain = FakeChain(position_fret=8.0, observation_confidence=0.9)
    processor = HudFrameProcessor(
        chain=chain,  # type: ignore[arg-type]
        estimator=PositionEstimator(
            EstimatorConfig(acquisition_duration_s=0.0, shift_duration_s=0.0)
        ),
    )

    try:
        result = processor.process_jpeg(_jpeg())
    finally:
        processor.close()

    assert result["detection"]["index_fret"] == 5.0  # type: ignore[index]
    assert result["position"]["raw_index_fret"] == 8.0  # type: ignore[index]
    assert result["position"]["label"] == "Position VIII"  # type: ignore[index]
    assert result["position"]["observation_confidence"] == 0.9  # type: ignore[index]


def test_processor_does_not_bypass_a_composite_abstention_with_index_fallback() -> None:
    chain = FakeChain(
        position_fret=None,
        observation_confidence=0.0,
        composite_available=True,
    )
    processor = HudFrameProcessor(
        chain=chain,  # type: ignore[arg-type]
        estimator=PositionEstimator(
            EstimatorConfig(acquisition_duration_s=0.0, shift_duration_s=0.0)
        ),
    )

    try:
        result = processor.process_jpeg(_jpeg())
    finally:
        processor.close()

    assert result["detection"]["index_fret"] == 5.0  # type: ignore[index]
    assert result["position"]["state"] == "lost"  # type: ignore[index]
    assert result["position"]["position"] is None  # type: ignore[index]


def test_processor_retains_index_fallback_for_legacy_detection_fixtures() -> None:
    chain = FakeChain(
        position_fret=None,
        observation_confidence=0.0,
        composite_available=False,
    )
    processor = HudFrameProcessor(
        chain=chain,  # type: ignore[arg-type]
        estimator=PositionEstimator(
            EstimatorConfig(acquisition_duration_s=0.0, shift_duration_s=0.0)
        ),
    )

    try:
        result = processor.process_jpeg(_jpeg())
    finally:
        processor.close()

    assert result["position"]["label"] == "Position V"  # type: ignore[index]


def test_processor_caps_inference_frame_while_preserving_aspect_ratio() -> None:
    chain = FakeChain()
    processor = HudFrameProcessor(chain=chain)  # type: ignore[arg-type]

    try:
        result = processor.process_jpeg(_jpeg(width=1920, height=1080))
    finally:
        processor.close()

    assert result["frame"] == {"width": 640, "height": 360}
    assert chain.frame_shape == (360, 640, 3)


def test_session_reset_starts_a_new_mediapipe_video_stream() -> None:
    class RuntimeAwareChain(FakeChain):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_resets: list[bool] = []

        def reset_tracking(self, *, reset_hand_runtime: bool = False) -> None:
            self.runtime_resets.append(reset_hand_runtime)

        def wait_for_background_detector(self) -> None:
            return None

    chain = RuntimeAwareChain()
    processor = HudFrameProcessor(chain=chain)  # type: ignore[arg-type]

    try:
        processor.warmup()
        processor.reset()
    finally:
        processor.close()

    assert chain.runtime_resets == [False, True]


def test_processor_controls_handedness_and_session_calibration() -> None:
    chain = FakeChain()
    processor = HudFrameProcessor(chain=chain)  # type: ignore[arg-type]

    try:
        settings = processor.handle_control(
            {"type": "settings", "player_handedness": "left"}
        )
        started = processor.handle_control({"type": "calibrate"})
        two_point = processor.handle_control(
            {"type": "calibrate_two_point", "upper_position": 9}
        )
        reset = processor.handle_control({"type": "reset_calibration"})

        assert settings == {
            "type": "control",
            "status": "settings_applied",
            "player_handedness": "left",
        }
        assert chain.player_handedness == "left"
        assert started["status"] == "calibration_started"
        assert started["calibration"]["status"] == "collecting"  # type: ignore[index]
        assert two_point["status"] == "two_point_calibration_started"
        assert two_point["calibration"]["mode"] == "two_point"  # type: ignore[index]
        assert two_point["calibration"]["target_position"] == 1  # type: ignore[index]
        assert reset["status"] == "calibration_reset"
        assert reset["calibration"]["status"] == "idle"  # type: ignore[index]

        with pytest.raises(ValueError, match="player_handedness"):
            processor.handle_control(
                {"type": "settings", "player_handedness": "ambidextrous"}
            )
        with pytest.raises(ValueError, match="unknown control"):
            processor.handle_control({"type": "bogus"})
        with pytest.raises(ValueError, match="upper_position"):
            processor.handle_control(
                {"type": "calibrate_two_point", "upper_position": 7}
            )
        with pytest.raises(ValueError, match="upper_position"):
            processor.handle_control(
                {"type": "calibrate_two_point", "upper_position": None}
            )
    finally:
        processor.close()


def test_position_calibration_uses_robust_session_only_residual() -> None:
    calibration = PositionCalibration(
        min_samples=4,
        min_duration_s=0.3,
        max_duration_s=1.0,
    )
    calibration.start(target_position=1, timestamp_s=0.0)

    completed = [
        calibration.observe(
            value,
            confidence=0.8,
            geometry_status="tracked",
            timestamp_s=timestamp,
        )
        for value, timestamp in zip(
            (1.25, 1.20, 1.30, 1.25),
            (0.0, 0.1, 0.2, 0.31),
            strict=True,
        )
    ]

    assert completed == [False, False, False, True]
    assert calibration.state().status == "calibrated"
    assert calibration.state().offset_fret == pytest.approx(-0.25)
    assert calibration.state().samples == 4
    assert calibration.apply(5.0) == pytest.approx(4.75)


def test_two_point_calibration_corrects_upper_neck_scale_and_offset() -> None:
    calibration = PositionCalibration(
        min_samples=3,
        min_duration_s=0.2,
        max_duration_s=1.0,
    )
    calibration.start_two_point(upper_position=9, timestamp_s=0.0)

    for value, timestamp in zip(
        (1.2, 1.2, 1.2),
        (0.0, 0.1, 0.21),
        strict=True,
    ):
        calibration.observe(
            value,
            confidence=0.9,
            geometry_status="tracked",
            timestamp_s=timestamp,
        )

    first = calibration.state()
    assert first.status == "awaiting_second"
    assert first.anchors == ((1, 1.2),)
    assert first.next_target_position == 9
    calibration.continue_next(timestamp_s=0.3)

    for value, timestamp in zip(
        (8.4, 8.4, 8.4),
        (0.3, 0.4, 0.51),
        strict=True,
    ):
        calibration.observe(
            value,
            confidence=0.9,
            geometry_status="detected",
            timestamp_s=timestamp,
        )

    state = calibration.state()
    assert state.status == "calibrated"
    assert state.mode == "two_point"
    assert state.scale == pytest.approx(8.0 / 7.2, abs=1e-4)
    assert calibration.apply(1.2) == pytest.approx(1.0)
    assert calibration.apply(8.4) == pytest.approx(9.0)


def test_two_point_calibration_rejects_implausible_scale() -> None:
    calibration = PositionCalibration(
        min_samples=3,
        min_duration_s=0.2,
        max_duration_s=1.0,
    )
    calibration.start_two_point(upper_position=5, timestamp_s=0.0)
    for timestamp in (0.0, 0.1, 0.21):
        calibration.observe(
            1.0,
            confidence=0.9,
            geometry_status="tracked",
            timestamp_s=timestamp,
        )
    calibration.continue_next(timestamp_s=0.3)
    for timestamp in (0.3, 0.4, 0.51):
        calibration.observe(
            3.0,
            confidence=0.9,
            geometry_status="tracked",
            timestamp_s=timestamp,
        )

    assert calibration.state().status == "failed"
    assert "scale" in calibration.state().message
    assert calibration.apply(5.0) == 5.0


def test_processor_pushes_two_point_transform_into_detection_chain() -> None:
    class CalibratableChain(FakeChain):
        def __init__(self) -> None:
            super().__init__(position_fret=1.2)
            self.transforms: list[tuple[float, float]] = []

        def set_position_calibration(self, *, scale: float, offset: float) -> None:
            self.transforms.append((scale, offset))

    chain = CalibratableChain()
    processor = HudFrameProcessor(chain=chain)  # type: ignore[arg-type]
    processor.calibration = PositionCalibration(
        min_samples=3,
        min_duration_s=0.0,
        max_duration_s=1.0,
    )

    try:
        processor.handle_control({"type": "calibrate_two_point", "upper_position": 9})
        for _ in range(3):
            processor.process_jpeg(_jpeg())
        assert processor.calibration.state().status == "awaiting_second"

        processor.handle_control({"type": "continue_calibration"})
        chain.position_fret = 8.4
        for _ in range(3):
            processor.process_jpeg(_jpeg())
    finally:
        processor.close()

    assert processor.calibration.state().status == "calibrated"
    scale, offset = chain.transforms[-1]
    assert scale == pytest.approx(8.0 / 7.2)
    assert offset == pytest.approx(1.0 - scale * 1.2)


@pytest.mark.parametrize("fps", [2.0, 4.0, 8.0])
def test_default_calibration_completes_at_supported_live_frame_rates(
    fps: float,
) -> None:
    calibration = PositionCalibration()
    calibration.start(target_position=1, timestamp_s=0.0)

    timestamp = 0.0
    while calibration.state().status == "collecting" and timestamp <= 2.0:
        calibration.observe(
            1.25,
            confidence=0.8,
            geometry_status="tracked",
            timestamp_s=timestamp,
        )
        timestamp += 1.0 / fps

    assert calibration.state().status == "calibrated"
    assert calibration.state().samples >= 3
    assert calibration.state().offset_fret == pytest.approx(-0.25)


@pytest.mark.parametrize(
    ("values", "expected_message"),
    [
        ((0.5, 1.5, 0.5, 1.5), "unstable"),
        ((3.0, 3.0, 3.0, 3.0), "implausible"),
    ],
)
def test_position_calibration_rejects_unreliable_residuals(
    values: tuple[float, ...],
    expected_message: str,
) -> None:
    calibration = PositionCalibration(
        min_samples=4,
        min_duration_s=0.3,
        max_duration_s=1.0,
    )
    calibration.start(target_position=1, timestamp_s=0.0)

    for value, timestamp in zip(
        values,
        (0.0, 0.1, 0.2, 0.31),
        strict=True,
    ):
        calibration.observe(
            value,
            confidence=0.8,
            geometry_status="detected",
            timestamp_s=timestamp,
        )

    assert calibration.state().status == "failed"
    assert expected_message in calibration.state().message
    assert calibration.apply(5.0) == 5.0


def test_position_calibration_times_out_without_fresh_confident_geometry() -> None:
    calibration = PositionCalibration(
        min_samples=4,
        min_duration_s=0.3,
        max_duration_s=0.8,
    )
    calibration.start(target_position=1, timestamp_s=0.0)

    assert (
        calibration.observe(
            1.0,
            confidence=0.9,
            geometry_status="stale",
            timestamp_s=0.4,
        )
        is False
    )
    assert (
        calibration.observe(
            1.0,
            confidence=0.2,
            geometry_status="tracked",
            timestamp_s=0.8,
        )
        is True
    )
    assert calibration.state().status == "failed"
    assert calibration.state().samples == 0
    assert "timed out" in calibration.state().message


def test_processor_rejects_non_jpeg_payload() -> None:
    processor = HudFrameProcessor(chain=FakeChain())  # type: ignore[arg-type]

    try:
        try:
            processor.process_jpeg(b"not a jpeg")
        except ValueError as exc:
            assert "decodable" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("invalid JPEG was accepted")
    finally:
        processor.close()
