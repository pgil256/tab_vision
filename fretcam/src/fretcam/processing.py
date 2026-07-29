"""In-memory JPEG to live-HUD response processing."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Protocol

import cv2
import numpy as np

from fretcam.detection import DetectionChain, FrameDetection, HandSearchHint
from fretcam.guidance import assess_guidance
from fretcam.position import PositionEstimate, PositionEstimator


class FrameProcessor(Protocol):
    def warmup(self) -> None: ...

    def reset(self) -> None: ...

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float | None = None,
    ) -> dict[str, object]: ...

    def handle_control(self, message: dict[str, object]) -> dict[str, object]: ...

    def close(self) -> None: ...


FrameProcessorFactory = Callable[[], FrameProcessor]


def build_hand_search_hint(
    detection: FrameDetection,
    estimate: PositionEstimate,
) -> HandSearchHint:
    """Build the shared cadence feedback used by live and direct replay paths."""
    return HandSearchHint(
        position_state=estimate.state,
        position_reason=estimate.reason,
        observation_confidence=estimate.observation_confidence,
        landmark_quality=detection.confidence_factors.landmark_quality,
        blockers=tuple(detection.confidence_factors.blockers),
        hand_visible=bool(detection.hand_points),
        pose_quality=detection.hand_pose_quality,
    )


@dataclass(frozen=True)
class CalibrationState:
    status: str = "idle"
    target_position: int = 1
    offset_fret: float = 0.0
    scale: float = 1.0
    mode: str = "single"
    anchors: tuple[tuple[int, float], ...] = ()
    next_target_position: int | None = None
    samples: int = 0
    message: str = "Optional Position-I calibration is idle."

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PositionCalibration:
    """Session-only robust residual calibration; never stores frame data."""

    def __init__(
        self,
        *,
        min_samples: int = 3,
        min_duration_s: float = 0.75,
        max_duration_s: float = 3.0,
        acquisition_timeout_s: float | None = None,
        max_mad_fret: float = 0.35,
        max_offset_fret: float = 1.5,
        max_scale_error: float = 0.35,
    ) -> None:
        if min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        if not 0.0 <= min_duration_s < max_duration_s:
            raise ValueError("expected 0 <= min_duration_s < max_duration_s")
        self.min_samples = min_samples
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.acquisition_timeout_s = (
            max_duration_s if acquisition_timeout_s is None else acquisition_timeout_s
        )
        if self.acquisition_timeout_s <= 0.0:
            raise ValueError("acquisition_timeout_s must be positive")
        self.max_mad_fret = max_mad_fret
        self.max_offset_fret = max_offset_fret
        self.max_scale_error = max_scale_error
        self.reset()

    def reset(self) -> None:
        self._status = "idle"
        self._target = 1
        self._offset = 0.0
        self._scale = 1.0
        self._mode = "single"
        self._upper_target = 5
        self._anchors: list[tuple[int, float]] = []
        self._values: list[float] = []
        self._started_s: float | None = None
        self._first_sample_s: float | None = None
        self._message = "Optional Position-I calibration is idle."

    def start(
        self,
        *,
        target_position: int = 1,
        timestamp_s: float | None = None,
    ) -> None:
        if target_position < 1:
            raise ValueError("target_position must be positive")
        self._mode = "single"
        self._anchors = []
        self._scale = 1.0
        self._offset = 0.0
        self._begin_collection(
            target_position,
            timestamp_s=timestamp_s,
        )

    def start_two_point(
        self,
        *,
        upper_position: int = 5,
        timestamp_s: float | None = None,
    ) -> None:
        """Start Position-I plus Position-V/IX scale-and-offset calibration."""
        if upper_position not in {5, 9}:
            raise ValueError("upper_position must be 5 or 9")
        self._mode = "two_point"
        self._upper_target = upper_position
        self._anchors = []
        self._scale = 1.0
        self._offset = 0.0
        self._begin_collection(1, timestamp_s=timestamp_s)
        self._message = (
            f"Two-point calibration: hold Position I steadily, then "
            f"Position {upper_position}."
        )

    def continue_next(self, *, timestamp_s: float | None = None) -> None:
        """Begin the upper-position capture after the Position-I anchor."""
        if (
            self._mode != "two_point"
            or self._status != "awaiting_second"
            or len(self._anchors) != 1
        ):
            raise ValueError("two-point calibration is not awaiting its second point")
        self._begin_collection(
            self._upper_target,
            timestamp_s=timestamp_s,
        )

    def _begin_collection(
        self,
        target_position: int,
        *,
        timestamp_s: float | None,
    ) -> None:
        self._status = "collecting"
        self._target = target_position
        self._values = []
        self._started_s = time.monotonic() if timestamp_s is None else timestamp_s
        self._first_sample_s = None
        self._message = f"Hold Position {target_position} steadily."

    def observe(
        self,
        value: float | None,
        *,
        confidence: float,
        geometry_status: str,
        timestamp_s: float,
    ) -> bool:
        """Collect one scalar observation; return True when state completes."""
        if self._status != "collecting":
            return False
        fresh_geometry = geometry_status in {"detected", "tracked"}
        if value is not None and confidence >= 0.45 and fresh_geometry:
            if self._first_sample_s is None:
                self._first_sample_s = timestamp_s
            self._values.append(float(value))
        stable_elapsed = (
            0.0
            if self._first_sample_s is None
            else max(0.0, timestamp_s - self._first_sample_s)
        )
        acquisition_elapsed = (
            0.0 if self._started_s is None else max(0.0, timestamp_s - self._started_s)
        )
        if (
            len(self._values) >= self.min_samples
            and stable_elapsed >= self.min_duration_s
        ):
            median = float(statistics.median(self._values))
            mad = float(
                statistics.median(abs(value - median) for value in self._values)
            )
            if mad > self.max_mad_fret:
                self._fail("Calibration was unstable; hold one position steadily.")
            elif self._mode == "two_point" and not self._anchors:
                self._anchors.append((self._target, median))
                self._status = "awaiting_second"
                self._message = (
                    f"Position I captured. Move to Position {self._upper_target}, "
                    "then continue calibration."
                )
            elif self._mode == "two_point":
                self._complete_two_point(median)
            else:
                self._complete_single(median)
            return True
        timed_out = (
            self._first_sample_s is None
            and acquisition_elapsed >= self.acquisition_timeout_s
        ) or (
            self._first_sample_s is not None and stable_elapsed >= self.max_duration_s
        )
        if timed_out:
            self._fail("Calibration timed out without enough reliable observations.")
            return True
        return False

    def apply(self, value: float | None) -> float | None:
        if value is None or self._status != "calibrated":
            return value
        return self._scale * value + self._offset

    def state(self) -> CalibrationState:
        return CalibrationState(
            status=self._status,
            target_position=self._target,
            offset_fret=round(self._offset, 3),
            scale=round(self._scale, 4),
            mode=self._mode,
            anchors=tuple((target, round(raw, 3)) for target, raw in self._anchors),
            next_target_position=(
                self._upper_target if self._status == "awaiting_second" else None
            ),
            samples=len(self._values),
            message=self._message,
        )

    def transform(self) -> tuple[float, float]:
        """Return the exact session scale and offset (unrounded)."""
        return self._scale, self._offset

    def _complete_single(self, median: float) -> None:
        offset = float(self._target) - median
        if abs(offset) > self.max_offset_fret:
            self._fail("Calibration offset was implausible; reframe the neck.")
            return
        self._offset = offset
        self._scale = 1.0
        self._anchors = [(self._target, median)]
        self._status = "calibrated"
        self._message = f"Calibrated with {offset:+.2f} fret residual for this session."

    def _complete_two_point(self, median: float) -> None:
        first_target, first_raw = self._anchors[0]
        raw_span = median - first_raw
        target_span = float(self._target - first_target)
        if raw_span <= 1.0 or target_span <= 0.0:
            self._fail(
                "Calibration points were not separated enough; "
                "reframe and capture Position I plus the upper position."
            )
            return
        scale = target_span / raw_span
        offset = float(first_target) - scale * first_raw
        if abs(scale - 1.0) > self.max_scale_error:
            self._fail("Calibration scale was implausible; reframe the full neck.")
            return
        if abs(offset) > self.max_offset_fret:
            self._fail("Calibration offset was implausible; reframe the neck.")
            return
        self._anchors.append((self._target, median))
        self._scale = scale
        self._offset = offset
        self._status = "calibrated"
        self._message = (
            f"Two-point calibration active (scale {scale:.3f}, offset {offset:+.2f})."
        )

    def _fail(self, message: str) -> None:
        self._status = "failed"
        self._offset = 0.0
        self._scale = 1.0
        self._message = message


class HudFrameProcessor:
    """Own one stateful detection and position chain per WebSocket session."""

    def __init__(
        self,
        *,
        chain: DetectionChain | None = None,
        estimator: PositionEstimator | None = None,
        max_frame_width: int = 640,
        max_frame_height: int = 480,
    ) -> None:
        self.chain = chain or DetectionChain(
            detector_hz=1.0,
            background_detector=True,
            crop_hand=True,
        )
        self.estimator = estimator or PositionEstimator()
        self.calibration = PositionCalibration()
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self._lock = RLock()
        self._chain_calibration_active = False

    def warmup(self) -> None:
        """Pay one-time model initialization before the server reports ready."""
        with self._lock:
            self.chain.process_frame(
                np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=0.0
            )
            waiter = getattr(self.chain, "wait_for_background_detector", None)
            if waiter is not None:
                waiter()
            self._reset_tracking(reset_hand_runtime=False)
            self.estimator.reset()
            self.calibration.reset()
            self._chain_calibration_active = False

    def reset(self) -> None:
        with self._lock:
            self._reset_tracking(reset_hand_runtime=True)
            self.estimator.reset()
            self.calibration.reset()
            self._chain_calibration_active = False

    def _reset_tracking(self, *, reset_hand_runtime: bool) -> None:
        resetter = getattr(self.chain, "reset_tracking", None)
        if resetter is None:
            self.chain.reset()
        else:
            try:
                resetter(reset_hand_runtime=reset_hand_runtime)
            except TypeError:
                resetter()

    def handle_control(self, message: dict[str, object]) -> dict[str, object]:
        with self._lock:
            message_type = message.get("type")
            if message_type == "settings":
                handedness = str(message.get("player_handedness", "right"))
                if handedness not in {"right", "left"}:
                    raise ValueError("player_handedness must be 'right' or 'left'")
                self.chain.set_player_handedness(handedness)
                self.estimator.reset()
                self.calibration.reset()
                self._clear_chain_calibration()
                return {
                    "type": "control",
                    "status": "settings_applied",
                    "player_handedness": handedness,
                }
            if message_type == "calibrate":
                self._clear_chain_calibration()
                self.calibration.start(target_position=1)
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "calibration_started",
                    "calibration": self.calibration.state().as_dict(),
                }
            if message_type == "calibrate_two_point":
                self._clear_chain_calibration()
                raw_upper_position = message.get("upper_position", 5)
                if isinstance(raw_upper_position, bool) or not isinstance(
                    raw_upper_position,
                    (int, float, str),
                ):
                    raise ValueError("upper_position must be 5 or 9")
                try:
                    upper_position = int(raw_upper_position)
                except (TypeError, ValueError) as exc:
                    raise ValueError("upper_position must be 5 or 9") from exc
                self.calibration.start_two_point(
                    upper_position=upper_position,
                )
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "two_point_calibration_started",
                    "calibration": self.calibration.state().as_dict(),
                }
            if message_type == "continue_calibration":
                self.calibration.continue_next()
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "calibration_continued",
                    "calibration": self.calibration.state().as_dict(),
                }
            if message_type == "reacquire":
                # Live board re-acquisition: clear tracking and estimator state
                # (a stale quad from a mid-session guitar entry has no other
                # recovery) while preserving handedness and any accepted
                # session calibration.
                self._reset_tracking(reset_hand_runtime=True)
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "board_reacquired",
                    "calibration": self.calibration.state().as_dict(),
                }
            if message_type == "reset_calibration":
                self.calibration.reset()
                self.estimator.reset()
                self._clear_chain_calibration()
                return {
                    "type": "control",
                    "status": "calibration_reset",
                    "calibration": self.calibration.state().as_dict(),
                }
            raise ValueError("unknown control message")

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float | None = None,
    ) -> dict[str, object]:
        with self._lock:
            return self._process_jpeg_locked(payload, timestamp_s=timestamp_s)

    def _process_jpeg_locked(
        self,
        payload: bytes,
        *,
        timestamp_s: float | None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("payload is not a decodable colour JPEG")
        frame = self._fit_frame(frame)

        processor_timestamp_s = (
            time.monotonic() if timestamp_s is None else float(timestamp_s)
        )
        if not math.isfinite(processor_timestamp_s):
            raise ValueError("timestamp_s must be finite")
        detection = self.chain.process_frame(
            frame,
            timestamp_s=processor_timestamp_s,
        )
        observation, confidence = self._position_observation(detection)
        calibration_observation = observation
        calibration_confidence = confidence
        if (
            self.calibration.state().status == "collecting"
            and calibration_observation is None
            and detection.neck_locked
        ):
            calibration_observation = (
                detection.index_fret_raw
                if detection.index_fret_raw is not None
                else detection.index_fret
            )
            calibration_confidence = min(
                float(detection.homography_confidence),
                max(
                    float(detection.anchor.confidence),
                    float(detection.observation_confidence),
                ),
            )
        chain_calibration_was_active = self._chain_calibration_active
        calibration_completed = self.calibration.observe(
            calibration_observation,
            confidence=calibration_confidence,
            geometry_status=detection.geometry_status,
            timestamp_s=processor_timestamp_s,
        )
        if calibration_completed:
            self.estimator.reset()
            calibration_state = self.calibration.state()
            if calibration_state.status == "calibrated":
                scale, offset = self.calibration.transform()
                self._chain_calibration_active = self._set_chain_calibration(
                    scale=scale, offset=offset
                )
            elif calibration_state.status == "failed":
                self._clear_chain_calibration()
        adjusted_observation = (
            observation
            if chain_calibration_was_active
            else self.calibration.apply(observation)
        )
        estimate = self.estimator.update(
            index_fret=adjusted_observation,
            vision_confidence=confidence,
            timestamp_s=detection.timestamp_s,
        )
        hint_setter = getattr(self.chain, "set_hand_search_hint", None)
        if hint_setter is not None:
            hint_setter(build_hand_search_hint(detection, estimate))
        height, width = frame.shape[:2]
        guidance = assess_guidance(
            detection,
            estimate,
            frame_width=width,
            frame_height=height,
        )
        return {
            "type": "hud",
            "version": 2,
            "frame": {"width": width, "height": height},
            "detection": detection.as_dict(),
            "position": estimate.as_dict(),
            "guidance": guidance.as_dict(),
            "calibration": self.calibration.state().as_dict(),
            "server_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def _position_observation(
        self,
        detection: FrameDetection,
    ) -> tuple[float | None, float]:
        if detection.composite_available or detection.position_fret is not None:
            return detection.position_fret, detection.observation_confidence
        return (
            detection.index_fret,
            detection.anchor.confidence if detection.neck_locked else 0.0,
        )

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            self.max_frame_width / max(width, 1),
            self.max_frame_height / max(height, 1),
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("invalid frame dimensions")
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _set_chain_calibration(self, *, scale: float, offset: float) -> bool:
        setter = getattr(self.chain, "set_position_calibration", None)
        if setter is None:
            return False
        setter(scale=scale, offset=offset)
        return True

    def _clear_chain_calibration(self) -> None:
        setter = getattr(self.chain, "set_position_calibration", None)
        if setter is not None:
            setter(scale=1.0, offset=0.0)
        self._chain_calibration_active = False

    def close(self) -> None:
        with self._lock:
            self.chain.close()


__all__ = [
    "CalibrationState",
    "FrameProcessor",
    "FrameProcessorFactory",
    "HudFrameProcessor",
    "PositionCalibration",
]
