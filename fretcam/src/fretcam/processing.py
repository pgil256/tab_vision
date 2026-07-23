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

from fretcam.detection import DetectionChain, FrameDetection
from fretcam.guidance import assess_guidance
from fretcam.position import PositionEstimator


class FrameProcessor(Protocol):
    def warmup(self) -> None: ...

    def reset(self) -> None: ...

    def process_jpeg(self, payload: bytes) -> dict[str, object]: ...

    def handle_control(self, message: dict[str, object]) -> dict[str, object]: ...

    def close(self) -> None: ...


FrameProcessorFactory = Callable[[], FrameProcessor]


@dataclass(frozen=True)
class CalibrationState:
    status: str = "idle"
    target_position: int = 1
    offset_fret: float = 0.0
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
        self.reset()

    def reset(self) -> None:
        self._status = "idle"
        self._target = 1
        self._offset = 0.0
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
            offset = float(self._target) - median
            if mad > self.max_mad_fret:
                self._fail("Calibration was unstable; hold one position steadily.")
            elif abs(offset) > self.max_offset_fret:
                self._fail("Calibration offset was implausible; reframe the neck.")
            else:
                self._offset = offset
                self._status = "calibrated"
                self._message = (
                    f"Calibrated with {offset:+.2f} fret residual for this session."
                )
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
        return value + self._offset

    def state(self) -> CalibrationState:
        return CalibrationState(
            status=self._status,
            target_position=self._target,
            offset_fret=round(self._offset, 3),
            samples=len(self._values),
            message=self._message,
        )

    def _fail(self, message: str) -> None:
        self._status = "failed"
        self._offset = 0.0
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

    def warmup(self) -> None:
        """Pay one-time model initialization before the server reports ready."""
        with self._lock:
            self.chain.process_frame(
                np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=0.0
            )
            waiter = getattr(self.chain, "wait_for_background_detector", None)
            if waiter is not None:
                waiter()
            self._reset_tracking()
            self.estimator.reset()
            self.calibration.reset()

    def reset(self) -> None:
        with self._lock:
            self._reset_tracking()
            self.estimator.reset()
            self.calibration.reset()

    def _reset_tracking(self) -> None:
        resetter = getattr(self.chain, "reset_tracking", None)
        if resetter is None:
            self.chain.reset()
        else:
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
                return {
                    "type": "control",
                    "status": "settings_applied",
                    "player_handedness": handedness,
                }
            if message_type == "calibrate":
                self.calibration.start(target_position=1)
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "calibration_started",
                    "calibration": self.calibration.state().as_dict(),
                }
            if message_type == "reset_calibration":
                self.calibration.reset()
                self.estimator.reset()
                return {
                    "type": "control",
                    "status": "calibration_reset",
                    "calibration": self.calibration.state().as_dict(),
                }
            raise ValueError("unknown control message")

    def process_jpeg(self, payload: bytes) -> dict[str, object]:
        with self._lock:
            return self._process_jpeg_locked(payload)

    def _process_jpeg_locked(self, payload: bytes) -> dict[str, object]:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("payload is not a decodable colour JPEG")
        frame = self._fit_frame(frame)

        timestamp_s = time.monotonic()
        detection = self.chain.process_frame(frame, timestamp_s=timestamp_s)
        observation, confidence = self._position_observation(detection)
        calibration_completed = self.calibration.observe(
            observation,
            confidence=confidence,
            geometry_status=detection.geometry_status,
            timestamp_s=timestamp_s,
        )
        if calibration_completed:
            self.estimator.reset()
        adjusted_observation = self.calibration.apply(observation)
        estimate = self.estimator.update(
            index_fret=adjusted_observation,
            vision_confidence=confidence,
            timestamp_s=detection.timestamp_s,
        )
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
