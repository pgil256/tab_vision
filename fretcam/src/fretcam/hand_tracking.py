"""Temporal hand-landmark tracking for FretCam's live camera path.

This module deliberately stays inside the quarantined FretCam package.  It
accepts detector-neutral 21-point hand observations, so the live extractor can
use MediaPipe's ``VIDEO`` running mode without changing TabVision's immutable
hand-backend contract.

The tracker combines:

* strictly increasing millisecond timestamps suitable for
  ``HandLandmarker.detect_for_video``;
* a per-joint One Euro filter;
* optional pyramidal Lucas-Kanade propagation between detector observations;
* per-finger anatomical validation; and
* short, quality-decaying retention through detector gaps.

No frames are retained beyond the immediately previous grayscale image.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

LANDMARK_COUNT = 21
WRIST_INDEX = 0
FINGER_LANDMARKS: Mapping[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FRETTING_FINGERS: tuple[str, ...] = ("index", "middle", "ring", "pinky")
PALM_INNOVATION_ANCHORS: tuple[int, ...] = (0, 5, 9, 13, 17)

TrackSource = Literal[
    "detector",
    "optical_flow",
    "held",
    "held_rejected",
    "expired",
]


def _readonly_copy(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    copied = np.asarray(array, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class LandmarkObservation:
    """One detector observation in image-pixel coordinates.

    ``landmarks_xy`` must follow MediaPipe's 21-landmark order. Missing joints
    may be represented by non-finite coordinates or zero ``joint_quality``.
    ``landmarks_z`` remains in the detector's relative-depth coordinate system.
    """

    landmarks_xy: np.ndarray
    landmarks_z: np.ndarray | None = None
    confidence: float = 1.0
    joint_quality: np.ndarray | None = None
    is_left_hand: bool | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.landmarks_xy, dtype=np.float64)
        if xy.shape != (LANDMARK_COUNT, 2):
            raise ValueError(
                f"landmarks_xy must have shape ({LANDMARK_COUNT}, 2), got {xy.shape}"
            )
        z = (
            np.zeros(LANDMARK_COUNT, dtype=np.float64)
            if self.landmarks_z is None
            else np.asarray(self.landmarks_z, dtype=np.float64)
        )
        if z.shape != (LANDMARK_COUNT,):
            raise ValueError(
                f"landmarks_z must have shape ({LANDMARK_COUNT},), got {z.shape}"
            )
        quality = (
            np.ones(LANDMARK_COUNT, dtype=np.float64)
            if self.joint_quality is None
            else np.asarray(self.joint_quality, dtype=np.float64)
        )
        if quality.shape != (LANDMARK_COUNT,):
            raise ValueError(
                "joint_quality must have shape "
                f"({LANDMARK_COUNT},), got {quality.shape}"
            )
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        quality = np.where(np.isfinite(quality), quality, 0.0)
        quality = np.clip(quality, 0.0, 1.0)
        object.__setattr__(self, "landmarks_xy", _readonly_copy(xy, dtype=np.float64))
        object.__setattr__(self, "landmarks_z", _readonly_copy(z, dtype=np.float64))
        object.__setattr__(
            self,
            "joint_quality",
            _readonly_copy(quality, dtype=np.float64),
        )

    @classmethod
    def from_mediapipe(
        cls,
        landmarks: Sequence[object],
        *,
        frame_width: int,
        frame_height: int,
        confidence: float,
        is_left_hand: bool | None = None,
    ) -> LandmarkObservation:
        """Convert MediaPipe-like normalised landmark objects without importing it."""
        if len(landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"expected {LANDMARK_COUNT} MediaPipe landmarks, got {len(landmarks)}"
            )
        xy = np.asarray(
            [
                (
                    float(getattr(point, "x")) * frame_width,
                    float(getattr(point, "y")) * frame_height,
                )
                for point in landmarks
            ],
            dtype=np.float64,
        )
        z = np.asarray(
            [float(getattr(point, "z", 0.0)) for point in landmarks],
            dtype=np.float64,
        )
        qualities = np.asarray(
            [
                float(
                    getattr(
                        point,
                        "visibility",
                        getattr(point, "presence", 1.0),
                    )
                )
                for point in landmarks
            ],
            dtype=np.float64,
        )
        return cls(
            landmarks_xy=xy,
            landmarks_z=z,
            confidence=confidence,
            joint_quality=qualities,
            is_left_hand=is_left_hand,
        )


@dataclass(frozen=True)
class OpticalFlowResult:
    """Pointwise result returned by an optical-flow implementation."""

    points_xy: np.ndarray
    status: np.ndarray
    error_px: np.ndarray


OpticalFlowFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    OpticalFlowResult,
]


@dataclass(frozen=True)
class FingerTrackingQuality:
    """Current evidence quality for one finger."""

    quality: float
    source: TrackSource
    detector_observed: bool
    retained: bool
    age_ms: float | None
    anatomical_score: float
    flow_inlier_ratio: float
    joint_quality: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedHandLandmarks:
    """Smoothed hand landmarks plus per-joint and per-finger provenance."""

    timestamp_s: float
    detector_timestamp_ms: int
    landmarks_xy: np.ndarray
    landmarks_z: np.ndarray
    joint_quality: np.ndarray
    finger_quality: Mapping[str, FingerTrackingQuality]
    is_left_hand: bool | None
    hand_quality: float
    detector_observation_age_ms: float | None
    used_optical_flow: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "landmarks_xy",
            _readonly_copy(self.landmarks_xy, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "landmarks_z",
            _readonly_copy(self.landmarks_z, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "joint_quality",
            _readonly_copy(self.joint_quality, dtype=np.float64),
        )
        object.__setattr__(self, "finger_quality", dict(self.finger_quality))

    def finger_axis_xy(self, finger: str) -> tuple[tuple[float, float], ...]:
        """Return finite MCP/PIP/DIP/tip points for ``finger``."""
        indices = FINGER_LANDMARKS.get(finger)
        if indices is None:
            raise KeyError(f"unknown finger: {finger}")
        points = self.landmarks_xy[np.asarray(indices, dtype=np.int64)]
        return tuple(
            (float(point[0]), float(point[1]))
            for point in points
            if np.all(np.isfinite(point))
        )

    @property
    def fretting_finger_quality(self) -> dict[str, float]:
        return {name: self.finger_quality[name].quality for name in FRETTING_FINGERS}


@dataclass(frozen=True)
class TemporalHandTrackerConfig:
    """Runtime-independent tracker thresholds."""

    min_cutoff_hz: float = 1.2
    speed_coefficient: float = 0.015
    derivative_cutoff_hz: float = 1.0
    max_occlusion_s: float = 0.18
    use_optical_flow: bool = True
    lk_window_size: int = 21
    lk_max_level: int = 2
    max_flow_error_px: float = 20.0
    flow_error_scale_px: float = 8.0
    max_flow_displacement_fraction: float = 0.20
    min_flow_joint_fraction: float = 0.75
    min_segment_palm_fraction: float = 0.015
    max_segment_palm_fraction: float = 1.20
    max_segment_imbalance: float = 5.0
    segment_reference_ratio_min: float = 0.45
    segment_reference_ratio_max: float = 2.20
    min_joint_angle_deg: float = 8.0
    max_joint_angle_delta_deg: float = 110.0
    anatomy_reference_alpha: float = 0.08
    max_finger_innovation_fraction: float = 0.90

    def __post_init__(self) -> None:
        if self.min_cutoff_hz <= 0.0:
            raise ValueError("min_cutoff_hz must be positive")
        if self.speed_coefficient < 0.0:
            raise ValueError("speed_coefficient cannot be negative")
        if self.derivative_cutoff_hz <= 0.0:
            raise ValueError("derivative_cutoff_hz must be positive")
        if self.max_occlusion_s <= 0.0:
            raise ValueError("max_occlusion_s must be positive")
        if self.lk_window_size < 3 or self.lk_window_size % 2 == 0:
            raise ValueError("lk_window_size must be an odd integer >= 3")
        if self.lk_max_level < 0:
            raise ValueError("lk_max_level cannot be negative")
        if self.max_flow_error_px <= 0.0 or self.flow_error_scale_px <= 0.0:
            raise ValueError("optical-flow error thresholds must be positive")
        if not 0.0 < self.max_flow_displacement_fraction <= 1.0:
            raise ValueError("max_flow_displacement_fraction must be in (0, 1]")
        if not 0.0 < self.min_flow_joint_fraction <= 1.0:
            raise ValueError("min_flow_joint_fraction must be in (0, 1]")
        if not 0.0 < self.segment_reference_ratio_min < 1.0:
            raise ValueError("segment_reference_ratio_min must be in (0, 1)")
        if self.segment_reference_ratio_max <= 1.0:
            raise ValueError("segment_reference_ratio_max must be > 1")
        if not 0.0 < self.anatomy_reference_alpha <= 1.0:
            raise ValueError("anatomy_reference_alpha must be in (0, 1]")
        if self.max_finger_innovation_fraction <= 0.0:
            raise ValueError("max_finger_innovation_fraction must be positive")


class VideoModeTimestampClock:
    """Map source seconds to strictly increasing MediaPipe VIDEO timestamps."""

    def __init__(self) -> None:
        self._last_timestamp_s: float | None = None
        self._last_timestamp_ms: int | None = None

    @property
    def last_timestamp_ms(self) -> int | None:
        return self._last_timestamp_ms

    @property
    def last_timestamp_s(self) -> float | None:
        return self._last_timestamp_s

    def next_ms(self, timestamp_s: float) -> int:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("timestamp_s must be strictly increasing")
        timestamp_ms = int(round(timestamp_s * 1000.0))
        if self._last_timestamp_ms is not None:
            timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_s = timestamp_s
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def reset(self) -> None:
        self._last_timestamp_s = None
        self._last_timestamp_ms = None


class _OneEuroVectorFilter:
    """Small vector-valued implementation of the One Euro filter."""

    def __init__(self, config: TemporalHandTrackerConfig) -> None:
        self._config = config
        self._last_timestamp_s: float | None = None
        self._last_raw: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray | float, dt_s: float) -> np.ndarray:
        cutoff = np.asarray(cutoff_hz, dtype=np.float64)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt_s)

    def filter(self, value: np.ndarray, timestamp_s: float) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            self._last_raw = raw.copy()
            self._filtered = raw.copy()
            self._derivative = np.zeros_like(raw)
            return raw.copy()

        assert self._last_raw is not None
        assert self._filtered is not None
        assert self._derivative is not None
        dt_s = timestamp_s - self._last_timestamp_s
        if dt_s <= 0.0:
            raise ValueError("filter timestamps must be strictly increasing")
        derivative = (raw - self._last_raw) / dt_s
        derivative_alpha = self._alpha(self._config.derivative_cutoff_hz, dt_s)
        filtered_derivative = (
            derivative_alpha * derivative + (1.0 - derivative_alpha) * self._derivative
        )
        cutoff = self._config.min_cutoff_hz + self._config.speed_coefficient * np.abs(
            filtered_derivative
        )
        value_alpha = self._alpha(cutoff, dt_s)
        filtered = value_alpha * raw + (1.0 - value_alpha) * self._filtered

        self._last_timestamp_s = timestamp_s
        self._last_raw = raw.copy()
        self._filtered = filtered.copy()
        self._derivative = filtered_derivative.copy()
        return filtered


class TemporalHandLandmarkTracker:
    """Track one selected hand across timestamped live frames.

    A caller typically:

    1. asks :meth:`next_video_timestamp_ms` for MediaPipe's VIDEO timestamp;
    2. calls ``detect_for_video`` when a detector refresh is due; and
    3. passes the resulting :class:`LandmarkObservation`, or ``None`` between
       detector observations, to :meth:`update`.

    Passing ``None`` does not imply a blank frame: LK flow is attempted first,
    then the last anatomically valid track is retained for at most
    ``max_occlusion_s``.
    """

    def __init__(
        self,
        config: TemporalHandTrackerConfig | None = None,
        *,
        optical_flow: OpticalFlowFunction | None = None,
    ) -> None:
        self.config = config or TemporalHandTrackerConfig()
        self.video_clock = VideoModeTimestampClock()
        self._optical_flow = optical_flow or self._default_optical_flow
        self._last_timestamp_s: float | None = None
        self._last_gray: np.ndarray | None = None
        self._landmarks_xyz: np.ndarray | None = None
        self._joint_quality = np.zeros(LANDMARK_COUNT, dtype=np.float64)
        self._detector_joint_quality = np.zeros(
            LANDMARK_COUNT,
            dtype=np.float64,
        )
        self._filters: list[_OneEuroVectorFilter | None] = [
            None for _ in range(LANDMARK_COUNT)
        ]
        self._finger_last_observed_s: dict[str, float] = {}
        self._finger_quality: dict[str, float] = {}
        self._finger_anatomy_score: dict[str, float] = {}
        self._segment_reference: dict[str, np.ndarray] = {}
        self._angle_reference: dict[str, np.ndarray] = {}
        self._is_left_hand: bool | None = None

    def next_video_timestamp_ms(self, timestamp_s: float) -> int:
        """Return a strictly increasing timestamp for ``detect_for_video``."""
        return self.video_clock.next_ms(timestamp_s)

    def reset(self) -> None:
        self.video_clock.reset()
        self._last_timestamp_s = None
        self._last_gray = None
        self._clear_track()

    def update(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float,
        observation: LandmarkObservation | None,
    ) -> TrackedHandLandmarks | None:
        """Advance the track by one BGR or grayscale frame."""
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("timestamp_s must be strictly increasing")
        gray = self._to_gray(frame)

        latest_observed = self._latest_fretting_observation_s()
        if (
            observation is not None
            and latest_observed is not None
            and timestamp_s - latest_observed > self.config.max_occlusion_s
        ):
            self._clear_track()

        if observation is not None:
            output = self._update_from_detector(
                observation,
                timestamp_s=timestamp_s,
            )
        else:
            output = self._update_without_detector(
                gray,
                timestamp_s=timestamp_s,
            )

        self._last_gray = gray.copy()
        self._last_timestamp_s = timestamp_s
        if output is None:
            return None
        return self._snapshot(
            timestamp_s=timestamp_s,
            finger_sources=output[0],
            observed_fingers=output[1],
            flow_ratios=output[2],
            used_optical_flow=output[3],
        )

    def _update_from_detector(
        self,
        observation: LandmarkObservation,
        *,
        timestamp_s: float,
    ) -> (
        tuple[
            dict[str, TrackSource],
            set[str],
            dict[str, float],
            bool,
        ]
        | None
    ):
        candidate_xyz = np.column_stack(
            (observation.landmarks_xy, observation.landmarks_z)
        )
        previous_xyz = (
            None if self._landmarks_xyz is None else self._landmarks_xyz.copy()
        )
        if self._landmarks_xyz is None:
            self._landmarks_xyz = np.full(
                (LANDMARK_COUNT, 3),
                np.nan,
                dtype=np.float64,
            )

        palm_scale = self._palm_scale(observation.landmarks_xy)
        common_translation = self._common_motion_translation(
            observation.landmarks_xy,
            previous_xyz,
        )
        finger_sources: dict[str, TrackSource] = {}
        observed_fingers: set[str] = set()
        flow_ratios = {name: 0.0 for name in FINGER_LANDMARKS}

        wrist_finite = np.all(np.isfinite(candidate_xyz[WRIST_INDEX]))
        wrist_quality = float(observation.joint_quality[WRIST_INDEX])
        if wrist_finite and wrist_quality > 0.0:
            self._set_filtered_joint(
                WRIST_INDEX,
                candidate_xyz[WRIST_INDEX],
                timestamp_s=timestamp_s,
            )
            self._joint_quality[WRIST_INDEX] = wrist_quality * observation.confidence
            self._detector_joint_quality[WRIST_INDEX] = self._joint_quality[WRIST_INDEX]

        for name, indices in FINGER_LANDMARKS.items():
            index_array = np.asarray(indices, dtype=np.int64)
            points_xy = observation.landmarks_xy[index_array]
            points_finite = np.all(np.isfinite(candidate_xyz[index_array]), axis=1)
            input_quality = np.asarray(observation.joint_quality[index_array])
            complete = bool(np.all(points_finite & (input_quality > 0.0)))
            accepted = False
            anatomy_score = 0.0
            normalized_lengths = np.empty(0, dtype=np.float64)
            angles = np.empty(0, dtype=np.float64)
            if complete:
                (
                    accepted,
                    anatomy_score,
                    normalized_lengths,
                    angles,
                ) = self._validate_finger(
                    name,
                    points_xy,
                    palm_scale=palm_scale,
                )
                if accepted and not self._finger_innovation_is_valid(
                    name,
                    points_xy,
                    previous_xyz=previous_xyz,
                    common_translation=common_translation,
                    palm_scale=palm_scale,
                ):
                    accepted = False
                    anatomy_score = 0.0

            if accepted:
                for joint_index in indices:
                    self._set_filtered_joint(
                        joint_index,
                        candidate_xyz[joint_index],
                        timestamp_s=timestamp_s,
                    )
                quality_values = input_quality * observation.confidence * anatomy_score
                self._joint_quality[index_array] = quality_values
                self._detector_joint_quality[index_array] = quality_values
                finger_quality = float(np.mean(quality_values))
                self._finger_quality[name] = finger_quality
                self._finger_anatomy_score[name] = anatomy_score
                self._finger_last_observed_s[name] = timestamp_s
                self._update_anatomy_reference(
                    name,
                    normalized_lengths,
                    angles,
                )
                finger_sources[name] = "detector"
                observed_fingers.add(name)
            else:
                source = "held_rejected" if complete else "held"
                retained = self._retain_finger(
                    name,
                    timestamp_s=timestamp_s,
                    source_penalty=0.60 if complete else 0.78,
                )
                finger_sources[name] = source if retained else "expired"

        if observation.is_left_hand is not None and observed_fingers:
            self._is_left_hand = observation.is_left_hand
        if not self._has_fretting_track():
            self._clear_track()
            return None
        return finger_sources, observed_fingers, flow_ratios, False

    def _update_without_detector(
        self,
        gray: np.ndarray,
        *,
        timestamp_s: float,
    ) -> (
        tuple[
            dict[str, TrackSource],
            set[str],
            dict[str, float],
            bool,
        ]
        | None
    ):
        if self._landmarks_xyz is None:
            return None

        previous_xyz = self._landmarks_xyz.copy()
        propagated = self._propagate_flow(gray, timestamp_s=timestamp_s)
        flow_points: np.ndarray | None = None
        flow_valid = np.zeros(LANDMARK_COUNT, dtype=bool)
        flow_quality = np.zeros(LANDMARK_COUNT, dtype=np.float64)
        if propagated is not None:
            flow_points, flow_valid, flow_quality = propagated

        finger_sources: dict[str, TrackSource] = {}
        flow_ratios: dict[str, float] = {}
        used_optical_flow = False
        palm_scale = (
            self._palm_scale(flow_points)
            if flow_points is not None
            else self._palm_scale(self._landmarks_xyz[:, :2])
        )
        common_translation = (
            self._common_motion_translation(flow_points, previous_xyz)
            if flow_points is not None
            else None
        )

        for name, indices in FINGER_LANDMARKS.items():
            age_s = self._finger_age_s(name, timestamp_s)
            if age_s is None or age_s > self.config.max_occlusion_s:
                self._expire_finger(name)
                finger_sources[name] = "expired"
                flow_ratios[name] = 0.0
                continue

            index_array = np.asarray(indices, dtype=np.int64)
            ratio = float(np.mean(flow_valid[index_array]))
            flow_ratios[name] = ratio
            tip_valid = bool(flow_valid[indices[-1]])
            can_use_flow = (
                flow_points is not None
                and tip_valid
                and ratio >= self.config.min_flow_joint_fraction
            )
            rejected_flow = False
            anatomy_score = 0.0
            if can_use_flow:
                (
                    accepted,
                    anatomy_score,
                    _normalized_lengths,
                    _angles,
                ) = self._validate_finger(
                    name,
                    flow_points[index_array],
                    palm_scale=palm_scale,
                )
                can_use_flow = accepted
                rejected_flow = not accepted
                if can_use_flow and not self._finger_innovation_is_valid(
                    name,
                    flow_points[index_array],
                    previous_xyz=previous_xyz,
                    common_translation=common_translation,
                    palm_scale=palm_scale,
                ):
                    can_use_flow = False
                    rejected_flow = True

            if can_use_flow:
                assert flow_points is not None
                for joint_index in indices:
                    if flow_valid[joint_index]:
                        xyz = self._landmarks_xyz[joint_index].copy()
                        xyz[:2] = flow_points[joint_index]
                        self._set_filtered_joint(
                            joint_index,
                            xyz,
                            timestamp_s=timestamp_s,
                        )
                freshness = self._retention_factor(age_s)
                point_quality = (
                    self._detector_joint_quality[index_array]
                    * freshness
                    * np.maximum(flow_quality[index_array], 0.15)
                    * anatomy_score
                )
                self._joint_quality[index_array] = point_quality
                self._finger_quality[name] = float(np.mean(point_quality))
                self._finger_anatomy_score[name] = anatomy_score
                finger_sources[name] = "optical_flow"
                used_optical_flow = True
            else:
                retained = self._retain_finger(
                    name,
                    timestamp_s=timestamp_s,
                    source_penalty=0.78,
                )
                if retained:
                    finger_sources[name] = "held_rejected" if rejected_flow else "held"
                else:
                    finger_sources[name] = "expired"

        if not self._has_fretting_track():
            self._clear_track()
            return None
        return finger_sources, set(), flow_ratios, used_optical_flow

    def _snapshot(
        self,
        *,
        timestamp_s: float,
        finger_sources: Mapping[str, TrackSource],
        observed_fingers: set[str],
        flow_ratios: Mapping[str, float],
        used_optical_flow: bool,
    ) -> TrackedHandLandmarks:
        assert self._landmarks_xyz is not None
        qualities: dict[str, FingerTrackingQuality] = {}
        for name, indices in FINGER_LANDMARKS.items():
            age_s = self._finger_age_s(name, timestamp_s)
            age_ms = None if age_s is None else age_s * 1000.0
            joint_quality = tuple(
                float(self._joint_quality[index]) for index in indices
            )
            quality = float(np.mean(joint_quality))
            source = finger_sources.get(name, "expired")
            qualities[name] = FingerTrackingQuality(
                quality=float(np.clip(quality, 0.0, 1.0)),
                source=source,
                detector_observed=name in observed_fingers,
                retained=source != "expired",
                age_ms=age_ms,
                anatomical_score=float(
                    np.clip(self._finger_anatomy_score.get(name, 0.0), 0.0, 1.0)
                ),
                flow_inlier_ratio=float(np.clip(flow_ratios.get(name, 0.0), 0.0, 1.0)),
                joint_quality=joint_quality,
            )

        fretting_quality = [
            qualities[name].quality
            for name in FRETTING_FINGERS
            if qualities[name].retained
        ]
        hand_quality = float(np.mean(fretting_quality)) if fretting_quality else 0.0
        latest_observed = self._latest_fretting_observation_s()
        observation_age_ms = (
            None
            if latest_observed is None
            else max(0.0, timestamp_s - latest_observed) * 1000.0
        )
        detector_timestamp_ms = (
            self.video_clock.last_timestamp_ms
            if self.video_clock.last_timestamp_s == timestamp_s
            and self.video_clock.last_timestamp_ms is not None
            else int(round(timestamp_s * 1000.0))
        )
        return TrackedHandLandmarks(
            timestamp_s=timestamp_s,
            detector_timestamp_ms=detector_timestamp_ms,
            landmarks_xy=self._landmarks_xyz[:, :2],
            landmarks_z=self._landmarks_xyz[:, 2],
            joint_quality=self._joint_quality,
            finger_quality=qualities,
            is_left_hand=self._is_left_hand,
            hand_quality=float(np.clip(hand_quality, 0.0, 1.0)),
            detector_observation_age_ms=observation_age_ms,
            used_optical_flow=used_optical_flow,
        )

    def _set_filtered_joint(
        self,
        joint_index: int,
        value: np.ndarray,
        *,
        timestamp_s: float,
    ) -> None:
        point = np.asarray(value, dtype=np.float64)
        if self._filters[joint_index] is None:
            self._filters[joint_index] = _OneEuroVectorFilter(self.config)
        filtered = self._filters[joint_index].filter(point, timestamp_s)
        assert self._landmarks_xyz is not None
        self._landmarks_xyz[joint_index] = filtered

    def _retain_finger(
        self,
        name: str,
        *,
        timestamp_s: float,
        source_penalty: float,
    ) -> bool:
        age_s = self._finger_age_s(name, timestamp_s)
        if age_s is None or age_s > self.config.max_occlusion_s:
            self._expire_finger(name)
            return False
        indices = np.asarray(FINGER_LANDMARKS[name], dtype=np.int64)
        freshness = self._retention_factor(age_s)
        self._joint_quality[indices] = (
            self._detector_joint_quality[indices] * freshness * source_penalty
        )
        self._finger_quality[name] = float(np.mean(self._joint_quality[indices]))
        return bool(np.any(np.isfinite(self._landmarks_xyz[indices, :2])))

    def _retention_factor(self, age_s: float) -> float:
        scale = max(self.config.max_occlusion_s * 0.75, 1e-6)
        return float(np.exp(-max(age_s, 0.0) / scale))

    def _propagate_flow(
        self,
        gray: np.ndarray,
        *,
        timestamp_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if (
            not self.config.use_optical_flow
            or self._last_gray is None
            or self._landmarks_xyz is None
        ):
            return None
        points = self._landmarks_xyz[:, :2]
        finite_indices = np.flatnonzero(np.all(np.isfinite(points), axis=1))
        if finite_indices.size == 0:
            return None
        previous_points = points[finite_indices]
        try:
            result = self._optical_flow(
                self._last_gray,
                gray,
                previous_points,
            )
        except (ValueError, RuntimeError):
            return None
        next_points = np.asarray(result.points_xy, dtype=np.float64)
        status = np.asarray(result.status, dtype=bool).reshape(-1)
        errors = np.asarray(result.error_px, dtype=np.float64).reshape(-1)
        expected_shape = (finite_indices.size, 2)
        if (
            next_points.shape != expected_shape
            or status.shape != (finite_indices.size,)
            or errors.shape != (finite_indices.size,)
        ):
            return None

        height, width = gray.shape[:2]
        diagonal = float(np.hypot(width, height))
        displacements = np.linalg.norm(next_points - previous_points, axis=1)
        valid = (
            status
            & np.all(np.isfinite(next_points), axis=1)
            & np.isfinite(errors)
            & (errors <= self.config.max_flow_error_px)
            & (
                displacements
                <= self.config.max_flow_displacement_fraction * max(diagonal, 1.0)
            )
            & (next_points[:, 0] >= 0.0)
            & (next_points[:, 0] < width)
            & (next_points[:, 1] >= 0.0)
            & (next_points[:, 1] < height)
        )
        if not np.any(valid):
            return None
        propagated = points.copy()
        propagated[finite_indices[valid]] = next_points[valid]
        valid_full = np.zeros(LANDMARK_COUNT, dtype=bool)
        valid_full[finite_indices] = valid
        quality_full = np.zeros(LANDMARK_COUNT, dtype=np.float64)
        quality_full[finite_indices[valid]] = np.exp(
            -errors[valid] / self.config.flow_error_scale_px
        )
        return propagated, valid_full, quality_full

    def _validate_finger(
        self,
        name: str,
        points_xy: np.ndarray,
        *,
        palm_scale: float,
    ) -> tuple[bool, float, np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return False, 0.0, np.empty(0), np.empty(0)
        if not math.isfinite(palm_scale) or palm_scale <= 1e-6:
            return False, 0.0, np.empty(0), np.empty(0)

        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        normalized = segments / palm_scale
        if (
            np.any(~np.isfinite(normalized))
            or np.any(normalized < self.config.min_segment_palm_fraction)
            or np.any(normalized > self.config.max_segment_palm_fraction)
        ):
            return False, 0.0, normalized, np.empty(0)
        smallest = max(float(np.min(segments)), 1e-9)
        if float(np.max(segments)) / smallest > self.config.max_segment_imbalance:
            return False, 0.0, normalized, np.empty(0)

        angles = np.asarray(
            [
                self._joint_angle(points[0], points[1], points[2]),
                self._joint_angle(points[1], points[2], points[3]),
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(angles)) or np.any(
            angles < self.config.min_joint_angle_deg
        ):
            return False, 0.0, normalized, angles

        length_score = 1.0
        reference = self._segment_reference.get(name)
        if reference is not None:
            ratios = normalized / np.maximum(reference, 1e-9)
            if np.any(ratios < self.config.segment_reference_ratio_min) or np.any(
                ratios > self.config.segment_reference_ratio_max
            ):
                return False, 0.0, normalized, angles
            length_score = float(np.exp(-np.mean(np.abs(np.log(ratios)))))

        angle_score = float(
            np.clip(
                np.min(angles) / max(45.0, self.config.min_joint_angle_deg),
                0.0,
                1.0,
            )
        )
        angle_reference = self._angle_reference.get(name)
        if angle_reference is not None:
            angle_delta = np.abs(angles - angle_reference)
            if np.any(angle_delta > self.config.max_joint_angle_delta_deg):
                return False, 0.0, normalized, angles
            angle_score *= float(
                np.exp(
                    -np.mean(angle_delta)
                    / max(self.config.max_joint_angle_delta_deg, 1e-6)
                )
            )

        balance_score = float(
            np.clip(
                self.config.max_segment_imbalance
                / max(float(np.max(segments)) / smallest, 1.0),
                0.0,
                1.0,
            )
        )
        score = float(
            np.clip(
                0.50 * length_score + 0.35 * angle_score + 0.15 * balance_score,
                0.0,
                1.0,
            )
        )
        return True, score, normalized, angles

    @staticmethod
    def _common_motion_translation(
        candidate_xy: np.ndarray,
        previous_xyz: np.ndarray | None,
    ) -> np.ndarray | None:
        """Estimate coherent hand motion without trusting any single finger."""
        if previous_xyz is None:
            return None
        previous_xy = np.asarray(previous_xyz, dtype=np.float64)[:, :2]
        candidate = np.asarray(candidate_xy, dtype=np.float64)
        indices = np.asarray(PALM_INNOVATION_ANCHORS, dtype=np.int64)
        valid = np.all(np.isfinite(previous_xy[indices]), axis=1) & np.all(
            np.isfinite(candidate[indices]),
            axis=1,
        )
        if int(np.sum(valid)) < 3:
            return None
        return np.median(
            candidate[indices[valid]] - previous_xy[indices[valid]],
            axis=0,
        )

    def _finger_innovation_is_valid(
        self,
        name: str,
        points_xy: np.ndarray,
        *,
        previous_xyz: np.ndarray | None,
        common_translation: np.ndarray | None,
        palm_scale: float,
    ) -> bool:
        """Reject an isolated finger jump while allowing coherent hand shifts."""
        if previous_xyz is None or common_translation is None:
            return True
        indices = np.asarray(FINGER_LANDMARKS[name], dtype=np.int64)
        previous_points = np.asarray(previous_xyz, dtype=np.float64)[indices, :2]
        candidate_points = np.asarray(points_xy, dtype=np.float64)
        valid = np.all(np.isfinite(previous_points), axis=1) & np.all(
            np.isfinite(candidate_points),
            axis=1,
        )
        if int(np.sum(valid)) < 2:
            return True
        previous_scale = self._palm_scale(
            np.asarray(previous_xyz, dtype=np.float64)[:, :2]
        )
        scales = [
            scale
            for scale in (palm_scale, previous_scale)
            if math.isfinite(scale) and scale > 1e-6
        ]
        if not scales:
            return True
        residual = (
            candidate_points[valid]
            - previous_points[valid]
            - np.asarray(common_translation, dtype=np.float64)
        )
        maximum_innovation = float(np.max(np.linalg.norm(residual, axis=1)))
        return maximum_innovation <= (
            self.config.max_finger_innovation_fraction * max(scales)
        )

    def _update_anatomy_reference(
        self,
        name: str,
        normalized_lengths: np.ndarray,
        angles: np.ndarray,
    ) -> None:
        alpha = self.config.anatomy_reference_alpha
        old_lengths = self._segment_reference.get(name)
        old_angles = self._angle_reference.get(name)
        self._segment_reference[name] = (
            normalized_lengths.copy()
            if old_lengths is None
            else (1.0 - alpha) * old_lengths + alpha * normalized_lengths
        )
        self._angle_reference[name] = (
            angles.copy()
            if old_angles is None
            else (1.0 - alpha) * old_angles + alpha * angles
        )

    @staticmethod
    def _joint_angle(
        previous: np.ndarray,
        joint: np.ndarray,
        following: np.ndarray,
    ) -> float:
        first = previous - joint
        second = following - joint
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-9:
            return math.nan
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _palm_scale(points_xy: np.ndarray) -> float:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape != (LANDMARK_COUNT, 2):
            return math.nan
        candidate_pairs = ((WRIST_INDEX, 9), (5, 17), (WRIST_INDEX, 5))
        distances = [
            float(np.linalg.norm(points[a] - points[b]))
            for a, b in candidate_pairs
            if np.all(np.isfinite(points[[a, b]]))
        ]
        usable = [distance for distance in distances if distance > 1e-6]
        return float(np.median(usable)) if usable else math.nan

    def _finger_age_s(self, name: str, timestamp_s: float) -> float | None:
        observed_s = self._finger_last_observed_s.get(name)
        return None if observed_s is None else max(0.0, timestamp_s - observed_s)

    def _latest_fretting_observation_s(self) -> float | None:
        values = [
            self._finger_last_observed_s[name]
            for name in FRETTING_FINGERS
            if name in self._finger_last_observed_s
        ]
        return max(values) if values else None

    def _expire_finger(self, name: str) -> None:
        if self._landmarks_xyz is None:
            return
        indices = np.asarray(FINGER_LANDMARKS[name], dtype=np.int64)
        self._landmarks_xyz[indices] = np.nan
        self._joint_quality[indices] = 0.0
        self._detector_joint_quality[indices] = 0.0
        self._finger_quality[name] = 0.0
        self._finger_anatomy_score[name] = 0.0

    def _has_fretting_track(self) -> bool:
        if self._landmarks_xyz is None:
            return False
        for name in FRETTING_FINGERS:
            tip_index = FINGER_LANDMARKS[name][-1]
            if np.all(np.isfinite(self._landmarks_xyz[tip_index, :2])):
                return True
        return False

    def _clear_track(self) -> None:
        self._landmarks_xyz = None
        self._joint_quality = np.zeros(LANDMARK_COUNT, dtype=np.float64)
        self._detector_joint_quality = np.zeros(
            LANDMARK_COUNT,
            dtype=np.float64,
        )
        self._filters = [None for _ in range(LANDMARK_COUNT)]
        self._finger_last_observed_s.clear()
        self._finger_quality.clear()
        self._finger_anatomy_score.clear()
        self._segment_reference.clear()
        self._angle_reference.clear()
        self._is_left_hand = None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim == 2:
            if image.size == 0:
                raise ValueError("frame cannot be empty")
            return image.astype(np.uint8, copy=False)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected grayscale or BGR frame, got shape {image.shape}"
            )
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("frame cannot be empty")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("opencv-python is required for hand tracking") from exc
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _default_optical_flow(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        points_xy: np.ndarray,
    ) -> OpticalFlowResult:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("opencv-python is required for optical flow") from exc
        try:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2),
                None,
                winSize=(self.config.lk_window_size, self.config.lk_window_size),
                maxLevel=self.config.lk_max_level,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.03,
                ),
            )
        except cv2.error:
            next_points = status = error = None
        count = len(points_xy)
        if next_points is None or status is None or error is None:
            return OpticalFlowResult(
                points_xy=np.asarray(points_xy, dtype=np.float64),
                status=np.zeros(count, dtype=bool),
                error_px=np.full(count, np.inf, dtype=np.float64),
            )
        return OpticalFlowResult(
            points_xy=next_points.reshape(-1, 2),
            status=status.reshape(-1).astype(bool),
            error_px=error.reshape(-1),
        )


__all__ = [
    "FINGER_LANDMARKS",
    "FRETTING_FINGERS",
    "FingerTrackingQuality",
    "LandmarkObservation",
    "OpticalFlowFunction",
    "OpticalFlowResult",
    "TemporalHandLandmarkTracker",
    "TemporalHandTrackerConfig",
    "TrackedHandLandmarks",
    "VideoModeTimestampClock",
]
