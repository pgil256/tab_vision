"""F2 live-frame detection chain built from TabVision's vision library."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Protocol

import numpy as np

from fretcam.geometry_refinement import (
    GeometryRefinement,
    nearest_string,
    refine_live_geometry,
    transform_string_curves,
)
from fretcam.hand_tracking import (
    FINGER_LANDMARKS as TRACKED_FINGER_LANDMARKS,
    LandmarkObservation,
    TemporalHandLandmarkTracker,
    TrackedHandLandmarks,
)
from fretcam.tracking import (
    OpticalBoardTracker,
    TrackingSnapshot,
    align_detection_homography,
)
from tabvision.errors import BackendError
from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import (
    RULE_OF_18_RATIO,
    calibrate_board,
    project_to_canonical,
)
from tabvision.video.guitar.yolo_backend import OBBPredictions, YoloOBBBackend
from tabvision.video.hand.fingertip_to_fret import (
    FRETTING_FINGERS,
    FingerSample,
    HandSample,
)
from tabvision.video.hand.mediapipe_backend import (
    MediaPipeHandBackend,
    _build_hand_sample,
    _select_fretting_hand,
)
from tabvision.video.hand.neck_anchor import HandNeckAnchor

Point = tuple[float, float]
BoardCalibrator = Callable[
    [OBBPredictions, GuitarConfig], tuple[Homography, np.ndarray | None]
]

# The keypoint homography's unit neck runs from the nut to the body joint. The
# existing overlay and Phase-3 eval convention define that endpoint as fret 12.
# A fitted fret map supersedes this fallback whenever the detector sees enough
# wires to establish a per-frame nonlinear coordinate.
FALLBACK_BODY_JOINT_FRET = 12

# A position observation is only eligible for temporal locking when its source
# landmark lies on the detected canonical neck.  Keep this strict: tolerance at
# the image boundary would turn a picking hand beside the soundhole back into a
# plausible fret observation.
CANONICAL_NECK_MIN = 0.0
CANONICAL_NECK_MAX = 1.0
MIN_ON_NECK_FINGERTIPS = 3
LONGITUDINAL_WRIST_EDGE_ZONE = 0.15
LONGITUDINAL_WRIST_OUTWARD_MARGIN = 0.15
ROI_CROP_COOLDOWN_FRAMES = 10
HAND_SEARCH_UPSCALE_MIN_LONG_EDGE = 512
HAND_SEARCH_UPSCALE_MAX_SCALE = 2.5
GEOMETRY_REFINE_INTERVAL_S = 0.20
FULL_HAND_VIDEO_REFRESH_INTERVAL_S = 0.50
LOCKED_HAND_DETECTOR_HZ = 5.0
ACTIVE_HAND_DETECTOR_HZ = 15.0
LOW_HAND_TRACKING_QUALITY = 0.35
DETECTOR_REFRESH_MAX_RMS_FRACTION = 0.06
DETECTOR_REFRESH_MAX_CORNER_FRACTION = 0.10
LIVE_FRET_AXIS_MIN_FRET_SUPPORT = 0.12
LIVE_FRET_AXIS_MIN_STRING_SUPPORT = 0.08
LIVE_FRET_AXIS_MIN_FINITE_STRINGS = 4
LIVE_FRET_AXIS_CONFIRMATIONS = 2
LIVE_FRET_AXIS_MAX_LOWER_WIRE_CELL_FRACTION = 0.75
FRET_WIRE_DEADBAND_FRACTION = 0.35
BARRE_INDEX_MIN_EXTENSION = 0.85
BARRE_MIN_CROSS_NECK_SPAN = 0.70
PARTIAL_BARRE_MIN_CROSS_NECK_SPAN = 0.34
BARRE_MIN_CROSS_TO_ALONG_RATIO = 3.0
INDEX_AXIS_LANDMARKS = (5, 6, 7, 8)
FINGER_AXIS_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_POSITION_OFFSETS = {
    "index": 0.0,
    "middle": 1.0,
    "ring": 2.0,
    "pinky": 3.0,
}
FINGER_BASE_WEIGHTS = {
    "index": 1.0,
    "middle": 0.90,
    "ring": 0.82,
    "pinky": 0.74,
}
MIN_POSITION_OBSERVATION_CONFIDENCE = 0.20


class Detector(Protocol):
    def predict_all(self, frame: np.ndarray) -> OBBPredictions: ...


@dataclass(frozen=True)
class HandObservation:
    """FretCam-only hand sample plus richer per-finger landmark evidence."""

    hand: HandSample
    finger_axes_xy: dict[str, tuple[Point, ...]]
    finger_quality: dict[str, float] = field(default_factory=dict)
    handedness_label: str = ""
    handedness_score: float = 0.0
    source: str = "detector"
    landmarks_xy: tuple[Point, ...] = ()
    landmarks_z: tuple[float, ...] = ()
    joint_quality: tuple[float, ...] = ()
    finger_stillness: dict[str, float] = field(default_factory=dict)
    pose_quality: float = 0.0
    pose_continuity: float = 0.0
    pose_identity_score: float = 0.0
    pose_residual_fraction: float = 1.0
    pose_predicted: bool = False
    detector_observation_accepted: bool = False

    @property
    def index_axis_xy(self) -> tuple[Point, ...]:
        return self.finger_axes_xy.get("index", ())


class HandExtractor(Protocol):
    def extract(self, frame: np.ndarray) -> HandObservation | HandSample | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HandSearchHint:
    """One-frame-delayed estimator feedback for adaptive hand acquisition."""

    position_state: str
    position_reason: str
    observation_confidence: float
    landmark_quality: float
    blockers: tuple[str, ...]
    hand_visible: bool
    pose_quality: float = 0.0


class MediaPipeHandExtractor:
    """Retain the shared hand sample plus FretCam-only finger joint axes.

    TabVision's public ``detect_anchor`` intentionally returns only the coarse
    anchor. FretCam needs the hand marker and the index MCP/PIP/DIP/tip line so
    an extended barre finger is not represented by its tip alone. This adapter
    reuses the backend's loaded MediaPipe model and sample-building helpers; the
    richer observation remains quarantined here and does not change §8.
    """

    def __init__(
        self,
        backend: MediaPipeHandBackend | None = None,
        *,
        player_handedness: str = "right",
    ) -> None:
        if player_handedness not in {"right", "left"}:
            raise ValueError("player_handedness must be 'right' or 'left'")
        self.backend = backend or MediaPipeHandBackend()
        self.player_handedness = player_handedness
        self._video_landmarker = None
        self._last_video_timestamp_ms = -1

    def set_player_handedness(self, value: str) -> None:
        if value not in {"right", "left"}:
            raise ValueError("player_handedness must be 'right' or 'left'")
        self.player_handedness = value

    def extract(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float | None = None,
    ) -> HandObservation | None:
        candidates = self.extract_candidates(frame, timestamp_s=timestamp_s)
        if not candidates:
            return None
        selected = self._select_observation(candidates)
        return candidates[selected]

    def extract_candidates(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float | None = None,
        use_video: bool = True,
    ) -> tuple[HandObservation, ...]:
        """Return every detected hand so neck geometry can choose the fretting one.

        The shared TabVision backend intentionally retains its IMAGE-mode API.
        FretCam owns a timestamped VIDEO-mode landmarker so this live side quest
        gains MediaPipe's temporal detector/tracker without changing a section-8
        contract or any default TabVision behavior.
        """
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise BackendError(
                "opencv-python and mediapipe are required. Install with: "
                "pip install '.[vision]'."
            ) from exc

        landmarker = (
            self._load_video_landmarker() if use_video else self.backend._load()
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if use_video and hasattr(landmarker, "detect_for_video"):
            requested_ms = int(
                round(
                    1000.0
                    * (
                        time.monotonic()
                        if timestamp_s is None or not math.isfinite(timestamp_s)
                        else float(timestamp_s)
                    )
                )
            )
            timestamp_ms = max(self._last_video_timestamp_ms + 1, requested_ms)
            self._last_video_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            # Test doubles and older MediaPipe builds retain the IMAGE surface.
            result = landmarker.detect(mp_image)
        if not result.hand_landmarks or not result.handedness:
            return ()

        height, width = frame.shape[:2]
        observations: list[HandObservation] = []
        for landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness,
            strict=False,
        ):
            hand = _build_hand_sample(
                landmarks,
                handedness,
                frame_width=width,
                frame_height=height,
            )
            finger_axes = {
                name: tuple(
                    (
                        float(landmarks[index].x) * width,
                        float(landmarks[index].y) * height,
                    )
                    for index in indices
                )
                for name, indices in FINGER_AXIS_LANDMARKS.items()
            }
            category = handedness[0] if handedness else None
            label = str(getattr(category, "category_name", ""))
            handedness_score = float(getattr(category, "score", hand.confidence))
            finger_quality = {
                name: self._finger_landmark_quality(
                    landmarks,
                    indices,
                )
                for name, indices in FINGER_AXIS_LANDMARKS.items()
            }
            joint_quality = tuple(
                self._joint_landmark_quality(
                    landmark,
                )
                for landmark in landmarks
            )
            observations.append(
                HandObservation(
                    hand=hand,
                    finger_axes_xy=finger_axes,
                    finger_quality=finger_quality,
                    handedness_label=label,
                    handedness_score=handedness_score,
                    source=(
                        "mediapipe_video"
                        if use_video and hasattr(landmarker, "detect_for_video")
                        else "mediapipe_image_crop"
                    ),
                    landmarks_xy=tuple(
                        (
                            float(landmark.x) * width,
                            float(landmark.y) * height,
                        )
                        for landmark in landmarks
                    ),
                    landmarks_z=tuple(
                        float(getattr(landmark, "z", 0.0)) for landmark in landmarks
                    ),
                    joint_quality=joint_quality,
                )
            )
        return tuple(observations)

    def _load_video_landmarker(self):
        if self._video_landmarker is not None:
            return self._video_landmarker
        # Non-production test doubles already provide a deterministic landmarker.
        if not isinstance(self.backend, MediaPipeHandBackend):
            self._video_landmarker = self.backend._load()
            return self._video_landmarker
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise BackendError("mediapipe is required for live hand tracking.") from exc
        if not self.backend.model_path.exists():
            # Preserve the backend's actionable missing-model error text.
            self.backend._load()
        config = self.backend.config
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.backend.model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._video_landmarker = vision.HandLandmarker.create_from_options(options)
        return self._video_landmarker

    @staticmethod
    def _finger_landmark_quality(
        landmarks: object,
        indices: tuple[int, ...],
    ) -> float:
        qualities = [
            MediaPipeHandExtractor._joint_landmark_quality(
                landmarks[index],  # type: ignore[index]
            )
            for index in indices
        ]
        return min(qualities) if qualities else 0.0

    @staticmethod
    def _joint_landmark_quality(
        landmark: object,
    ) -> float:
        raw_visibility = getattr(landmark, "visibility", None)
        raw_presence = getattr(landmark, "presence", None)
        visibility = 1.0 if raw_visibility is None else float(raw_visibility)
        presence = 1.0 if raw_presence is None else float(raw_presence)
        x = float(getattr(landmark, "x", -1.0))
        y = float(getattr(landmark, "y", -1.0))
        boundary = min(x, y, 1.0 - x, 1.0 - y)
        boundary_quality = float(np.clip(boundary / 0.04, 0.0, 1.0))
        return float(
            np.clip(
                visibility * presence * boundary_quality,
                0.0,
                1.0,
            )
        )

    def _select_observation(
        self,
        observations: tuple[HandObservation, ...],
    ) -> int:
        # Preserve the legacy direct-extractor behavior. DetectionChain performs
        # the final neck-aware selection from all returned candidates.
        wanted = "Right" if self.player_handedness == "right" else "Left"
        for index, observation in enumerate(observations):
            if observation.handedness_label == wanted:
                return index
        return max(
            range(len(observations)),
            key=lambda index: observations[index].hand.confidence,
        )

    def _select_hand(self, landmarks: object, handedness: object) -> int:
        if self.player_handedness == "right":
            # Preserve the proven F4c selector exactly for the default
            # right-handed-player path, including its first-match and
            # fingertip-spread fallback semantics.
            return _select_fretting_hand(landmarks, handedness)
        if len(landmarks) == 1:  # type: ignore[arg-type]
            return 0
        for index, info in enumerate(handedness):  # type: ignore[arg-type]
            if info and info[0].category_name == "Left":
                return index
        best_index, best_spread = 0, -1.0
        for index, hand_landmarks in enumerate(landmarks):  # type: ignore[arg-type]
            tip_xs = [
                float(hand_landmarks[indices[-1]].x)
                for indices in FINGER_AXIS_LANDMARKS.values()
            ]
            spread = max(tip_xs) - min(tip_xs)
            if spread > best_spread:
                best_index, best_spread = index, spread
        return best_index

    def close(self) -> None:
        if self._video_landmarker is not None:
            self._video_landmarker.close()
            self._video_landmarker = None
        self.backend.close()

    def reset(self) -> None:
        """Clear backend-local inference state at a new source boundary."""
        self.close()
        self._last_video_timestamp_ms = -1

    def reset_video_tracking(self) -> None:
        """Start a new VIDEO coordinate stream while retaining IMAGE state."""
        if self._video_landmarker is not None:
            self._video_landmarker.close()
            self._video_landmarker = None
        self._last_video_timestamp_ms = -1


@dataclass(frozen=True)
class HandPoint:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class FretTick:
    fret: int
    start: Point
    end: Point


@dataclass(frozen=True)
class StageLatency:
    detector_ms: float
    homography_ms: float
    hand_ms: float
    anchor_ms: float
    total_ms: float


@dataclass(frozen=True)
class _DetectorJobResult:
    generation: int
    timestamp_s: float
    source_frame: np.ndarray
    homography: Homography
    fret_centers: np.ndarray | None
    detector_ms: float
    homography_ms: float


@dataclass(frozen=True)
class FingerContact:
    """One finger's physical contact hypothesis and pose-derived reliability."""

    name: str
    fret: int
    raw_fret: float
    weight: float
    curl_ratio: float
    pressing_score: float
    barre: bool = False
    string: int | None = None
    visible: bool = True
    pressing: bool = True
    quality: float = 1.0
    contact_source: str = "tip"
    barre_strings: tuple[int, ...] = ()
    motion_score: float = 1.0


@dataclass(frozen=True)
class ConfidenceFactors:
    """Auditable factors used for one position observation."""

    board: float
    freshness: float
    stability: float
    landmark_quality: float
    on_neck: float
    finger_agreement: float
    coarse_agreement: float
    support_sufficiency: float
    combined: float
    chord_compatibility: float = 0.0
    blockers: tuple[str, ...] = ()


EMPTY_CONFIDENCE_FACTORS = ConfidenceFactors(
    board=0.0,
    freshness=0.0,
    stability=0.0,
    landmark_quality=0.0,
    on_neck=0.0,
    finger_agreement=0.0,
    coarse_agreement=0.0,
    support_sufficiency=0.0,
    combined=0.0,
    chord_compatibility=0.0,
    blockers=("no_observation",),
)


@dataclass(frozen=True)
class FrameDetection:
    timestamp_s: float
    detector_ran: bool
    neck_locked: bool
    fret_map_locked: bool
    homography_confidence: float
    homography_method: str
    neck_quad: tuple[Point, ...]
    fret_ticks: tuple[FretTick, ...]
    hand_points: tuple[HandPoint, ...]
    index_fret: float | None
    anchor: HandNeckAnchor
    stage_latency: StageLatency
    index_fret_raw: float | None = None
    composite_available: bool = False
    position_fret: float | None = None
    observation_confidence: float = 0.0
    confidence_factors: ConfidenceFactors = EMPTY_CONFIDENCE_FACTORS
    finger_contacts: tuple[FingerContact, ...] = ()
    geometry_status: str = "missing"
    geometry_age_ms: float = 0.0
    detector_age_ms: float = 0.0
    geometry_stability: float = 0.0
    hand_source: str = "none"
    fret_refinement_support: float = 0.0
    string_refinement_support: float = 0.0
    geometry_distortion_residual: float = 0.0
    nut_x: float | None = None
    body_joint_x: float | None = None
    boundary_support: float = 0.0
    body_joint_fret: int | None = None
    detector_requested: bool = False
    detector_pending: bool = False
    hand_detector_ran: bool = False
    hand_schedule_mode: str = "default"
    hand_detector_interval_ms: float = 100.0
    hand_refresh_reason: str = "deadline"
    hand_search_source: str = "none"
    hand_tracking_quality: float = 0.0
    hand_pose_quality: float = 0.0
    hand_pose_continuity: float = 0.0
    hand_pose_identity_score: float = 0.0
    hand_pose_residual_fraction: float = 1.0
    hand_pose_predicted: bool = False
    hand_detector_calls: int = 0
    hand_search_attempts: tuple[str, ...] = ()
    detector_result_consumed: bool = False
    detector_result_accepted: bool = False
    detector_request_timestamp_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready representation for replay now and the WebSocket later."""
        return asdict(self)


def _empty_homography() -> Homography:
    return Homography(H=np.eye(3, dtype=np.float64), confidence=0.0, method="missing")


def _project_canonical(homography: Homography, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    projected = homogeneous @ homography.H.T
    safe = np.abs(projected[:, 2]) >= 1e-12
    out = np.zeros((points.shape[0], 2), dtype=np.float64)
    out[safe] = projected[safe, :2] / projected[safe, 2:3]
    return out


def _neck_quad(homography: Homography) -> tuple[Point, ...]:
    if homography.confidence <= 0.0:
        return ()
    canonical = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    return tuple(
        (float(x), float(y)) for x, y in _project_canonical(homography, canonical)
    )


def _detector_refresh_is_continuous(
    previous: Homography,
    fresh: Homography,
    *,
    frame_shape: tuple[int, ...],
) -> bool:
    """Return whether a fresh detector result retains the same board frame.

    Fret centers live in canonical neck coordinates, so ordinary camera motion
    does not invalidate a trusted nonlinear fret axis. A large detector jump
    after tracking was lost can indicate a new board/endpoint interpretation;
    only that discontinuity should discard the axis.
    """
    if previous.confidence <= 0.0 or fresh.confidence <= 0.0:
        return False
    if len(frame_shape) < 2:
        return False
    height, width = frame_shape[:2]
    diagonal = max(1.0, math.hypot(width, height))
    old_quad = np.asarray(_neck_quad(previous), dtype=np.float64)
    new_quad = np.asarray(_neck_quad(fresh), dtype=np.float64)
    if (
        old_quad.shape != (4, 2)
        or new_quad.shape != (4, 2)
        or not np.all(np.isfinite(old_quad))
        or not np.all(np.isfinite(new_quad))
    ):
        return False
    displacements = np.linalg.norm(new_quad - old_quad, axis=1)
    rms = float(np.sqrt(np.mean(np.square(displacements)))) / diagonal
    maximum = float(np.max(displacements)) / diagonal
    return bool(
        rms <= DETECTOR_REFRESH_MAX_RMS_FRACTION
        and maximum <= DETECTOR_REFRESH_MAX_CORNER_FRACTION
    )


def _fret_wire_xs(fret_centers: np.ndarray) -> np.ndarray:
    """Recover rule-of-18 wire locations from calibrated cell centres."""
    centers = np.asarray(fret_centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.isfinite(centers)):
        return np.empty(0, dtype=np.float64)
    u0 = 1.0 - RULE_OF_18_RATIO**0.5
    u1 = 1.0 - RULE_OF_18_RATIO**1.5
    denom = u1 - u0
    if abs(denom) < 1e-12:
        return np.empty(0, dtype=np.float64)
    scale = float((centers[1] - centers[0]) / denom)
    origin = float(centers[0] - scale * u0)
    frets = np.arange(centers.size + 1, dtype=np.float64)
    rule18_wires = origin + scale * (1.0 - np.power(RULE_OF_18_RATIO, frets))
    center_frets = np.arange(centers.size, dtype=np.float64) + 0.5
    modeled_centers = origin + scale * (1.0 - np.power(RULE_OF_18_RATIO, center_frets))
    tolerance = max(1e-6, 1e-3 * float(np.ptp(centers)))
    if float(np.max(np.abs(modeled_centers - centers))) <= tolerance:
        return rule18_wires

    # A session affine calibration intentionally makes the fret-number axis
    # non-rule-18 in canonical coordinates. Preserve those calibrated cell
    # centers rather than fitting them back to an uncalibrated exponential.
    wires = np.empty(centers.size + 1, dtype=np.float64)
    wires[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    wires[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    wires[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return wires


def _fret_ticks(
    homography: Homography, fret_centers: np.ndarray | None
) -> tuple[FretTick, ...]:
    if homography.confidence <= 0.0 or fret_centers is None:
        return ()
    wire_xs = _fret_wire_xs(fret_centers)
    if wire_xs.size == 0:
        return ()
    endpoints = np.array(
        [[[x, 0.0], [x, 1.0]] for x in wire_xs],
        dtype=np.float64,
    ).reshape(-1, 2)
    projected = _project_canonical(homography, endpoints).reshape(-1, 2, 2)
    return tuple(
        FretTick(
            fret=fret,
            start=(float(line[0, 0]), float(line[0, 1])),
            end=(float(line[1, 0]), float(line[1, 1])),
        )
        for fret, line in enumerate(projected)
    )


def _hand_points(hand: HandSample | None) -> tuple[HandPoint, ...]:
    if hand is None:
        return ()
    points = [HandPoint("wrist", *hand.wrist_xy)]
    points.extend(
        HandPoint(name, *hand.fingers[name].tip_xy)
        for name in FRETTING_FINGERS
        if name in hand.fingers
    )
    return tuple(points)


def _empty_anchor() -> HandNeckAnchor:
    return HandNeckAnchor(
        center_fret=0.0,
        min_fret=0.0,
        max_fret=0.0,
        confidence=0.0,
        method="missing",
    )


def _fret_positions_from_canonical_x(
    canonical_x: np.ndarray,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Convert canonical neck x to fret numbers using the calibrated axis.

    ``calibrate_board`` returns one canonical-x cell centre per fret. The map
    may be increasing or decreasing depending on the detected nut direction,
    so interpolation must preserve its orientation. When there are not enough
    detected wires for a map, use the repository's established unit-neck
    convention: canonical x=0 is the nut and x=1 is fret 12, with rule-of-18
    spacing between them. This replaces the incorrect ``x * cfg.max_fret``
    conversion that treated the body joint as fret 24.
    """
    xs = np.asarray(canonical_x, dtype=np.float64)
    if fret_centers is not None:
        centers = np.asarray(fret_centers, dtype=np.float64)
        valid = (
            centers.shape == (cfg.max_fret + 1,)
            and np.all(np.isfinite(centers))
            and (np.all(np.diff(centers) > 0.0) or np.all(np.diff(centers) < 0.0))
        )
        if valid:
            frets = np.arange(cfg.max_fret + 1, dtype=np.float64)
            # The map contains one extra cell centre so callers can interpolate
            # through the body-side half of the last configured fret. Bound the
            # valid span at the outer edge of fret 1 and fret ``max_fret`` before
            # interpolation; np.interp otherwise silently clamps arbitrary
            # off-board coordinates to 0/max and fabricates fret 1/24 readings.
            nut_boundary = centers[0] - 0.5 * (centers[1] - centers[0])
            body_boundary = centers[-2] + 0.5 * (centers[-1] - centers[-2])
            support_min = min(float(nut_boundary), float(body_boundary))
            support_max = max(float(nut_boundary), float(body_boundary))
            supported = np.isfinite(xs) & (xs >= support_min) & (xs <= support_max)
            if centers[0] > centers[-1]:
                centers = centers[::-1]
                frets = frets[::-1]
            positions = np.full(xs.shape, np.nan, dtype=np.float64)
            positions[supported] = np.interp(xs[supported], centers, frets)
            return positions, "calibrated_fret_map"

    supported = (
        np.isfinite(xs) & (xs >= CANONICAL_NECK_MIN) & (xs <= CANONICAL_NECK_MAX)
    )
    positions = np.full(xs.shape, np.nan, dtype=np.float64)
    body_fraction = 1.0 - RULE_OF_18_RATIO**FALLBACK_BODY_JOINT_FRET
    positions[supported] = np.log1p(-xs[supported] * body_fraction) / math.log(
        RULE_OF_18_RATIO
    )
    return positions, "rule18_fret12_fallback"


def _fret_wire_axis(
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Return physical fret-wire coordinates from nut through ``max_fret``."""
    if fret_centers is not None:
        centers = np.asarray(fret_centers, dtype=np.float64)
        valid = (
            centers.shape == (cfg.max_fret + 1,)
            and np.all(np.isfinite(centers))
            and (np.all(np.diff(centers) > 0.0) or np.all(np.diff(centers) < 0.0))
        )
        if valid:
            wires = _fret_wire_xs(centers)
            if wires.size >= cfg.max_fret + 1:
                return wires[: cfg.max_fret + 1], "calibrated_fret_map"

    frets = np.arange(cfg.max_fret + 1, dtype=np.float64)
    body_fraction = 1.0 - RULE_OF_18_RATIO**FALLBACK_BODY_JOINT_FRET
    wires = (1.0 - np.power(RULE_OF_18_RATIO, frets)) / body_fraction
    return wires, "rule18_fret12_fallback"


def _rule18_fret_centers(
    cfg: GuitarConfig,
    *,
    body_joint_fret: int = FALLBACK_BODY_JOINT_FRET,
) -> np.ndarray:
    denominator = 1.0 - RULE_OF_18_RATIO**body_joint_fret
    frets = np.arange(cfg.max_fret + 1, dtype=np.float64) + 0.5
    return (1.0 - np.power(RULE_OF_18_RATIO, frets)) / denominator


def _apply_fret_axis_calibration(
    centers: np.ndarray,
    *,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Return cell centers whose decoded fret axis is scale*raw + offset."""
    base = np.asarray(centers, dtype=np.float64)
    if base.ndim != 1 or base.size < 2 or not np.all(np.isfinite(base)):
        raise ValueError("centers must be a finite one-dimensional fret axis")
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
        raise ValueError("calibration scale/offset must be finite and positive")
    target_positions = np.arange(base.size, dtype=np.float64)
    raw_positions = (target_positions - offset) / scale
    source_positions = np.arange(base.size, dtype=np.float64)
    calibrated = np.interp(raw_positions, source_positions, base)
    below = raw_positions < 0.0
    above = raw_positions > source_positions[-1]
    calibrated[below] = base[0] + raw_positions[below] * (base[1] - base[0])
    calibrated[above] = base[-1] + (raw_positions[above] - source_positions[-1]) * (
        base[-1] - base[-2]
    )
    if not (np.all(np.diff(calibrated) > 0.0) or np.all(np.diff(calibrated) < 0.0)):
        raise ValueError("calibrated fret axis is not monotonic")
    return calibrated


def _transform_fret_axis(
    centers: np.ndarray,
    canonical_transform: np.ndarray,
) -> np.ndarray:
    """Move base-canonical fret centers into the corrected canonical frame."""
    values = np.asarray(centers, dtype=np.float64)
    transform = np.asarray(canonical_transform, dtype=np.float64)
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return values.copy()
    points = np.column_stack(
        (
            values,
            np.full(values.shape, 0.5, dtype=np.float64),
            np.ones(values.shape, dtype=np.float64),
        )
    )
    projected = points @ inverse.T
    safe = np.abs(projected[:, 2]) > 1e-9
    if not np.all(safe):
        return values.copy()
    transformed = projected[:, 0] / projected[:, 2]
    if not (np.all(np.diff(transformed) > 0.0) or np.all(np.diff(transformed) < 0.0)):
        return values.copy()
    return transformed


def _live_fret_axis_is_trustworthy(
    refinement: GeometryRefinement,
    detector_centers: np.ndarray | None,
    cfg: GuitarConfig,
) -> bool:
    """Require independent string evidence before replacing the fret axis."""
    if (
        refinement.fret_centers is None
        or refinement.body_joint_fret is None
        or refinement.boundary_support < 0.12
        or refinement.fret_support < LIVE_FRET_AXIS_MIN_FRET_SUPPORT
        or refinement.string_support < LIVE_FRET_AXIS_MIN_STRING_SUPPORT
    ):
        return False
    candidate = np.asarray(refinement.fret_centers, dtype=np.float64)
    if (
        candidate.shape != (cfg.max_fret + 1,)
        or not np.all(np.isfinite(candidate))
        or not (np.all(np.diff(candidate) > 0.0) or np.all(np.diff(candidate) < 0.0))
    ):
        return False
    finite_strings = sum(
        np.asarray(curve, dtype=np.float64).shape == (3,) and np.all(np.isfinite(curve))
        for curve in refinement.string_curves
    )
    if finite_strings < min(cfg.n_strings, LIVE_FRET_AXIS_MIN_FINITE_STRINGS):
        return False

    if detector_centers is None:
        return True
    reference = np.asarray(detector_centers, dtype=np.float64)
    if (
        reference.shape != candidate.shape
        or not np.all(np.isfinite(reference))
        or not (np.all(np.diff(reference) > 0.0) or np.all(np.diff(reference) < 0.0))
    ):
        return False
    candidate_wires = _fret_wire_xs(candidate)
    reference_wires = _fret_wire_xs(reference)
    count = min(7, candidate_wires.size, reference_wires.size)
    if count < 3:
        return False
    candidate_lower = candidate_wires[:count]
    reference_lower = reference_wires[:count]
    candidate_direction = np.sign(np.median(np.diff(candidate_lower)))
    reference_direction = np.sign(np.median(np.diff(reference_lower)))
    if candidate_direction == 0.0 or candidate_direction != reference_direction:
        return False
    reference_cell = float(np.median(np.abs(np.diff(reference_lower))))
    allowed = max(
        0.025,
        LIVE_FRET_AXIS_MAX_LOWER_WIRE_CELL_FRACTION * reference_cell,
    )
    return bool(float(np.max(np.abs(candidate_lower - reference_lower))) <= allowed)


def _fret_cell_from_canonical_x(
    canonical_x: float,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    *,
    deadband_fraction: float = FRET_WIRE_DEADBAND_FRACTION,
) -> tuple[int, str] | None:
    """Classify a contact by fret-wire interval, favoring just-behind-wire play.

    Cell ``f`` is bounded by wires ``f-1`` and ``f``. A contact that projects
    just past wire ``f`` receives a local-width deadband and remains fret ``f``;
    this models normal placement immediately behind a wire without applying a
    fixed fret-number bias that would be wrong higher on the neck.
    """
    if not 0.0 <= deadband_fraction < 0.5:
        raise ValueError("deadband_fraction must be in [0, 0.5)")
    x = float(canonical_x)
    if not math.isfinite(x) or not CANONICAL_NECK_MIN <= x <= CANONICAL_NECK_MAX:
        return None

    wires, method = _fret_wire_axis(cfg, fret_centers)
    direction = float(np.sign(wires[-1] - wires[0]))
    if direction == 0.0:
        return None
    oriented_wires = wires * direction
    if not np.all(np.diff(oriented_wires) > 0.0):
        return None
    oriented_x = x * direction
    if oriented_x < oriented_wires[0] or oriented_x > oriented_wires[-1]:
        return None

    cell = max(1, int(np.searchsorted(oriented_wires, oriented_x, side="left")))
    if cell > cfg.max_fret:
        return None
    if cell > 1:
        nut_wire = oriented_wires[cell - 1]
        cell_width = oriented_wires[cell] - nut_wire
        distance_past_wire = oriented_x - nut_wire
        if 0.0 < distance_past_wire <= deadband_fraction * cell_width:
            cell -= 1
    return cell, method


def _on_canonical_neck(canonical_points: np.ndarray) -> np.ndarray:
    """Return a mask for finite image landmarks projected inside the neck."""
    points = np.asarray(canonical_points, dtype=np.float64)
    return (
        np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= CANONICAL_NECK_MIN)
        & (points[:, 0] <= CANONICAL_NECK_MAX)
        & (points[:, 1] >= CANONICAL_NECK_MIN)
        & (points[:, 1] <= CANONICAL_NECK_MAX)
    )


def _has_outward_longitudinal_wrist(
    hand: HandSample,
    homography: Homography,
) -> bool:
    """Reject a hand whose wrist extends outward from an end-of-neck grip.

    A picking hand can place three or four fingertips over the final sliver of
    the detected neck while its wrist remains farther out toward the soundhole.
    Mirror the same test at the nut end so camera/player orientation cannot
    change the result. Cross-string wrist displacement is deliberately ignored:
    a valid fretting wrist normally sits beside the neck.
    """
    fingertips = [
        hand.fingers[name].tip_xy for name in FRETTING_FINGERS if name in hand.fingers
    ]
    if not fingertips:
        return False
    try:
        canonical = project_to_canonical(
            homography,
            np.asarray([hand.wrist_xy, *fingertips], dtype=np.float64),
        )
    except np.linalg.LinAlgError:
        return False
    wrist_x = float(canonical[0, 0])
    tip_xs = canonical[1:, 0]
    tip_xs = tip_xs[np.isfinite(tip_xs)]
    if not math.isfinite(wrist_x) or tip_xs.size == 0:
        return False
    tip_x = float(np.median(tip_xs))
    edge = min(tip_x, 1.0 - tip_x)
    if edge > LONGITUDINAL_WRIST_EDGE_ZONE:
        return False
    if tip_x <= 0.5:
        return tip_x - wrist_x >= LONGITUDINAL_WRIST_OUTWARD_MARGIN
    return wrist_x - tip_x >= LONGITUDINAL_WRIST_OUTWARD_MARGIN


def _hand_overlaps_neck(hand: HandSample | None, homography: Homography) -> bool:
    """Require most fretting fingertips to overlap the canonical neck.

    A picking index can briefly cross the soundhole end of the board even while
    the rest of that hand is outside it.  Three of the four fingertips provides
    a hand-level geometry check while allowing one occluded or hovering finger.
    The wrist is intentionally excluded because a real fretting wrist normally
    sits beyond the cross-string edge of the neck.
    """
    if hand is None or homography.confidence <= 0.0:
        return False
    fingertips = [
        hand.fingers[name].tip_xy for name in FRETTING_FINGERS if name in hand.fingers
    ]
    if len(fingertips) < MIN_ON_NECK_FINGERTIPS:
        return False
    try:
        canonical_points = project_to_canonical(
            homography, np.asarray(fingertips, dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return False
    return int(
        np.count_nonzero(_on_canonical_neck(canonical_points))
    ) >= MIN_ON_NECK_FINGERTIPS and not _has_outward_longitudinal_wrist(
        hand, homography
    )


def compute_position_anchor(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
) -> HandNeckAnchor:
    """Project a coarse hand centroid through the calibrated fret coordinate."""
    if hand is None or homography.confidence <= 0.0:
        return _empty_anchor()

    points = [hand.wrist_xy]
    points.extend(
        hand.fingers[name].tip_xy for name in FRETTING_FINGERS if name in hand.fingers
    )
    if not points:
        return _empty_anchor()
    try:
        canonical_points = project_to_canonical(
            homography, np.asarray(points, dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return _empty_anchor()

    on_neck = _on_canonical_neck(canonical_points)
    if not np.any(on_neck):
        return _empty_anchor()

    fret_positions, method = _fret_positions_from_canonical_x(
        canonical_points[on_neck, 0], cfg, fret_centers
    )
    fret_positions = fret_positions[np.isfinite(fret_positions)]
    if fret_positions.size == 0:
        return _empty_anchor()
    raw_min = float(fret_positions.min())
    raw_max = float(fret_positions.max())
    center = float(np.median(fret_positions))
    spread = max(0.0, raw_max - raw_min)
    span_penalty = min(0.5, spread / max(float(cfg.max_fret), 1.0))
    confidence = min(
        1.0,
        float(hand.confidence) * float(homography.confidence) * (1.0 - span_penalty),
    )
    return HandNeckAnchor(
        center_fret=center,
        min_fret=max(0.0, raw_min - 1.0),
        max_fret=min(float(cfg.max_fret), raw_max + 1.0),
        confidence=max(0.0, confidence),
        method=f"mediapipe_{method}",
    )


def _barre_contact(
    hand: HandSample,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    finger_name: str,
    axis_xy: tuple[Point, ...],
    string_curves: tuple[tuple[float, float, float], ...] = (),
) -> tuple[float | None, tuple[int, ...]]:
    finger = hand.fingers.get(finger_name)
    if (
        finger is None
        or finger.curl_ratio < BARRE_INDEX_MIN_EXTENSION
        or len(axis_xy) < 3
    ):
        return None, ()
    try:
        canonical_axis = project_to_canonical(
            homography, np.asarray(axis_xy[-3:], dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return None, ()
    usable = (
        np.all(np.isfinite(canonical_axis), axis=1)
        & (canonical_axis[:, 0] >= CANONICAL_NECK_MIN)
        & (canonical_axis[:, 0] <= CANONICAL_NECK_MAX)
        & (canonical_axis[:, 1] >= CANONICAL_NECK_MIN - 0.08)
        & (canonical_axis[:, 1] <= CANONICAL_NECK_MAX + 0.08)
    )
    if np.count_nonzero(usable) < 2:
        return None, ()
    usable_axis = canonical_axis[usable]
    along_span = float(np.ptp(usable_axis[:, 0]))
    cross_span = float(np.ptp(usable_axis[:, 1]))
    minimum_span = (
        BARRE_MIN_CROSS_NECK_SPAN
        if finger_name == "index"
        else PARTIAL_BARRE_MIN_CROSS_NECK_SPAN
    )
    barre = (
        cross_span >= minimum_span
        and cross_span >= BARRE_MIN_CROSS_TO_ALONG_RATIO * max(along_span, 1e-6)
    )
    if not barre:
        return None, ()
    axis_x = float(np.median(usable_axis[:, 0]))
    if _fret_cell_from_canonical_x(axis_x, cfg, fret_centers) is None:
        return None, ()
    string_values = np.asarray(
        [
            nearest_string(
                float(point[0]),
                float(point[1]),
                n_strings=cfg.n_strings,
                string_curves=string_curves,
            )
            for point in usable_axis
        ],
        dtype=np.int64,
    )
    first = int(string_values.min())
    last = int(string_values.max())
    if finger_name != "index" and last - first + 1 > 2:
        center = int(round(float(np.median(string_values))))
        first = int(np.clip(center, 1, cfg.n_strings - 1))
        last = first + 1
    strings = tuple(range(first, last + 1))
    if len(strings) < 2:
        return None, ()
    return axis_x, strings


def _raw_fret_from_canonical_x(
    canonical_x: float,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
) -> float | None:
    positions, method = _fret_positions_from_canonical_x(
        np.asarray([canonical_x], dtype=np.float64),
        cfg,
        fret_centers,
    )
    if not np.all(np.isfinite(positions)):
        return None
    offset = 1.0 if method == "calibrated_fret_map" else 0.5
    fret = float(positions[0] + offset)
    if not 0.5 <= fret <= float(cfg.max_fret) + 0.5:
        return None
    return fret


def compute_finger_contacts(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    *,
    finger_axes_xy: dict[str, tuple[Point, ...]] | None = None,
    finger_quality: dict[str, float] | None = None,
    finger_stillness: dict[str, float] | None = None,
    string_curves: tuple[tuple[float, float, float], ...] = (),
) -> tuple[FingerContact, ...]:
    """Classify visible, pressing, hovering, and barre contact hypotheses."""
    if hand is None or homography.confidence <= 0.0:
        return ()
    axes = finger_axes_xy or {}
    qualities = finger_quality or {}
    stillness = finger_stillness or {}
    contacts: list[FingerContact] = []
    for name in FRETTING_FINGERS:
        finger = hand.fingers.get(name)
        if finger is None:
            continue
        try:
            canonical_tip = project_to_canonical(
                homography,
                np.asarray([finger.tip_xy], dtype=np.float64),
            )
        except np.linalg.LinAlgError:
            continue
        if not _on_canonical_neck(canonical_tip)[0]:
            continue

        canonical_contact = canonical_tip[0].copy()
        contact_source = "tip"
        axis = axes.get(name, ())
        if len(axis) >= 2:
            try:
                distal = project_to_canonical(
                    homography,
                    np.asarray(axis[-2:], dtype=np.float64),
                )
            except np.linalg.LinAlgError:
                distal = np.empty((0, 2), dtype=np.float64)
            if distal.shape == (2, 2) and np.all(np.isfinite(distal)):
                # The string-facing finger pad sits slightly proximal to the
                # mathematical tip. A conservative blend avoids jumping a wire
                # when a curled distal segment is nearly parallel to the neck.
                canonical_contact = 0.20 * distal[0] + 0.80 * distal[1]
                contact_source = "finger_pad"

        contact_x = float(canonical_contact[0])
        barre_x, barre_strings = _barre_contact(
            hand,
            homography,
            cfg,
            fret_centers,
            name,
            axis,
            string_curves,
        )
        barre = barre_x is not None
        if barre_x is not None:
            contact_x = barre_x
            contact_source = "barre_axis"
        classified = _fret_cell_from_canonical_x(
            contact_x,
            cfg,
            fret_centers,
            deadband_fraction=(FRET_WIRE_DEADBAND_FRACTION if barre else 0.0),
        )
        raw_fret = _raw_fret_from_canonical_x(
            float(canonical_tip[0, 0]),
            cfg,
            fret_centers,
        )
        if classified is None or raw_fret is None:
            continue

        quality = float(np.clip(qualities.get(name, 1.0), 0.0, 1.0))
        motion_score = float(np.clip(stillness.get(name, 1.0), 0.0, 1.0))
        if barre:
            curl_score = 0.92 if name == "index" else 0.75
        else:
            curl_score = float(
                np.clip((0.96 - float(finger.curl_ratio)) / 0.18, 0.0, 1.0)
            )
        depth_delta = max(
            0.0,
            abs(float(finger.tip_z) - float(hand.wrist_z)) - 0.04,
        )
        depth_score = math.exp(-depth_delta / 0.08)
        neck_edge_distance = min(
            float(canonical_contact[1]),
            1.0 - float(canonical_contact[1]),
        )
        neck_proximity = float(np.clip(neck_edge_distance / 0.08, 0.0, 1.0))
        pressing_score = (
            curl_score
            * (0.75 + 0.15 * depth_score + 0.10 * neck_proximity)
            * (0.65 + 0.35 * quality)
            * (0.75 + 0.25 * motion_score)
        )
        pressing_threshold = (
            0.55 if name == "index" and barre else 0.72 if barre else 0.30
        )
        visible = quality >= 0.15
        pressing = visible and pressing_score >= pressing_threshold
        weight = (
            FINGER_BASE_WEIGHTS[name]
            * float(np.clip(hand.confidence, 0.0, 1.0))
            * pressing_score
            * quality
        )
        string = nearest_string(
            float(canonical_contact[0]),
            float(canonical_contact[1]),
            n_strings=cfg.n_strings,
            string_curves=string_curves,
        )
        contacts.append(
            FingerContact(
                name=name,
                fret=int(classified[0]),
                raw_fret=raw_fret,
                weight=float(np.clip(weight, 0.0, 1.0)),
                curl_ratio=float(finger.curl_ratio),
                pressing_score=float(np.clip(pressing_score, 0.0, 1.0)),
                barre=barre,
                string=string,
                visible=visible,
                pressing=pressing,
                quality=quality,
                contact_source=contact_source,
                barre_strings=barre_strings,
                motion_score=motion_score,
            )
        )
    return tuple(contacts)


def _chord_shape_compatibility(contacts: list[FingerContact]) -> float:
    """Return supporting physical compatibility without inventing a position."""
    if len(contacts) < 2:
        return 1.0
    order = {name: index for index, name in enumerate(FRETTING_FINGERS)}
    compatible_weight = 0.0
    total_weight = 0.0
    for left_index, left in enumerate(contacts):
        for right in contacts[left_index + 1 :]:
            pair_weight = max(1e-6, min(left.weight, right.weight))
            total_weight += pair_weight
            if left.barre or right.barre:
                compatible_weight += pair_weight
                continue
            earlier, later = (
                (left, right)
                if order[left.name] <= order[right.name]
                else (right, left)
            )
            fret_delta = later.fret - earlier.fret
            distinct_string = (
                earlier.string is None
                or later.string is None
                or earlier.string != later.string
            )
            # Normal chords, stretches, and contractions fit inside a broad
            # six-fret hand span. A later finger one fret behind is possible;
            # larger reversals or two contacts on one string are contradictory.
            if distinct_string and -1 <= fret_delta <= 6:
                compatible_weight += pair_weight
    if total_weight <= 1e-9:
        return 0.0
    return float(np.clip(compatible_weight / total_weight, 0.0, 1.0))


def solve_hand_position(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    anchor: HandNeckAnchor,
    *,
    finger_axes_xy: dict[str, tuple[Point, ...]] | None = None,
    finger_quality: dict[str, float] | None = None,
    finger_stillness: dict[str, float] | None = None,
    string_curves: tuple[tuple[float, float, float], ...] = (),
    freshness: float = 1.0,
    geometry_stability: float = 1.0,
) -> tuple[float | None, tuple[FingerContact, ...], ConfidenceFactors]:
    """Score every classical position from all reliable finger contacts."""
    blockers: list[str] = []
    if homography.confidence <= 0.0:
        blockers.append("no_board")
    if hand is None:
        blockers.append("no_hand")
    if blockers:
        factors = ConfidenceFactors(
            board=max(0.0, float(homography.confidence)),
            freshness=max(0.0, float(freshness)),
            stability=max(0.0, float(geometry_stability)),
            landmark_quality=0.0 if hand is None else float(hand.confidence),
            on_neck=0.0,
            finger_agreement=0.0,
            coarse_agreement=0.0,
            support_sufficiency=0.0,
            combined=0.0,
            chord_compatibility=0.0,
            blockers=tuple(blockers),
        )
        return None, (), factors

    assert hand is not None
    contacts = compute_finger_contacts(
        hand,
        homography,
        cfg,
        fret_centers,
        finger_axes_xy=finger_axes_xy,
        finger_quality=finger_quality,
        finger_stillness=finger_stillness,
        string_curves=string_curves,
    )
    available = [
        hand.fingers[name].tip_xy for name in FRETTING_FINGERS if name in hand.fingers
    ]
    on_neck_count = 0
    if available:
        try:
            canonical = project_to_canonical(
                homography, np.asarray(available, dtype=np.float64)
            )
            on_neck_count = int(np.count_nonzero(_on_canonical_neck(canonical)))
        except np.linalg.LinAlgError:
            on_neck_count = 0
    on_neck = on_neck_count / max(len(available), 1)
    if on_neck_count < MIN_ON_NECK_FINGERTIPS:
        blockers.append("off_neck")

    useful = [
        contact
        for contact in contacts
        if contact.visible and contact.pressing and contact.weight >= 0.05
    ]
    total_weight = sum(contact.weight for contact in useful)
    if not useful or total_weight <= 1e-9:
        blockers.append("too_few_contacts")
    single_non_index_barre = (
        len(useful) == 1 and useful[0].barre and useful[0].name != "index"
    )
    if single_non_index_barre:
        # A single partial middle/ring/pinky axis is too easy to manufacture
        # from perspective or a hovering finger. Preserve its diagnostic
        # candidate, but require independent contact evidence before locking.
        blockers.append("too_few_contacts")
    single_barre_at_boundary = len(useful) == 1 and useful[0].barre and on_neck < 0.99
    if single_barre_at_boundary:
        # A lone extended index is easy to hallucinate when the hand/neck is
        # clipped.  Require the rest of the detected hand to support the
        # board boundary before treating barre geometry as sufficient by
        # itself.
        blockers.append("boundary_clipped")

    candidate: float | None = None
    finger_agreement = 0.0
    support = 0.0
    coarse_agreement = 0.0
    chord_compatibility = 0.0
    if useful and total_weight > 1e-9:
        sigmas = {
            "index": 0.55,
            "middle": 1.0,
            "ring": 1.15,
            "pinky": 1.30,
        }
        index_contact = next(
            (
                contact
                for contact in useful
                if contact.name == "index" and contact.weight >= 0.5
            ),
            None,
        )
        contact_span = max(contact.fret for contact in useful) - min(
            contact.fret for contact in useful
        )
        stretched_from_index = (
            index_contact is not None
            and 4 <= contact_span <= 6
            and index_contact.fret == min(contact.fret for contact in useful)
        )
        scoring_weight = {
            contact.name: contact.weight
            * (1.5 if stretched_from_index and contact.name == "index" else 1.0)
            for contact in useful
        }
        scoring_total = sum(scoring_weight.values())
        scored: list[tuple[float, int]] = []
        for position in range(1, cfg.max_fret + 1):
            score = 0.0
            for contact in useful:
                sigma = 0.30 if contact.barre else sigmas[contact.name]
                residual = (
                    float(contact.fret)
                    - float(position)
                    - FINGER_POSITION_OFFSETS[contact.name]
                )
                compatibility = math.exp(-0.5 * (residual / sigma) ** 2)
                score += scoring_weight[contact.name] * compatibility
            scored.append((score / scoring_total, position))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_position = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        separation = float(np.clip((best_score - runner_up) / 0.20, 0.0, 1.0))
        finger_agreement = best_score * (0.5 + 0.5 * separation)
        support = float(np.clip(total_weight / 1.5, 0.0, 1.0))
        if any(
            contact.name == "index" and contact.barre and contact.weight >= 0.5
            for contact in useful
        ):
            support = max(support, 0.85)
        if single_non_index_barre:
            support = 0.0
        chord_compatibility = _chord_shape_compatibility(useful)
        # Chord geometry may strengthen trust in otherwise-supported evidence,
        # but it is never allowed to rescue a contradictory finger candidate.
        support *= 0.70 + 0.30 * chord_compatibility
        if anchor.confidence > 0.0:
            expected_center = float(best_position) + 1.5
            if anchor.min_fret <= expected_center <= anchor.max_fret:
                coarse_agreement = 1.0
            else:
                distance = min(
                    abs(expected_center - anchor.min_fret),
                    abs(expected_center - anchor.max_fret),
                )
                coarse_agreement = math.exp(-0.5 * (distance / 2.0) ** 2)
        if finger_agreement < 0.30:
            blockers.append("finger_conflict")
        if support < 0.35:
            blockers.append("too_few_contacts")
        # Preserve the discrete playable-position winner while exposing a
        # bounded continuous residual for the optional session calibration.
        residual_weight = 0.0
        residual_sum = 0.0
        for contact in useful:
            if contact.barre:
                continue
            residual = contact.raw_fret - float(contact.fret)
            residual_sum += contact.weight * residual
            residual_weight += contact.weight
        continuous_residual = (
            residual_sum / residual_weight if residual_weight > 1e-9 else 0.0
        )
        if abs(continuous_residual) < 0.03:
            continuous_residual = 0.0
        candidate = float(best_position) + float(
            np.clip(continuous_residual, -0.45, 0.45)
        )

    board = float(np.clip(homography.confidence, 0.0, 1.0))
    freshness = float(np.clip(freshness, 0.0, 1.0))
    stability = float(np.clip(geometry_stability, 0.0, 1.0))
    landmark_quality = float(np.clip(hand.confidence, 0.0, 1.0))
    on_neck = float(np.clip(on_neck, 0.0, 1.0))
    coarse_agreement = float(np.clip(coarse_agreement, 0.0, 1.0))
    finger_agreement = float(np.clip(finger_agreement, 0.0, 1.0))
    support = float(np.clip(support, 0.0, 1.0))
    chord_compatibility = float(np.clip(chord_compatibility, 0.0, 1.0))

    weighted = (
        max(finger_agreement, 1e-6) ** 0.30
        * max(landmark_quality, 1e-6) ** 0.15
        * max(board * freshness, 1e-6) ** 0.20
        * max(stability, 1e-6) ** 0.15
        * max(on_neck, 1e-6) ** 0.10
        * max(coarse_agreement, 1e-6) ** 0.10
    )
    # Agreement is both one component of the geometric mean and a hard
    # reliability gate.  Without the second factor, four mutually
    # contradictory fingers can still look "confident" because the other
    # healthy factors dominate the mean.  Preserve the best raw candidate for
    # diagnostics, but force disagreement below the estimator's abstention
    # threshold.
    combined = float(np.clip(weighted * support * finger_agreement, 0.0, 1.0))
    if single_barre_at_boundary:
        combined = 0.0
    if combined < MIN_POSITION_OBSERVATION_CONFIDENCE:
        blockers.append("low_confidence")
    factors = ConfidenceFactors(
        board=board,
        freshness=freshness,
        stability=stability,
        landmark_quality=landmark_quality,
        on_neck=on_neck,
        finger_agreement=finger_agreement,
        coarse_agreement=coarse_agreement,
        support_sufficiency=support,
        combined=combined,
        chord_compatibility=chord_compatibility,
        blockers=tuple(dict.fromkeys(blockers)),
    )
    return candidate, contacts, factors


def _hand_crop_rect(
    frame: np.ndarray,
    neck_quad: tuple[Point, ...],
    *,
    hand_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[int, int, int, int] | None:
    if len(neck_quad) != 4:
        return None
    quad = np.asarray(neck_quad, dtype=np.float64)
    if not np.all(np.isfinite(quad)):
        return None
    height, width = frame.shape[:2]
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    short_edge = float(np.median(np.sort(edges)[:2]))
    long_edge = float(np.median(np.sort(edges)[-2:]))
    padding = max(1.8 * short_edge, 0.06 * long_edge, 8.0)
    left = float(quad[:, 0].min() - padding)
    top = float(quad[:, 1].min() - padding)
    right = float(quad[:, 0].max() + padding)
    bottom = float(quad[:, 1].max() + padding)
    if hand_bounds is not None:
        hand_padding = max(0.6 * short_edge, 8.0)
        hand_left, hand_top, hand_right, hand_bottom = hand_bounds
        left = min(left, hand_left - hand_padding)
        top = min(top, hand_top - hand_padding)
        right = max(right, hand_right + hand_padding)
        bottom = max(bottom, hand_bottom + hand_padding)
    x0 = max(0, math.floor(left))
    y0 = max(0, math.floor(top))
    x1 = min(width, math.ceil(right))
    y1 = min(height, math.ceil(bottom))
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) >= 0.90 * width * height:
        return None
    return x0, y0, x1, y1


def _last_hand_crop_rect(
    frame: np.ndarray,
    hand_bounds: tuple[float, float, float, float] | None,
    neck_quad: tuple[Point, ...],
) -> tuple[int, int, int, int] | None:
    """Return a tighter last-known-hand crop for high-resolution landmarks."""
    if hand_bounds is None or len(neck_quad) != 4:
        return None
    height, width = frame.shape[:2]
    quad = np.asarray(neck_quad, dtype=np.float64)
    if not np.all(np.isfinite(quad)):
        return None
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    short_edge = float(np.median(np.sort(edges)[:2]))
    left, top, right, bottom = hand_bounds
    pad_x = max(1.4 * short_edge, 24.0)
    pad_y = max(1.8 * short_edge, 24.0)
    x0 = max(0, math.floor(left - pad_x))
    y0 = max(0, math.floor(top - pad_y))
    x1 = min(width, math.ceil(right + pad_x))
    y1 = min(height, math.ceil(bottom + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) >= 0.80 * width * height:
        return None
    return x0, y0, x1, y1


def _prepare_hand_crop(
    frame: np.ndarray,
    crop_rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, float, float]:
    """Upscale a small neck/hand search crop and return source-per-model scale."""
    import cv2

    x0, y0, x1, y1 = crop_rect
    crop = frame[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    long_edge = max(height, width)
    if long_edge <= 0:
        return crop, 1.0, 1.0
    scale = min(
        HAND_SEARCH_UPSCALE_MAX_SCALE,
        max(1.0, HAND_SEARCH_UPSCALE_MIN_LONG_EDGE / long_edge),
    )
    if scale <= 1.0:
        return crop, 1.0, 1.0
    resized = cv2.resize(
        crop,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, width / resized.shape[1], height / resized.shape[0]


def _hand_bounds(hand: HandSample) -> tuple[float, float, float, float]:
    points = np.asarray(
        [hand.wrist_xy, *(finger.tip_xy for finger in hand.fingers.values())],
        dtype=np.float64,
    )
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _translate_hand_sample(
    hand: HandSample,
    *,
    offset_x: float,
    offset_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    z_scale: float = 1.0,
) -> HandSample:
    translated_fingers = {
        name: type(finger)(
            name=finger.name,
            tip_xy=(
                float(finger.tip_xy[0]) * scale_x + offset_x,
                float(finger.tip_xy[1]) * scale_y + offset_y,
            ),
            tip_z=float(finger.tip_z) * z_scale,
            curl_ratio=finger.curl_ratio,
        )
        for name, finger in hand.fingers.items()
    }
    return HandSample(
        wrist_xy=(
            float(hand.wrist_xy[0]) * scale_x + offset_x,
            float(hand.wrist_xy[1]) * scale_y + offset_y,
        ),
        wrist_z=float(hand.wrist_z) * z_scale,
        is_left_hand=hand.is_left_hand,
        confidence=hand.confidence,
        fingers=translated_fingers,
    )


def _translate_hand_observation(
    observation: HandObservation | HandSample | None,
    *,
    offset_x: float,
    offset_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    z_scale: float = 1.0,
) -> HandObservation | HandSample | None:
    if observation is None:
        return None
    if isinstance(observation, HandObservation):
        return HandObservation(
            hand=_translate_hand_sample(
                observation.hand,
                offset_x=offset_x,
                offset_y=offset_y,
                scale_x=scale_x,
                scale_y=scale_y,
                z_scale=z_scale,
            ),
            finger_axes_xy={
                name: tuple(
                    (
                        x * scale_x + offset_x,
                        y * scale_y + offset_y,
                    )
                    for x, y in points
                )
                for name, points in observation.finger_axes_xy.items()
            },
            finger_quality=dict(observation.finger_quality),
            handedness_label=observation.handedness_label,
            handedness_score=observation.handedness_score,
            source=observation.source,
            landmarks_xy=tuple(
                (
                    x * scale_x + offset_x,
                    y * scale_y + offset_y,
                )
                for x, y in observation.landmarks_xy
            ),
            landmarks_z=tuple(
                float(value) * z_scale for value in observation.landmarks_z
            ),
            joint_quality=observation.joint_quality,
            finger_stillness=dict(observation.finger_stillness),
            pose_quality=observation.pose_quality,
            pose_continuity=observation.pose_continuity,
            pose_identity_score=observation.pose_identity_score,
            pose_residual_fraction=observation.pose_residual_fraction,
            pose_predicted=observation.pose_predicted,
            detector_observation_accepted=observation.detector_observation_accepted,
        )
    return _translate_hand_sample(
        observation,
        offset_x=offset_x,
        offset_y=offset_y,
        scale_x=scale_x,
        scale_y=scale_y,
        z_scale=z_scale,
    )


def _extract_candidates(
    extractor: HandExtractor,
    frame: np.ndarray,
    *,
    timestamp_s: float,
    use_video: bool = True,
) -> tuple[HandObservation | HandSample, ...]:
    candidate_extractor = getattr(extractor, "extract_candidates", None)
    if candidate_extractor is not None:
        candidates = candidate_extractor(
            frame,
            **_supported_extractor_kwargs(
                candidate_extractor,
                timestamp_s=timestamp_s,
                use_video=use_video,
            ),
        )
        return tuple(candidate for candidate in candidates if candidate is not None)
    extract = extractor.extract
    observation = extract(
        frame,
        **_supported_extractor_kwargs(
            extract,
            timestamp_s=timestamp_s,
        ),
    )
    return () if observation is None else (observation,)


def _supported_extractor_kwargs(
    function: Callable[..., object],
    **values: object,
) -> dict[str, object]:
    """Select supported keywords without retrying an inference-side TypeError."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        # Opaque callables get one attempt with the current full protocol.
        # Retrying a TypeError could repeat expensive inference and conceal the
        # extractor's actual failure.
        return dict(values)
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_keywords:
        return dict(values)
    return {
        name: value
        for name, value in values.items()
        if name in parameters
        and parameters[name].kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _hand_sample(
    observation: HandObservation | HandSample,
) -> HandSample:
    return observation.hand if isinstance(observation, HandObservation) else observation


def _map_candidate_from_crop(
    observation: HandObservation | HandSample,
    *,
    crop_rect: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    z_scale: float,
    source: str,
) -> HandObservation | HandSample:
    x0, y0, _, _ = crop_rect
    translated = _translate_hand_observation(
        observation,
        offset_x=float(x0),
        offset_y=float(y0),
        scale_x=scale_x,
        scale_y=scale_y,
        z_scale=z_scale,
    )
    assert translated is not None
    if isinstance(translated, HandObservation):
        return HandObservation(
            hand=translated.hand,
            finger_axes_xy=translated.finger_axes_xy,
            finger_quality=translated.finger_quality,
            handedness_label=translated.handedness_label,
            handedness_score=translated.handedness_score,
            source=source,
            landmarks_xy=translated.landmarks_xy,
            landmarks_z=translated.landmarks_z,
            joint_quality=translated.joint_quality,
            finger_stillness=dict(translated.finger_stillness),
            pose_quality=translated.pose_quality,
            pose_continuity=translated.pose_continuity,
            pose_identity_score=translated.pose_identity_score,
            pose_residual_fraction=translated.pose_residual_fraction,
            pose_predicted=translated.pose_predicted,
            detector_observation_accepted=translated.detector_observation_accepted,
        )
    return translated


def _tracked_curl_ratio(points_xy: np.ndarray, fallback: float) -> float:
    """Compute the shared MediaPipe curl ratio from a tracked finger axis."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points[[0, 1, 3]])):
        return float(fallback)
    mcp, pip, _, tip = points
    tip_pip = float(np.linalg.norm(tip - pip))
    pip_mcp = float(np.linalg.norm(pip - mcp))
    denominator = tip_pip + pip_mcp
    if denominator <= 1e-6:
        return float(fallback)
    return float(np.linalg.norm(tip - mcp) / denominator)


def _tracked_hand_observation(
    tracked: TrackedHandLandmarks,
    template: HandObservation,
) -> HandObservation | None:
    """Rebuild FretCam's hand sample from temporally filtered landmarks."""
    wrist_xy = tracked.landmarks_xy[0]
    if not np.all(np.isfinite(wrist_xy)):
        return None
    wrist_z = float(tracked.landmarks_z[0])
    if not math.isfinite(wrist_z):
        wrist_z = float(template.hand.wrist_z)

    fingers: dict[str, FingerSample] = {}
    axes: dict[str, tuple[Point, ...]] = {}
    for name in FRETTING_FINGERS:
        indices = TRACKED_FINGER_LANDMARKS[name]
        tracked_axis = tracked.landmarks_xy[np.asarray(indices, dtype=np.int64)]
        tip_xy = tracked_axis[-1]
        if not np.all(np.isfinite(tip_xy)):
            continue
        old_finger = template.hand.fingers.get(name)
        fallback_curl = 1.0 if old_finger is None else old_finger.curl_ratio
        tip_z = float(tracked.landmarks_z[indices[-1]])
        if not math.isfinite(tip_z):
            tip_z = 0.0 if old_finger is None else float(old_finger.tip_z)
        fingers[name] = FingerSample(
            name=name,
            tip_xy=(float(tip_xy[0]), float(tip_xy[1])),
            tip_z=tip_z,
            curl_ratio=_tracked_curl_ratio(tracked_axis, fallback_curl),
        )
        axes[name] = tuple(
            (float(point[0]), float(point[1]))
            for point in tracked_axis
            if np.all(np.isfinite(point))
        )
    if not fingers:
        return None

    source_values = {
        tracked.finger_quality[name].source
        for name in FRETTING_FINGERS
        if name in tracked.finger_quality
    }
    if tracked.pose.predicted:
        source = "temporal_pose_predicted"
    elif "optical_flow" in source_values:
        source = "temporal_optical_flow"
    elif source_values & {"held", "held_rejected"}:
        source = "temporal_held"
    else:
        source = "temporal_detector"
    # Landmark quality already contributes to pose quality, so multiplying all
    # three here would count the same weak joint evidence twice. Pose remains a
    # hard identity/geometry gate and a separate scheduler diagnostic; contact
    # admission is independently protected by per-finger quality.
    confidence = (
        max(0.0, float(template.hand.confidence))
        * max(0.0, float(tracked.hand_quality))
    ) ** 0.5
    hand = HandSample(
        wrist_xy=(float(wrist_xy[0]), float(wrist_xy[1])),
        wrist_z=wrist_z,
        is_left_hand=(
            template.hand.is_left_hand
            if tracked.is_left_hand is None
            else tracked.is_left_hand
        ),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        fingers=fingers,
    )
    return HandObservation(
        hand=hand,
        finger_axes_xy=axes,
        finger_quality=tracked.fretting_finger_quality,
        handedness_label=template.handedness_label,
        handedness_score=template.handedness_score,
        source=source,
        landmarks_xy=tuple(
            (float(point[0]), float(point[1])) for point in tracked.landmarks_xy
        ),
        landmarks_z=tuple(float(value) for value in tracked.landmarks_z),
        joint_quality=tuple(float(value) for value in tracked.joint_quality),
        finger_stillness=dict(template.finger_stillness),
        pose_quality=tracked.pose.quality,
        pose_continuity=tracked.pose.continuity,
        pose_identity_score=tracked.pose.identity_score,
        pose_residual_fraction=tracked.pose.residual_fraction,
        pose_predicted=tracked.pose.predicted,
        detector_observation_accepted=tracked.detector_observation_accepted,
    )


def compute_index_fret(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    *,
    index_axis_xy: tuple[Point, ...] = (),
    fallback: float | None = None,
) -> float | None:
    """Return the technique-aware physical fret cell used for position lock.

    A fretted note is defined by the wire interval containing its contact, not
    by the nearest cell centre. Extended, across-neck index fingers additionally
    use the median PIP/DIP/tip axis coordinate so an angled barre is not
    represented by its fingertip alone. The exact continuous tip coordinate
    remains available separately through :func:`compute_index_fret_raw`.
    """
    if hand is None or homography.confidence <= 0.0:
        return None
    index = hand.fingers.get("index")
    if index is None:
        return None if fallback is None else float(max(1, round(fallback)))
    try:
        canonical_point = project_to_canonical(
            homography, np.asarray([index.tip_xy], dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return fallback
    if not _on_canonical_neck(canonical_point)[0]:
        return None

    contact_x = float(canonical_point[0, 0])
    barre_oriented = False
    if index.curl_ratio >= BARRE_INDEX_MIN_EXTENSION and len(index_axis_xy) >= 3:
        try:
            canonical_axis = project_to_canonical(
                homography, np.asarray(index_axis_xy[-3:], dtype=np.float64)
            )
        except np.linalg.LinAlgError:
            canonical_axis = np.empty((0, 2), dtype=np.float64)
        usable = (
            np.all(np.isfinite(canonical_axis), axis=1)
            & (canonical_axis[:, 0] >= CANONICAL_NECK_MIN)
            & (canonical_axis[:, 0] <= CANONICAL_NECK_MAX)
        )
        if np.count_nonzero(usable) >= 2:
            usable_axis = canonical_axis[usable]
            along_span = float(np.ptp(usable_axis[:, 0]))
            cross_span = float(np.ptp(usable_axis[:, 1]))
            barre_oriented = (
                cross_span >= BARRE_MIN_CROSS_NECK_SPAN
                and cross_span >= BARRE_MIN_CROSS_TO_ALONG_RATIO * max(along_span, 1e-6)
            )
            if barre_oriented:
                axis_x = float(np.median(usable_axis[:, 0]))
                if _fret_cell_from_canonical_x(axis_x, cfg, fret_centers) is not None:
                    contact_x = axis_x

    classified = _fret_cell_from_canonical_x(
        contact_x,
        cfg,
        fret_centers,
        deadband_fraction=(FRET_WIRE_DEADBAND_FRACTION if barre_oriented else 0.0),
    )
    return None if classified is None else float(classified[0])


def compute_index_fret_raw(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    *,
    fallback: float | None = None,
) -> float | None:
    """Return the old continuous fingertip coordinate for diagnostics only."""
    if hand is None or homography.confidence <= 0.0:
        return None
    index = hand.fingers.get("index")
    if index is None:
        return fallback
    try:
        canonical_point = project_to_canonical(
            homography, np.asarray([index.tip_xy], dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return fallback
    if not _on_canonical_neck(canonical_point)[0]:
        return None
    positions, method = _fret_positions_from_canonical_x(
        canonical_point[:, 0], cfg, fret_centers
    )
    if not np.all(np.isfinite(positions)):
        return None
    cell_center_offset = 1.0 if method == "calibrated_fret_map" else 0.5
    physical_fret = positions[0] + cell_center_offset
    if not 0.5 <= physical_fret <= float(cfg.max_fret) + 0.5:
        return None
    return float(physical_fret)


def _frames_nearly_static(source: np.ndarray, target: np.ndarray) -> bool:
    """Conservatively allow an unaligned first lock on a texture-poor scene."""
    if source.shape != target.shape or source.ndim != 3:
        return False
    height, width = source.shape[:2]
    row_step = max(1, height // 120)
    column_step = max(1, width // 160)
    source_luma = np.mean(
        source[::row_step, ::column_step].astype(np.float32),
        axis=2,
    )
    target_luma = np.mean(
        target[::row_step, ::column_step].astype(np.float32),
        axis=2,
    )
    delta = target_luma - source_luma
    # Ignore a uniform exposure change; spatial residual catches camera/board
    # motion without requiring the eight corners needed by LK alignment.
    residual = delta - float(np.median(delta))
    return bool(
        float(np.mean(np.abs(residual))) <= 8.0
        and float(np.percentile(np.abs(residual), 95)) <= 24.0
    )


class DetectionChain:
    """Tracked fretboard plus per-frame multi-finger position inference."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        hand_extractor: HandExtractor | None = None,
        guitar_config: GuitarConfig | None = None,
        detector_hz: float = 2.0,
        hand_detector_hz: float = 10.0,
        min_lock_confidence: float = 0.2,
        calibrator: BoardCalibrator = calibrate_board,
        background_detector: bool = False,
        crop_hand: bool = True,
        board_tracker: OpticalBoardTracker | None = None,
        hand_tracker: TemporalHandLandmarkTracker | None = None,
    ) -> None:
        if detector_hz <= 0.0:
            raise ValueError("detector_hz must be positive")
        if hand_detector_hz <= 0.0:
            raise ValueError("hand_detector_hz must be positive")
        self.detector = detector or YoloOBBBackend()
        self.hand_extractor = hand_extractor or MediaPipeHandExtractor()
        self.guitar_config = guitar_config or GuitarConfig()
        self.detector_interval_s = 1.0 / detector_hz
        self.hand_detector_interval_s = 1.0 / hand_detector_hz
        self._base_hand_detector_interval_ns = max(
            1,
            round(1_000_000_000.0 / hand_detector_hz),
        )
        self._hand_detector_interval_ns = self._base_hand_detector_interval_ns
        self.min_lock_confidence = min_lock_confidence
        self.calibrator = calibrator
        self.background_detector = background_detector
        self.crop_hand = crop_hand
        self.board_tracker = board_tracker or OpticalBoardTracker()
        self.hand_tracker = hand_tracker or TemporalHandLandmarkTracker()
        self._lock = RLock()
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="fretcam-yolo")
            if background_detector
            else None
        )
        self._detector_future: Future[_DetectorJobResult] | None = None
        self._closed = False
        self._generation = 0
        self._last_detection_s: float | None = None
        self._fret_centers: np.ndarray | None = None
        self._detector_fret_centers: np.ndarray | None = None
        self._geometry_fret_centers: np.ndarray | None = None
        self._position_scale = 1.0
        self._position_offset = 0.0
        self._pending_body_joint_fret: int | None = None
        self._pending_body_joint_count = 0
        self._geometry_refinement: GeometryRefinement | None = None
        self._geometry_correction = np.eye(3, dtype=np.float64)
        self._last_geometry_refine_s: float | None = None
        self._local_geometry_stability = 1.0
        self._roi_misses = 0
        self._roi_cooldown_frames = 0
        self._crop_ready = False
        self._full_hand_streak = 0
        self._last_hand_bounds: tuple[float, float, float, float] | None = None
        self._last_hand_timestamp_s: float | None = None
        self._hand_search_phase = 0
        self._force_full_hand_frame = False
        self._last_hand_source = "full"
        self._last_tracking_template: HandObservation | None = None
        self._last_hand_detection_s: float | None = None
        self._next_hand_detection_ns: int | None = None
        self._last_hand_frame_ns: int | None = None
        self._last_hand_frame_shape: tuple[int, int] | None = None
        self._last_full_hand_video_s: float | None = None
        self._last_finger_tips: dict[str, Point] = {}
        self._last_finger_motion_s: float | None = None
        self._hand_schedule_mode = "default"
        self._force_hand_detection = False
        self._force_neck_search = False
        self._hand_refresh_reason = "bootstrap"
        self._last_hand_tracking_quality = 0.0
        self._last_hand_pose_quality = 0.0
        self._last_hint_recovery = False
        self._frame_hand_detector_calls = 0
        self._frame_hand_search_attempts: list[str] = []
        self._board_detector_pending_for_hand = False
        self._last_detector_request_s: float | None = None

    def reset(self) -> None:
        """Clear all source state, including the MediaPipe runtime."""
        with self._lock:
            resetter = getattr(self.hand_extractor, "reset", None)
            if resetter is not None:
                resetter()
            self._reset_tracking_locked()

    def reset_tracking(self, *, reset_hand_runtime: bool = False) -> None:
        """Clear geometry/ROI state while retaining a warmed hand model."""
        with self._lock:
            if reset_hand_runtime:
                reset_video = getattr(
                    self.hand_extractor,
                    "reset_video_tracking",
                    None,
                )
                if reset_video is not None:
                    reset_video()
            self._reset_tracking_locked()

    def _reset_tracking_locked(self) -> None:
        self._generation += 1
        self._retire_future()
        self._last_detection_s = None
        self._fret_centers = None
        self._detector_fret_centers = None
        self._geometry_fret_centers = None
        self._position_scale = 1.0
        self._position_offset = 0.0
        self._pending_body_joint_fret = None
        self._pending_body_joint_count = 0
        self._geometry_refinement = None
        self._geometry_correction = np.eye(3, dtype=np.float64)
        self._last_geometry_refine_s = None
        self._local_geometry_stability = 1.0
        self._roi_misses = 0
        self._roi_cooldown_frames = 0
        self._crop_ready = False
        self._full_hand_streak = 0
        self._last_hand_bounds = None
        self._last_hand_timestamp_s = None
        self._hand_search_phase = 0
        self._force_full_hand_frame = False
        self._last_hand_source = "full"
        self._last_tracking_template = None
        self._last_hand_detection_s = None
        self._next_hand_detection_ns = None
        self._last_hand_frame_ns = None
        self._last_hand_frame_shape = None
        self._last_full_hand_video_s = None
        self._last_finger_tips = {}
        self._last_finger_motion_s = None
        self._hand_schedule_mode = "default"
        self._hand_detector_interval_ns = self._base_hand_detector_interval_ns
        self._force_hand_detection = False
        self._force_neck_search = False
        self._hand_refresh_reason = "bootstrap"
        self._last_hand_tracking_quality = 0.0
        self._last_hand_pose_quality = 0.0
        self._last_hint_recovery = False
        self._frame_hand_detector_calls = 0
        self._frame_hand_search_attempts = []
        self._board_detector_pending_for_hand = False
        self._last_detector_request_s = None
        self.hand_tracker.reset()
        self.board_tracker.reset()

    def wait_for_background_detector(self) -> None:
        """Wait for an already-submitted warmup job outside the live path."""
        with self._lock:
            self._consume_future(wait=True)

    def set_position_calibration(self, *, scale: float, offset: float) -> None:
        """Apply session calibration to the physical fret map, not just the HUD."""
        if (
            not math.isfinite(scale)
            or not 0.65 <= scale <= 1.35
            or not math.isfinite(offset)
            or abs(offset) > 1.5
        ):
            raise ValueError("position calibration is outside safe bounds")
        with self._lock:
            self._position_scale = float(scale)
            self._position_offset = float(offset)
            self._refresh_fret_centers()

    def _refresh_fret_centers(self) -> None:
        base = (
            self._geometry_fret_centers
            if self._geometry_fret_centers is not None
            else self._detector_fret_centers
        )
        calibrated = (
            abs(self._position_scale - 1.0) > 1e-9 or abs(self._position_offset) > 1e-9
        )
        if base is None and calibrated:
            base = _rule18_fret_centers(self.guitar_config)
        if base is None:
            self._fret_centers = None
            return
        self._fret_centers = _apply_fret_axis_calibration(
            base,
            scale=self._position_scale,
            offset=self._position_offset,
        )

    def set_player_handedness(self, value: str) -> None:
        with self._lock:
            if value not in {"right", "left"}:
                raise ValueError("player_handedness must be 'right' or 'left'")
            setter = getattr(self.hand_extractor, "set_player_handedness", None)
            if setter is None:
                return
            if getattr(self.hand_extractor, "player_handedness", None) == value:
                return
            setter(value)
            # Handedness changes which detected hand is selected, not the board
            # geometry or the loaded MediaPipe model. Force one full-frame hand
            # pass without discarding either of those expensive states.
            self._roi_misses = 0
            self._roi_cooldown_frames = 0
            self._crop_ready = False
            self._full_hand_streak = 0
            self._last_hand_bounds = None
            self._last_hand_timestamp_s = None
            self._hand_search_phase = 0
            self._force_full_hand_frame = True
            self._last_hand_source = "full"
            self._last_tracking_template = None
            self._last_hand_detection_s = None
            self._next_hand_detection_ns = None
            self._last_hand_frame_ns = None
            self._last_full_hand_video_s = None
            self._last_finger_tips = {}
            self._last_finger_motion_s = None
            self._hand_schedule_mode = "acquiring"
            self._set_hand_schedule_interval(
                round(1_000_000_000.0 / ACTIVE_HAND_DETECTOR_HZ)
            )
            self._force_hand_detection = True
            self._force_neck_search = True
            self._hand_refresh_reason = "settings_changed"
            self._last_hand_tracking_quality = 0.0
            self._last_hand_pose_quality = 0.0
            self._last_hint_recovery = True
            self.hand_tracker.reset()

    def set_hand_search_hint(self, hint: HandSearchHint) -> None:
        """Adapt MediaPipe cadence and crop breadth from the previous estimate."""
        with self._lock:
            acquisition_blockers = {"no_hand", "off_neck"}
            blocker_recovery = bool(acquisition_blockers.intersection(hint.blockers))
            pose_low = 0.0 < hint.pose_quality < LOW_HAND_TRACKING_QUALITY
            landmark_low = (
                hint.hand_visible and hint.landmark_quality < LOW_HAND_TRACKING_QUALITY
            )
            recovery = bool(
                not hint.hand_visible
                or blocker_recovery
                or landmark_low
                or pose_low
                or hint.position_state in {"holding", "lost"}
            )
            if hint.position_state == "shifting":
                mode = "shifting"
                interval_ns = round(1_000_000_000.0 / ACTIVE_HAND_DETECTOR_HZ)
            elif recovery:
                mode = "recovering"
                interval_ns = round(1_000_000_000.0 / ACTIVE_HAND_DETECTOR_HZ)
            elif hint.position_state == "locked":
                mode = "locked"
                interval_ns = round(1_000_000_000.0 / LOCKED_HAND_DETECTOR_HZ)
            else:
                mode = "acquiring"
                interval_ns = round(1_000_000_000.0 / ACTIVE_HAND_DETECTOR_HZ)

            previous_mode = self._hand_schedule_mode
            self._hand_schedule_mode = mode
            self._set_hand_schedule_interval(interval_ns)
            newly_unstable = recovery and not self._last_hint_recovery
            entered_active_mode = mode in {"shifting", "recovering"} and (
                previous_mode != mode
            )
            if newly_unstable or entered_active_mode:
                self._force_hand_detection = True
                self._force_neck_search = True
                self._hand_refresh_reason = (
                    "quality_drop" if newly_unstable else "position_shift"
                )
            self._last_hint_recovery = recovery

    def _set_hand_schedule_interval(self, interval_ns: int) -> None:
        new_interval = max(1, int(interval_ns))
        old_interval = self._hand_detector_interval_ns
        if new_interval == old_interval:
            return
        self._hand_detector_interval_ns = new_interval
        if self._last_hand_detection_s is None:
            self._next_hand_detection_ns = None
            return
        candidate = round(self._last_hand_detection_s * 1_000_000_000.0) + new_interval
        if self._next_hand_detection_ns is None:
            self._next_hand_detection_ns = candidate
        elif new_interval < old_interval:
            self._next_hand_detection_ns = min(
                self._next_hand_detection_ns,
                candidate,
            )
        else:
            self._next_hand_detection_ns = max(
                self._next_hand_detection_ns,
                candidate,
            )

    def _rescale_hand_state_for_frame(self, frame_shape: tuple[int, ...]) -> None:
        """Keep chain-level crop and stillness state aligned across size changes."""
        height, width = frame_shape[:2]
        previous = self._last_hand_frame_shape
        if previous is not None and previous != (height, width):
            old_height, old_width = previous
            if old_height > 0 and old_width > 0 and height > 0 and width > 0:
                scale_x = width / old_width
                scale_y = height / old_height
                if self._last_hand_bounds is not None:
                    left, top, right, bottom = self._last_hand_bounds
                    self._last_hand_bounds = (
                        left * scale_x,
                        top * scale_y,
                        right * scale_x,
                        bottom * scale_y,
                    )
                self._last_finger_tips = {
                    name: (point[0] * scale_x, point[1] * scale_y)
                    for name, point in self._last_finger_tips.items()
                }
        self._last_hand_frame_shape = (height, width)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._consume_future(wait=True)
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self.hand_extractor.close()

    def __enter__(self) -> DetectionChain:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float | None = None,
    ) -> FrameDetection:
        with self._lock:
            return self._process_frame_locked(frame, timestamp_s=timestamp_s)

    def _process_frame_locked(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float | None,
    ) -> FrameDetection:
        if self._closed:
            raise RuntimeError("detection chain is closed")
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"expected BGR frame with shape (H, W, 3), got {frame.shape}"
            )
        now_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(now_s):
            raise ValueError("timestamp_s must be finite")

        self._rescale_hand_state_for_frame(frame.shape)
        total_started = time.perf_counter()
        snapshot = self.board_tracker.advance(frame, timestamp_s=now_s)
        detector_ms = 0.0
        homography_ms = 0.0
        detector_ran = False
        detector_requested = False
        detector_result_consumed = False
        detector_result_accepted = False

        if self.background_detector:
            result = self._consume_future(wait=False)
            if result is not None:
                detector_result_consumed = True
                if result.generation == self._generation:
                    applied = self._apply_detector_result(result, frame, now_s)
                    detector_ran = applied
                    detector_result_accepted = applied
                    detector_ms = result.detector_ms
                    homography_ms = result.homography_ms
                    if applied:
                        self._last_detection_s = now_s
            snapshot = self.board_tracker.snapshot(now_s)
            if self._should_detect(now_s, snapshot) and self._detector_future is None:
                assert self._executor is not None
                detector_requested = True
                self._detector_future = self._executor.submit(
                    self._run_detector_job,
                    frame.copy(),
                    now_s,
                    self._generation,
                )
                self._last_detection_s = now_s
                self._last_detector_request_s = now_s
        elif self._should_detect(now_s, snapshot):
            detector_requested = True
            result = self._run_detector_job(frame.copy(), now_s, self._generation)
            detector_result_consumed = True
            detector_ran = self._apply_detector_result(result, frame, now_s)
            detector_result_accepted = detector_ran
            detector_ms = result.detector_ms
            homography_ms = result.homography_ms
            self._last_detection_s = now_s
            self._last_detector_request_s = now_s

        snapshot = self.board_tracker.snapshot(now_s)
        homography = snapshot.homography
        locked = (
            homography.confidence >= self.min_lock_confidence
            and snapshot.status != "missing"
        )
        if locked:
            refinement_started = time.perf_counter()
            homography = self._refined_homography(frame, snapshot, now_s)
            homography_ms += (time.perf_counter() - refinement_started) * 1000.0

        hand_started = time.perf_counter()
        self._frame_hand_detector_calls = 0
        self._frame_hand_search_attempts = []
        self._board_detector_pending_for_hand = self._detector_future is not None
        previous_hand_tracking_quality = self._last_hand_tracking_quality
        previous_hand_pose_quality = self._last_hand_pose_quality
        now_ns = round(now_s * 1_000_000_000.0)
        hand_clock_reset = (
            self._last_hand_frame_ns is not None and now_ns <= self._last_hand_frame_ns
        )
        if hand_clock_reset:
            self.hand_tracker.reset()
            self._last_tracking_template = None
            self._last_hand_detection_s = None
            self._next_hand_detection_ns = None
            self._last_finger_tips = {}
            self._last_finger_motion_s = None
            self._hand_refresh_reason = "clock_reset"
        should_update_hand = locked and (
            self._force_hand_detection
            or self._next_hand_detection_ns is None
            or now_ns >= self._next_hand_detection_ns
        )
        if should_update_hand:
            forced_refresh = self._force_hand_detection
            scheduled_deadline_ns = (
                None if forced_refresh else self._next_hand_detection_ns
            )
            if forced_refresh:
                self._force_hand_detection = False
            elif self._next_hand_detection_ns is None and not hand_clock_reset:
                self._hand_refresh_reason = "bootstrap"
            else:
                if not hand_clock_reset:
                    self._hand_refresh_reason = "deadline"
            extracted_hand = self._extract_hand(
                frame,
                homography,
                locked,
                timestamp_s=now_s,
            )
            hand_search_source = self._last_hand_source
            self._last_hand_detection_s = now_s
            if scheduled_deadline_ns is None or now_ns < scheduled_deadline_ns:
                self._next_hand_detection_ns = now_ns + self._hand_detector_interval_ns
            else:
                elapsed_slots = (
                    now_ns - scheduled_deadline_ns
                ) // self._hand_detector_interval_ns
                self._next_hand_detection_ns = (
                    scheduled_deadline_ns
                    + (elapsed_slots + 1) * self._hand_detector_interval_ns
                )
        else:
            extracted_hand = None
            hand_search_source = "board_pending" if not locked else "temporal_only"
            self._hand_refresh_reason = "board_pending" if not locked else "not_due"
        extracted_hand = self._apply_temporal_hand_tracking(
            frame,
            extracted_hand,
            timestamp_s=now_s,
        )
        if isinstance(extracted_hand, HandObservation):
            extracted_hand = self._annotate_hand_motion(
                extracted_hand,
                timestamp_s=now_s,
            )
        self._last_hand_frame_ns = now_ns
        hand_ms = (time.perf_counter() - hand_started) * 1000.0
        if isinstance(extracted_hand, HandObservation):
            hand = extracted_hand.hand
            finger_axes_xy = extracted_hand.finger_axes_xy
            finger_quality = extracted_hand.finger_quality
            finger_stillness = extracted_hand.finger_stillness
            hand_tracking_quality = float(extracted_hand.hand.confidence)
            pose_quality = extracted_hand.pose_quality
            pose_continuity = extracted_hand.pose_continuity
            pose_identity_score = extracted_hand.pose_identity_score
            pose_residual_fraction = extracted_hand.pose_residual_fraction
            pose_predicted = extracted_hand.pose_predicted
        else:
            hand = extracted_hand
            finger_axes_xy = {}
            finger_quality = {}
            finger_stillness = {}
            hand_tracking_quality = (
                float(extracted_hand.confidence) if extracted_hand is not None else 0.0
            )
            pose_quality = 0.0
            pose_continuity = 0.0
            pose_identity_score = 0.0
            pose_residual_fraction = 1.0
            pose_predicted = False
        hand_source = (
            extracted_hand.source
            if isinstance(extracted_hand, HandObservation)
            else self._last_hand_source
            if extracted_hand is not None
            else "none"
        )
        self._last_hand_tracking_quality = hand_tracking_quality
        self._last_hand_pose_quality = pose_quality
        if should_update_hand and (
            (
                extracted_hand is None
                and previous_hand_tracking_quality >= LOW_HAND_TRACKING_QUALITY
            )
            or (
                isinstance(extracted_hand, HandObservation)
                and (
                    (
                        previous_hand_tracking_quality >= LOW_HAND_TRACKING_QUALITY
                        and hand_tracking_quality < LOW_HAND_TRACKING_QUALITY
                    )
                    or (
                        previous_hand_pose_quality >= LOW_HAND_TRACKING_QUALITY
                        and 0.0 < pose_quality < LOW_HAND_TRACKING_QUALITY
                    )
                )
            )
        ):
            self._force_hand_detection = True
            self._force_neck_search = locked
            if self._hand_refresh_reason not in {"quality_drop", "position_shift"}:
                self._hand_refresh_reason = "detector_miss"

        anchor_started = time.perf_counter()
        if not _hand_overlaps_neck(hand, homography):
            hand = None
            finger_axes_xy = {}
            finger_quality = {}
            self._crop_ready = False
            self._full_hand_streak = 0
            if (
                self._last_hand_timestamp_s is None
                or now_s - self._last_hand_timestamp_s > 0.20
            ):
                self._last_hand_bounds = None
        else:
            assert hand is not None
            self._last_hand_bounds = _hand_bounds(hand)
            self._last_hand_timestamp_s = now_s
            self._crop_ready = True
        anchor = compute_position_anchor(
            hand,
            homography,
            self.guitar_config,
            self._fret_centers,
        )
        index_fret = compute_index_fret(
            hand,
            homography,
            self.guitar_config,
            self._fret_centers,
            index_axis_xy=finger_axes_xy.get("index", ()),
            fallback=anchor.center_fret if anchor.confidence > 0.0 else None,
        )
        index_fret_raw = compute_index_fret_raw(
            hand,
            homography,
            self.guitar_config,
            self._fret_centers,
            fallback=anchor.center_fret if anchor.confidence > 0.0 else None,
        )
        freshness = self._geometry_freshness(snapshot)
        if self._fret_centers is None:
            freshness *= 0.8
        position_fret, contacts, confidence_factors = solve_hand_position(
            hand,
            homography,
            self.guitar_config,
            self._fret_centers,
            anchor,
            finger_axes_xy=finger_axes_xy,
            finger_quality=finger_quality,
            finger_stillness=finger_stillness,
            string_curves=(
                ()
                if self._geometry_refinement is None
                else self._geometry_refinement.string_curves
            ),
            freshness=freshness,
            geometry_stability=min(
                snapshot.stability,
                self._local_geometry_stability,
            ),
        )
        anchor_ms = (time.perf_counter() - anchor_started) * 1000.0

        latency = StageLatency(
            detector_ms=detector_ms,
            homography_ms=homography_ms,
            hand_ms=hand_ms,
            anchor_ms=anchor_ms,
            total_ms=(time.perf_counter() - total_started) * 1000.0,
        )
        return FrameDetection(
            timestamp_s=now_s,
            detector_ran=detector_ran,
            neck_locked=locked,
            fret_map_locked=self._fret_centers is not None,
            homography_confidence=float(homography.confidence),
            homography_method=homography.method,
            neck_quad=_neck_quad(homography) if locked else (),
            fret_ticks=(_fret_ticks(homography, self._fret_centers) if locked else ()),
            hand_points=_hand_points(hand),
            index_fret=index_fret,
            anchor=anchor,
            stage_latency=latency,
            index_fret_raw=index_fret_raw,
            composite_available=True,
            position_fret=position_fret,
            observation_confidence=confidence_factors.combined,
            confidence_factors=confidence_factors,
            finger_contacts=contacts,
            geometry_status=snapshot.status,
            geometry_age_ms=self._age_ms(snapshot.geometry_age_s),
            detector_age_ms=self._age_ms(snapshot.detector_age_s),
            geometry_stability=min(
                snapshot.stability,
                self._local_geometry_stability,
            ),
            hand_source=hand_source,
            fret_refinement_support=(
                0.0
                if self._geometry_refinement is None
                else self._geometry_refinement.fret_support
            ),
            string_refinement_support=(
                0.0
                if self._geometry_refinement is None
                else self._geometry_refinement.string_support
            ),
            geometry_distortion_residual=(
                0.0
                if self._geometry_refinement is None
                else self._geometry_refinement.distortion_residual
            ),
            nut_x=(
                None
                if self._geometry_refinement is None
                else self._geometry_refinement.nut_x
            ),
            body_joint_x=(
                None
                if self._geometry_refinement is None
                else self._geometry_refinement.body_x
            ),
            boundary_support=(
                0.0
                if self._geometry_refinement is None
                else self._geometry_refinement.boundary_support
            ),
            body_joint_fret=(
                None
                if self._geometry_refinement is None
                else self._geometry_refinement.body_joint_fret
            ),
            detector_requested=detector_requested,
            detector_pending=self._detector_future is not None,
            hand_detector_ran=self._frame_hand_detector_calls > 0,
            hand_schedule_mode=self._hand_schedule_mode,
            hand_detector_interval_ms=round(
                self._hand_detector_interval_ns / 1_000_000.0,
                3,
            ),
            hand_refresh_reason=self._hand_refresh_reason,
            hand_search_source=hand_search_source,
            hand_tracking_quality=hand_tracking_quality,
            hand_pose_quality=pose_quality,
            hand_pose_continuity=pose_continuity,
            hand_pose_identity_score=pose_identity_score,
            hand_pose_residual_fraction=pose_residual_fraction,
            hand_pose_predicted=pose_predicted,
            hand_detector_calls=self._frame_hand_detector_calls,
            hand_search_attempts=tuple(self._frame_hand_search_attempts),
            detector_result_consumed=detector_result_consumed,
            detector_result_accepted=detector_result_accepted,
            detector_request_timestamp_s=self._last_detector_request_s,
        )

    def _refined_homography(
        self,
        frame: np.ndarray,
        snapshot: TrackingSnapshot,
        timestamp_s: float,
    ) -> Homography:
        """Apply a conservative fret/string correction atop tracked motion."""
        base = snapshot.homography
        should_refine = (
            self._last_geometry_refine_s is None
            or timestamp_s < self._last_geometry_refine_s
            or timestamp_s - self._last_geometry_refine_s >= GEOMETRY_REFINE_INTERVAL_S
        )
        if should_refine:
            centers = self._detector_fret_centers
            if centers is None:
                wires, _ = _fret_wire_axis(self.guitar_config, None)
                centers = 0.5 * (wires[:-1] + wires[1:])
            refinement = refine_live_geometry(
                frame,
                base,
                centers,
                self.guitar_config,
            )
            if refinement.homography.method != base.method:
                try:
                    correction = np.linalg.inv(base.H) @ refinement.homography.H
                except np.linalg.LinAlgError:
                    correction = np.eye(3, dtype=np.float64)
                scale = float(correction[2, 2])
                if math.isfinite(scale) and abs(scale) > 1e-9:
                    correction = correction / scale
                if np.all(np.isfinite(correction)):
                    probe = np.asarray(
                        (
                            (0.0, 0.0),
                            (1.0, 0.0),
                            (1.0, 1.0),
                            (0.0, 1.0),
                            (0.5, 0.5),
                        ),
                        dtype=np.float64,
                    )
                    current_points = _project_canonical(
                        Homography(
                            H=self._geometry_correction,
                            confidence=1.0,
                            method="canonical_correction",
                        ),
                        probe,
                    )
                    candidate_points = _project_canonical(
                        Homography(
                            H=correction,
                            confidence=1.0,
                            method="canonical_correction",
                        ),
                        probe,
                    )
                    correction_jump = float(
                        np.max(
                            np.linalg.norm(
                                candidate_points - current_points,
                                axis=1,
                            )
                        )
                    )
                    was_identity = np.allclose(
                        self._geometry_correction,
                        np.eye(3, dtype=np.float64),
                        atol=1e-9,
                    )
                    maximum_jump = 0.030 if was_identity else 0.020
                    self._local_geometry_stability = float(
                        np.clip(
                            math.exp(-correction_jump / 0.020),
                            0.0,
                            1.0,
                        )
                    )
                    if correction_jump <= maximum_jump:
                        alpha = 0.50 if was_identity else 0.25
                        smoothed = (
                            1.0 - alpha
                        ) * self._geometry_correction + alpha * correction
                        smoothed_scale = float(smoothed[2, 2])
                        if math.isfinite(smoothed_scale) and abs(smoothed_scale) > 1e-9:
                            smoothed = smoothed / smoothed_scale
                        self._geometry_correction = smoothed
            elif refinement.string_support <= 0.0:
                # Keep the last conservative correction, but make confidence
                # honest when this refresh cannot establish non-crossing
                # string geometry.
                self._local_geometry_stability = min(
                    self._local_geometry_stability,
                    0.55,
                )
            else:
                self._local_geometry_stability = min(
                    1.0,
                    self._local_geometry_stability + 0.10,
                )
            trustworthy_axis = _live_fret_axis_is_trustworthy(
                refinement,
                self._detector_fret_centers,
                self.guitar_config,
            )
            if trustworthy_axis:
                assert refinement.fret_centers is not None
                assert refinement.body_joint_fret is not None
                if refinement.body_joint_fret == self._pending_body_joint_fret:
                    self._pending_body_joint_count += 1
                else:
                    self._pending_body_joint_fret = refinement.body_joint_fret
                    self._pending_body_joint_count = 1
                candidate_axis = _transform_fret_axis(
                    np.asarray(
                        refinement.fret_centers,
                        dtype=np.float64,
                    ),
                    self._geometry_correction,
                )
                if self._pending_body_joint_count >= LIVE_FRET_AXIS_CONFIRMATIONS:
                    if self._geometry_fret_centers is None:
                        self._geometry_fret_centers = candidate_axis
                    else:
                        self._geometry_fret_centers = (
                            0.75 * self._geometry_fret_centers + 0.25 * candidate_axis
                        )
                    self._refresh_fret_centers()
            else:
                # Weak/ambiguous image evidence may update diagnostics, but it
                # must not accumulate confirmations toward replacing the
                # detector or the last trusted nonlinear axis.
                self._pending_body_joint_fret = None
                self._pending_body_joint_count = 0
            boundary_xs = None
            if refinement.nut_x is not None and refinement.body_x is not None:
                boundary_xs = _transform_fret_axis(
                    np.asarray(
                        (refinement.nut_x, refinement.body_x),
                        dtype=np.float64,
                    ),
                    self._geometry_correction,
                )
            self._geometry_refinement = GeometryRefinement(
                homography=refinement.homography,
                fret_centers=self._fret_centers,
                string_curves=transform_string_curves(
                    refinement.string_curves,
                    self._geometry_correction,
                ),
                fret_support=refinement.fret_support,
                string_support=refinement.string_support,
                nut_x=(
                    refinement.nut_x if boundary_xs is None else float(boundary_xs[0])
                ),
                body_x=(
                    refinement.body_x if boundary_xs is None else float(boundary_xs[1])
                ),
                distortion_residual=refinement.distortion_residual,
                boundary_support=refinement.boundary_support,
                body_joint_fret=refinement.body_joint_fret,
            )
            self._last_geometry_refine_s = timestamp_s

        matrix = base.H @ self._geometry_correction
        scale = float(matrix[2, 2])
        if math.isfinite(scale) and abs(scale) > 1e-9:
            matrix = matrix / scale
        corrected = not np.allclose(
            self._geometry_correction,
            np.eye(3, dtype=np.float64),
            atol=1e-9,
        )
        return Homography(
            H=np.asarray(matrix, dtype=np.float64),
            confidence=base.confidence,
            method=(
                f"{base.method}+local_fret_string"
                if corrected and "local_fret_string" not in base.method
                else base.method
            ),
        )

    def _apply_temporal_hand_tracking(
        self,
        frame: np.ndarray,
        extracted: HandObservation | HandSample | None,
        *,
        timestamp_s: float,
    ) -> HandObservation | HandSample | None:
        """Smooth raw MediaPipe joints and bridge very short detector gaps."""
        detector_observation: LandmarkObservation | None = None
        candidate_template: HandObservation | None = None
        if isinstance(extracted, HandObservation):
            if (
                len(extracted.landmarks_xy) == 21
                and len(extracted.landmarks_z) == 21
                and len(extracted.joint_quality) == 21
            ):
                detector_observation = LandmarkObservation(
                    landmarks_xy=np.asarray(
                        extracted.landmarks_xy,
                        dtype=np.float64,
                    ),
                    landmarks_z=np.asarray(
                        extracted.landmarks_z,
                        dtype=np.float64,
                    ),
                    confidence=float(np.clip(extracted.hand.confidence, 0.0, 1.0)),
                    joint_quality=np.asarray(
                        extracted.joint_quality,
                        dtype=np.float64,
                    ),
                    is_left_hand=extracted.hand.is_left_hand,
                )
                candidate_template = extracted
            else:
                self.hand_tracker.reset()
                self._last_tracking_template = None
                return extracted
        elif extracted is not None:
            # Deterministic test doubles and alternate extractors do not expose
            # 21 joints; retain their legacy behavior without fabricating data.
            self.hand_tracker.reset()
            self._last_tracking_template = None
            return extracted
        elif self._last_tracking_template is None:
            return None

        try:
            tracked = self.hand_tracker.update(
                frame,
                timestamp_s=timestamp_s,
                observation=detector_observation,
            )
        except ValueError:
            # A new clip may restart its source clock without rebuilding the
            # WebSocket processor. Reset only temporal hand state and retry.
            self.hand_tracker.reset()
            tracked = self.hand_tracker.update(
                frame,
                timestamp_s=timestamp_s,
                observation=detector_observation,
            )
        if tracked is None:
            if detector_observation is None:
                self._last_tracking_template = None
                self._last_finger_tips = {}
                self._last_finger_motion_s = None
            return None
        if tracked.detector_observation_accepted and candidate_template is not None:
            self._last_tracking_template = candidate_template
        assert self._last_tracking_template is not None
        return _tracked_hand_observation(
            tracked,
            self._last_tracking_template,
        )

    def _annotate_hand_motion(
        self,
        observation: HandObservation,
        *,
        timestamp_s: float,
    ) -> HandObservation:
        """Attach a bounded per-finger stillness score for contact evidence."""
        tips = {
            name: finger.tip_xy
            for name, finger in observation.hand.fingers.items()
            if name in FRETTING_FINGERS
        }
        bounds = _hand_bounds(observation.hand)
        hand_scale = max(
            12.0,
            math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]),
        )
        elapsed = (
            None
            if self._last_finger_motion_s is None
            else timestamp_s - self._last_finger_motion_s
        )
        stillness: dict[str, float] = {}
        for name, tip in tips.items():
            previous = self._last_finger_tips.get(name)
            if previous is None or elapsed is None or elapsed <= 0.0:
                score = 1.0
            else:
                speed = math.dist(previous, tip) / hand_scale / elapsed
                score = math.exp(-speed / 6.0)
            stillness[name] = float(np.clip(score, 0.0, 1.0))
        self._last_finger_tips = dict(tips)
        self._last_finger_motion_s = timestamp_s
        return HandObservation(
            hand=observation.hand,
            finger_axes_xy=observation.finger_axes_xy,
            finger_quality=observation.finger_quality,
            handedness_label=observation.handedness_label,
            handedness_score=observation.handedness_score,
            source=observation.source,
            landmarks_xy=observation.landmarks_xy,
            landmarks_z=observation.landmarks_z,
            joint_quality=observation.joint_quality,
            finger_stillness=stillness,
            pose_quality=observation.pose_quality,
            pose_continuity=observation.pose_continuity,
            pose_identity_score=observation.pose_identity_score,
            pose_residual_fraction=observation.pose_residual_fraction,
            pose_predicted=observation.pose_predicted,
            detector_observation_accepted=observation.detector_observation_accepted,
        )

    def _extract_hand(
        self,
        frame: np.ndarray,
        homography: Homography,
        locked: bool,
        *,
        timestamp_s: float,
    ) -> HandObservation | HandSample | None:
        self._last_hand_source = "full"
        if self._force_full_hand_frame:
            self._force_full_hand_frame = False
            self._force_neck_search = False
            self._last_full_hand_video_s = timestamp_s
            candidates = self._extract_hand_candidates(
                frame,
                timestamp_s=timestamp_s,
                crop_rect=None,
                source="full",
            )
            return self._select_neck_hand(
                candidates,
                homography,
                timestamp_s=timestamp_s,
            )
        if not self.crop_hand or not locked:
            self._force_neck_search = False
            self._last_full_hand_video_s = timestamp_s
            candidates = self._extract_hand_candidates(
                frame,
                timestamp_s=timestamp_s,
                crop_rect=None,
                source="full",
            )
            return self._select_neck_hand(
                candidates,
                homography,
                timestamp_s=timestamp_s,
            )

        neck_quad = _neck_quad(homography)
        if self._force_neck_search:
            self._force_neck_search = False
            neck_crop = _hand_crop_rect(frame, neck_quad)
            if neck_crop is None:
                self._force_full_hand_frame = True
                self._force_hand_detection = True
                self._last_hand_source = "full_recovery_pending"
                return None
            candidates = self._extract_hand_candidates(
                frame,
                timestamp_s=timestamp_s,
                crop_rect=neck_crop,
                source="neck_recovery",
            )
            selected = self._select_neck_hand(
                candidates,
                homography,
                timestamp_s=timestamp_s,
            )
            self._last_hand_source = "neck_recovery"
            if selected is None:
                self._force_full_hand_frame = True
                self._force_hand_detection = True
            else:
                self._roi_misses = 0
            return selected

        if self._last_hand_bounds is not None:
            full_video_candidates: tuple[HandObservation | HandSample, ...] = ()
            full_refresh_ran = False
            if not self._board_detector_pending_for_hand and (
                self._last_full_hand_video_s is None
                or timestamp_s < self._last_full_hand_video_s
                or timestamp_s - self._last_full_hand_video_s
                >= FULL_HAND_VIDEO_REFRESH_INTERVAL_S
            ):
                full_refresh_ran = True
                full_video_candidates = self._extract_hand_candidates(
                    frame,
                    timestamp_s=timestamp_s,
                    crop_rect=None,
                    source="full_video_refresh",
                )
                self._last_full_hand_video_s = timestamp_s
            narrow = _last_hand_crop_rect(
                frame,
                self._last_hand_bounds,
                neck_quad,
            )
            narrow_candidates: tuple[HandObservation | HandSample, ...] = ()
            if narrow is not None:
                narrow_candidates = self._extract_hand_candidates(
                    frame,
                    timestamp_s=timestamp_s,
                    crop_rect=narrow,
                    source="last_hand_crop",
                )
            if full_refresh_ran:
                selected = self._select_neck_hand(
                    full_video_candidates + narrow_candidates,
                    homography,
                    timestamp_s=timestamp_s,
                )
                if selected is not None:
                    selected_from_narrow = any(
                        selected is candidate for candidate in narrow_candidates
                    )
                    self._last_hand_source = (
                        "last_hand_crop"
                        if selected_from_narrow
                        else "full_video_refresh"
                    )
                    self._roi_misses = 0
                    return selected
                # A scheduled identity refresh plus one tight fallback already
                # used this frame's two-call budget. Widen on the next frame.
                self._force_neck_search = True
                self._force_hand_detection = True
                self._last_hand_source = "neck_recovery_pending"
                return None
            if narrow_candidates:
                selected = self._select_neck_hand(
                    narrow_candidates,
                    homography,
                    timestamp_s=timestamp_s,
                )
                if selected is not None:
                    self._last_hand_source = "last_hand_crop"
                    self._roi_misses = 0
                    return selected

            # A tight crop can miss a fast shift. Search the complete tracked
            # neck at a second scale before giving up the current frame.
            neck_crop = _hand_crop_rect(frame, neck_quad)
            if neck_crop is not None:
                candidates = self._extract_hand_candidates(
                    frame,
                    timestamp_s=timestamp_s,
                    crop_rect=neck_crop,
                    source="neck_crop",
                )
                selected = self._select_neck_hand(
                    candidates,
                    homography,
                    timestamp_s=timestamp_s,
                )
                if selected is not None:
                    self._last_hand_source = "neck_crop"
                    self._roi_misses = 0
                    return selected

            if narrow is None and neck_crop is None:
                candidates = self._extract_hand_candidates(
                    frame,
                    timestamp_s=timestamp_s,
                    crop_rect=None,
                    source="full_reacquire",
                )
                self._last_full_hand_video_s = timestamp_s
                selected = self._select_neck_hand(
                    candidates,
                    homography,
                    timestamp_s=timestamp_s,
                )
                if selected is not None:
                    self._last_hand_source = "full_reacquire"
                return selected

            self._roi_misses += 1
            if self._roi_misses < 2:
                return None
            self._roi_misses = 0
            self._force_full_hand_frame = True
            self._force_hand_detection = True
            self._last_hand_source = "full_recovery_pending"
            return None

        # Bootstrap from board geometry instead of waiting for a successful
        # full-frame hand. Alternate an upscaled full-neck search with the full
        # image to retain context and bound per-frame MediaPipe work.
        use_neck_crop = self._hand_search_phase % 2 == 0
        self._hand_search_phase += 1
        crop_rect = _hand_crop_rect(frame, neck_quad) if use_neck_crop else None
        source = "neck_search" if crop_rect is not None else "full_search"
        candidates = self._extract_hand_candidates(
            frame,
            timestamp_s=timestamp_s,
            crop_rect=crop_rect,
            source=source,
        )
        if crop_rect is None:
            self._last_full_hand_video_s = timestamp_s
        selected = self._select_neck_hand(
            candidates,
            homography,
            timestamp_s=timestamp_s,
        )
        if selected is not None:
            self._last_hand_source = source
            self._roi_misses = 0
            if self._last_full_hand_video_s is None:
                self._last_full_hand_video_s = timestamp_s
        return selected

    def _extract_hand_candidates(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float,
        crop_rect: tuple[int, int, int, int] | None,
        source: str,
    ) -> tuple[HandObservation | HandSample, ...]:
        if self._frame_hand_detector_calls >= 2:
            self._force_full_hand_frame = True
            self._force_hand_detection = True
            self._last_hand_source = "full_recovery_pending"
            return ()
        self._frame_hand_detector_calls += 1
        self._frame_hand_search_attempts.append(source)
        self._last_hand_source = source
        if crop_rect is None:
            return _extract_candidates(
                self.hand_extractor,
                frame,
                timestamp_s=timestamp_s,
            )
        crop, scale_x, scale_y = _prepare_hand_crop(frame, crop_rect)
        crop_width = max(1, crop_rect[2] - crop_rect[0])
        z_scale = crop_width / max(frame.shape[1], 1)
        candidates = _extract_candidates(
            self.hand_extractor,
            crop,
            timestamp_s=timestamp_s,
            use_video=False,
        )
        return tuple(
            _map_candidate_from_crop(
                candidate,
                crop_rect=crop_rect,
                scale_x=scale_x,
                scale_y=scale_y,
                z_scale=z_scale,
                source=source,
            )
            for candidate in candidates
        )

    def _select_neck_hand(
        self,
        candidates: tuple[HandObservation | HandSample, ...],
        homography: Homography,
        *,
        timestamp_s: float | None = None,
    ) -> HandObservation | HandSample | None:
        """Choose the most neck-consistent hand before using handedness."""
        if not candidates or homography.confidence <= 0.0:
            return None
        last_center: tuple[float, float] | None = None
        if self._last_hand_bounds is not None:
            left, top, right, bottom = self._last_hand_bounds
            last_center = ((left + right) / 2.0, (top + bottom) / 2.0)
        neck = np.asarray(_neck_quad(homography), dtype=np.float64)
        neck_scale = max(
            1.0,
            float(np.linalg.norm(neck.max(axis=0) - neck.min(axis=0))),
        )
        wanted_label = (
            "Right"
            if getattr(self.hand_extractor, "player_handedness", "right") == "right"
            else "Left"
        )
        scored: list[tuple[float, int, HandObservation | HandSample]] = []
        for index, candidate in enumerate(candidates):
            hand = _hand_sample(candidate)
            points = [
                hand.fingers[name].tip_xy
                for name in FRETTING_FINGERS
                if name in hand.fingers
            ]
            if not points:
                continue
            try:
                canonical = project_to_canonical(
                    homography,
                    np.asarray(points, dtype=np.float64),
                )
            except np.linalg.LinAlgError:
                continue
            on_neck_count = int(np.count_nonzero(_on_canonical_neck(canonical)))
            if on_neck_count < MIN_ON_NECK_FINGERTIPS:
                continue
            if _has_outward_longitudinal_wrist(hand, homography):
                continue
            overlap = on_neck_count / max(len(points), 1)
            center = np.mean(np.asarray(points, dtype=np.float64), axis=0)
            continuity = 0.5
            if last_center is not None:
                distance = float(
                    np.linalg.norm(center - np.asarray(last_center, dtype=np.float64))
                )
                continuity = math.exp(-0.5 * (distance / (0.25 * neck_scale)) ** 2)
            quality = float(hand.confidence)
            handedness_bonus = 0.0
            pose_identity = 0.5
            if isinstance(candidate, HandObservation):
                if candidate.finger_quality:
                    quality = 0.5 * quality + 0.5 * float(
                        np.mean(tuple(candidate.finger_quality.values()))
                    )
                if (
                    len(candidate.landmarks_xy) == 21
                    and len(candidate.landmarks_z) == 21
                    and len(candidate.joint_quality) == 21
                ):
                    pose_identity, pose_hard_reject = (
                        self.hand_tracker.match_observation_details(
                            LandmarkObservation(
                                landmarks_xy=np.asarray(
                                    candidate.landmarks_xy,
                                    dtype=np.float64,
                                ),
                                landmarks_z=np.asarray(
                                    candidate.landmarks_z,
                                    dtype=np.float64,
                                ),
                                confidence=float(
                                    np.clip(candidate.hand.confidence, 0.0, 1.0)
                                ),
                                joint_quality=np.asarray(
                                    candidate.joint_quality,
                                    dtype=np.float64,
                                ),
                                is_left_hand=candidate.hand.is_left_hand,
                            ),
                            timestamp_s=(
                                time.monotonic() if timestamp_s is None else timestamp_s
                            ),
                            frame_shape=self._last_hand_frame_shape,
                        )
                    )
                    if pose_hard_reject:
                        continue
                if candidate.handedness_label == wanted_label:
                    handedness_bonus = 0.05 * candidate.handedness_score
            score = (
                0.50 * overlap
                + 0.18 * quality
                + 0.12 * continuity
                + 0.20 * pose_identity
            )
            score += handedness_bonus
            scored.append((score, -index, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _run_detector_job(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        generation: int,
    ) -> _DetectorJobResult:
        detector_started = time.perf_counter()
        predictions = self.detector.predict_all(frame)
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        homography_started = time.perf_counter()
        homography, centers = self.calibrator(predictions, self.guitar_config)
        homography_ms = (time.perf_counter() - homography_started) * 1000.0
        return _DetectorJobResult(
            generation=generation,
            timestamp_s=timestamp_s,
            source_frame=frame,
            homography=homography,
            fret_centers=(
                None
                if centers is None
                else np.asarray(centers, dtype=np.float64).copy()
            ),
            detector_ms=detector_ms,
            homography_ms=homography_ms,
        )

    def _apply_detector_result(
        self,
        result: _DetectorJobResult,
        current_frame: np.ndarray,
        timestamp_s: float,
    ) -> bool:
        fresh = result.homography
        if abs(result.timestamp_s - timestamp_s) > 1e-9:
            aligned = align_detection_homography(
                result.source_frame,
                current_frame,
                fresh,
            )
            if aligned is None:
                initial_static_lock = (
                    self.board_tracker.homography.confidence <= 0.0
                    and _frames_nearly_static(result.source_frame, current_frame)
                )
                if not initial_static_lock:
                    return False
            else:
                fresh = aligned[0]
        continuous_refresh = _detector_refresh_is_continuous(
            self.board_tracker.homography,
            fresh,
            frame_shape=current_frame.shape,
        )
        accepted = self.board_tracker.accept_detection(
            current_frame,
            fresh,
            timestamp_s=timestamp_s,
        )
        if accepted:
            self._detector_fret_centers = (
                None if result.fret_centers is None else result.fret_centers.copy()
            )
            if not continuous_refresh:
                self._geometry_fret_centers = None
                self._pending_body_joint_fret = None
                self._pending_body_joint_count = 0
                self._geometry_refinement = None
                self._geometry_correction = np.eye(3, dtype=np.float64)
                self._local_geometry_stability = 1.0
            self._refresh_fret_centers()
            # Re-evaluate image evidence immediately against the fresh
            # homography. On a routine refresh, retain the already trusted
            # canonical axis/correction until that new evidence is accepted.
            self._last_geometry_refine_s = None
        return accepted

    def _consume_future(self, *, wait: bool) -> _DetectorJobResult | None:
        future = self._detector_future
        if future is None:
            return None
        if not wait and not future.done():
            return None
        self._detector_future = None
        try:
            return future.result()
        except Exception:
            return None

    def _retire_future(self) -> None:
        """Detach an obsolete detector job without blocking a live reset."""
        future = self._detector_future
        self._detector_future = None
        if future is None or future.cancel():
            return

        def discard_result(done: Future[_DetectorJobResult]) -> None:
            try:
                done.result()
            except Exception:
                return

        future.add_done_callback(discard_result)

    def _should_detect(
        self,
        timestamp_s: float,
        snapshot: TrackingSnapshot,
    ) -> bool:
        if self._last_detection_s is None:
            return True
        if timestamp_s < self._last_detection_s:
            return True
        interval = (
            min(self.detector_interval_s, 0.50)
            if snapshot.status in {"missing", "stale"}
            else self.detector_interval_s
        )
        return timestamp_s - self._last_detection_s >= interval

    def _geometry_freshness(self, snapshot: TrackingSnapshot) -> float:
        if snapshot.status in {"detected", "tracked"}:
            return 1.0
        if snapshot.status == "held":
            return max(
                0.0,
                1.0
                - snapshot.geometry_age_s / max(self.board_tracker.hard_expire_s, 1e-9),
            )
        if snapshot.status == "stale":
            return 0.10
        return 0.0

    @staticmethod
    def _age_ms(age_s: float) -> float:
        if not math.isfinite(age_s):
            return 1_000_000_000.0
        return round(max(0.0, age_s) * 1000.0, 3)


def process_frame(
    frame: np.ndarray,
    *,
    chain: DetectionChain,
    timestamp_s: float | None = None,
) -> FrameDetection:
    """Functional entry point around a caller-owned stateful chain."""
    return chain.process_frame(frame, timestamp_s=timestamp_s)


__all__ = [
    "ConfidenceFactors",
    "DetectionChain",
    "FALLBACK_BODY_JOINT_FRET",
    "FingerContact",
    "FrameDetection",
    "FretTick",
    "HandObservation",
    "HandPoint",
    "MediaPipeHandExtractor",
    "StageLatency",
    "compute_finger_contacts",
    "compute_index_fret",
    "compute_index_fret_raw",
    "compute_position_anchor",
    "process_frame",
    "solve_hand_position",
]
