"""F2 live-frame detection chain built from TabVision's vision library."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Protocol

import numpy as np

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
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS, HandSample
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
ROI_CROP_COOLDOWN_FRAMES = 10
FRET_WIRE_DEADBAND_FRACTION = 0.35
BARRE_INDEX_MIN_EXTENSION = 0.85
BARRE_MIN_CROSS_NECK_SPAN = 0.70
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
    """FretCam-only hand sample plus each fretting finger's joint axis."""

    hand: HandSample
    finger_axes_xy: dict[str, tuple[Point, ...]]

    @property
    def index_axis_xy(self) -> tuple[Point, ...]:
        return self.finger_axes_xy.get("index", ())


class HandExtractor(Protocol):
    def extract(self, frame: np.ndarray) -> HandObservation | HandSample | None: ...

    def close(self) -> None: ...


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

    def set_player_handedness(self, value: str) -> None:
        if value not in {"right", "left"}:
            raise ValueError("player_handedness must be 'right' or 'left'")
        self.player_handedness = value

    def extract(self, frame: np.ndarray) -> HandObservation | None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise BackendError(
                "opencv-python and mediapipe are required. Install with: "
                "pip install '.[vision]'."
            ) from exc

        landmarker = self.backend._load()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks or not result.handedness:
            return None

        selected = self._select_hand(result.hand_landmarks, result.handedness)
        landmarks = result.hand_landmarks[selected]
        handedness = result.handedness[selected]
        height, width = frame.shape[:2]
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
        return HandObservation(hand=hand, finger_axes_xy=finger_axes)

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
        self.backend.close()

    def reset(self) -> None:
        """Clear backend-local inference state at a new source boundary."""
        self.backend.close()


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
    """One finger's physical fret contact and pose-derived reliability."""

    name: str
    fret: int
    raw_fret: float
    weight: float
    curl_ratio: float
    pressing_score: float
    barre: bool = False


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
    return origin + scale * (1.0 - np.power(RULE_OF_18_RATIO, frets))


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
    return int(np.count_nonzero(_on_canonical_neck(canonical_points))) >= (
        MIN_ON_NECK_FINGERTIPS
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


def _barre_contact_x(
    hand: HandSample,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    axis_xy: tuple[Point, ...],
) -> tuple[float | None, bool]:
    index = hand.fingers.get("index")
    if (
        index is None
        or index.curl_ratio < BARRE_INDEX_MIN_EXTENSION
        or len(axis_xy) < 3
    ):
        return None, False
    try:
        canonical_axis = project_to_canonical(
            homography, np.asarray(axis_xy[-3:], dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return None, False
    usable = (
        np.all(np.isfinite(canonical_axis), axis=1)
        & (canonical_axis[:, 0] >= CANONICAL_NECK_MIN)
        & (canonical_axis[:, 0] <= CANONICAL_NECK_MAX)
    )
    if np.count_nonzero(usable) < 2:
        return None, False
    usable_axis = canonical_axis[usable]
    along_span = float(np.ptp(usable_axis[:, 0]))
    cross_span = float(np.ptp(usable_axis[:, 1]))
    barre = (
        cross_span >= BARRE_MIN_CROSS_NECK_SPAN
        and cross_span >= BARRE_MIN_CROSS_TO_ALONG_RATIO * max(along_span, 1e-6)
    )
    if not barre:
        return None, False
    axis_x = float(np.median(usable_axis[:, 0]))
    if _fret_cell_from_canonical_x(axis_x, cfg, fret_centers) is None:
        return None, False
    return axis_x, True


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
) -> tuple[FingerContact, ...]:
    """Classify all usable fretting-finger contacts on the physical fret axis."""
    if hand is None or homography.confidence <= 0.0:
        return ()
    axes = finger_axes_xy or {}
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

        contact_x = float(canonical_tip[0, 0])
        barre = False
        if name == "index":
            barre_x, barre = _barre_contact_x(
                hand,
                homography,
                cfg,
                fret_centers,
                axes.get("index", ()),
            )
            if barre_x is not None:
                contact_x = barre_x
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

        if barre:
            curl_score = 1.0
        else:
            curl_score = float(
                np.clip((0.96 - float(finger.curl_ratio)) / 0.18, 0.0, 1.0)
            )
        depth_delta = max(
            0.0,
            abs(float(finger.tip_z) - float(hand.wrist_z)) - 0.04,
        )
        depth_score = math.exp(-depth_delta / 0.08)
        pressing_score = curl_score * (0.85 + 0.15 * depth_score)
        weight = (
            FINGER_BASE_WEIGHTS[name]
            * float(np.clip(hand.confidence, 0.0, 1.0))
            * pressing_score
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
            )
        )
    return tuple(contacts)


def solve_hand_position(
    hand: HandSample | None,
    homography: Homography,
    cfg: GuitarConfig,
    fret_centers: np.ndarray | None,
    anchor: HandNeckAnchor,
    *,
    finger_axes_xy: dict[str, tuple[Point, ...]] | None = None,
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

    useful = [contact for contact in contacts if contact.weight >= 0.05]
    total_weight = sum(contact.weight for contact in useful)
    if not useful or total_weight <= 1e-9:
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
        if any(contact.barre and contact.weight >= 0.5 for contact in useful):
            support = max(support, 0.85)
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
) -> HandSample:
    translated_fingers = {
        name: type(finger)(
            name=finger.name,
            tip_xy=(
                float(finger.tip_xy[0]) + offset_x,
                float(finger.tip_xy[1]) + offset_y,
            ),
            tip_z=finger.tip_z,
            curl_ratio=finger.curl_ratio,
        )
        for name, finger in hand.fingers.items()
    }
    return HandSample(
        wrist_xy=(
            float(hand.wrist_xy[0]) + offset_x,
            float(hand.wrist_xy[1]) + offset_y,
        ),
        wrist_z=hand.wrist_z,
        is_left_hand=hand.is_left_hand,
        confidence=hand.confidence,
        fingers=translated_fingers,
    )


def _translate_hand_observation(
    observation: HandObservation | HandSample | None,
    *,
    offset_x: float,
    offset_y: float,
) -> HandObservation | HandSample | None:
    if observation is None:
        return None
    if isinstance(observation, HandObservation):
        return HandObservation(
            hand=_translate_hand_sample(
                observation.hand,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
            finger_axes_xy={
                name: tuple((x + offset_x, y + offset_y) for x, y in points)
                for name, points in observation.finger_axes_xy.items()
            },
        )
    return _translate_hand_sample(
        observation,
        offset_x=offset_x,
        offset_y=offset_y,
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
        min_lock_confidence: float = 0.2,
        calibrator: BoardCalibrator = calibrate_board,
        background_detector: bool = False,
        crop_hand: bool = True,
        board_tracker: OpticalBoardTracker | None = None,
    ) -> None:
        if detector_hz <= 0.0:
            raise ValueError("detector_hz must be positive")
        self.detector = detector or YoloOBBBackend()
        self.hand_extractor = hand_extractor or MediaPipeHandExtractor()
        self.guitar_config = guitar_config or GuitarConfig()
        self.detector_interval_s = 1.0 / detector_hz
        self.min_lock_confidence = min_lock_confidence
        self.calibrator = calibrator
        self.background_detector = background_detector
        self.crop_hand = crop_hand
        self.board_tracker = board_tracker or OpticalBoardTracker()
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
        self._roi_misses = 0
        self._roi_cooldown_frames = 0
        self._crop_ready = False
        self._full_hand_streak = 0
        self._last_hand_bounds: tuple[float, float, float, float] | None = None
        self._last_hand_source = "full"

    def reset(self) -> None:
        """Clear all source state, including the MediaPipe runtime."""
        with self._lock:
            resetter = getattr(self.hand_extractor, "reset", None)
            if resetter is not None:
                resetter()
            self._reset_tracking_locked()

    def reset_tracking(self) -> None:
        """Clear geometry/ROI state while retaining a warmed hand model."""
        with self._lock:
            self._reset_tracking_locked()

    def _reset_tracking_locked(self) -> None:
        self._generation += 1
        self._retire_future()
        self._last_detection_s = None
        self._fret_centers = None
        self._roi_misses = 0
        self._roi_cooldown_frames = 0
        self._crop_ready = False
        self._full_hand_streak = 0
        self._last_hand_bounds = None
        self._last_hand_source = "full"
        self.board_tracker.reset()

    def wait_for_background_detector(self) -> None:
        """Wait for an already-submitted warmup job outside the live path."""
        with self._lock:
            self._consume_future(wait=True)

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
            self._last_hand_source = "full"

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

        total_started = time.perf_counter()
        snapshot = self.board_tracker.advance(frame, timestamp_s=now_s)
        detector_ms = 0.0
        homography_ms = 0.0
        detector_ran = False

        if self.background_detector:
            result = self._consume_future(wait=False)
            if result is not None and result.generation == self._generation:
                applied = self._apply_detector_result(result, frame, now_s)
                detector_ran = applied
                detector_ms = result.detector_ms
                homography_ms = result.homography_ms
                if applied:
                    self._last_detection_s = now_s
            snapshot = self.board_tracker.snapshot(now_s)
            if self._should_detect(now_s, snapshot) and self._detector_future is None:
                assert self._executor is not None
                self._detector_future = self._executor.submit(
                    self._run_detector_job,
                    frame.copy(),
                    now_s,
                    self._generation,
                )
                self._last_detection_s = now_s
        elif self._should_detect(now_s, snapshot):
            result = self._run_detector_job(frame.copy(), now_s, self._generation)
            detector_ran = self._apply_detector_result(result, frame, now_s)
            detector_ms = result.detector_ms
            homography_ms = result.homography_ms
            self._last_detection_s = now_s

        snapshot = self.board_tracker.snapshot(now_s)
        homography = snapshot.homography
        locked = (
            homography.confidence >= self.min_lock_confidence
            and snapshot.status != "missing"
        )

        hand_started = time.perf_counter()
        extracted_hand = self._extract_hand(frame, homography, locked)
        hand_ms = (time.perf_counter() - hand_started) * 1000.0
        if isinstance(extracted_hand, HandObservation):
            hand = extracted_hand.hand
            finger_axes_xy = extracted_hand.finger_axes_xy
        else:
            hand = extracted_hand
            finger_axes_xy = {}

        anchor_started = time.perf_counter()
        if not _hand_overlaps_neck(hand, homography):
            hand = None
            finger_axes_xy = {}
            self._crop_ready = False
            self._full_hand_streak = 0
            self._last_hand_bounds = None
        else:
            assert hand is not None
            self._last_hand_bounds = _hand_bounds(hand)
            if self._last_hand_source == "full":
                self._full_hand_streak += 1
                self._crop_ready = self._full_hand_streak >= 2
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
            freshness=freshness,
            geometry_stability=snapshot.stability,
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
            geometry_stability=snapshot.stability,
        )

    def _extract_hand(
        self,
        frame: np.ndarray,
        homography: Homography,
        locked: bool,
    ) -> HandObservation | HandSample | None:
        crop_rect = None
        self._last_hand_source = "full"
        if self._roi_cooldown_frames > 0:
            self._roi_cooldown_frames -= 1
        elif self.crop_hand and locked and self._crop_ready:
            crop_rect = _hand_crop_rect(
                frame,
                _neck_quad(homography),
                hand_bounds=self._last_hand_bounds,
            )
        if crop_rect is None:
            observation = self.hand_extractor.extract(frame)
            if not locked:
                self._roi_misses = 0
                self._roi_cooldown_frames = 0
        else:
            x0, y0, x1, y1 = crop_rect
            observation = self.hand_extractor.extract(frame[y0:y1, x0:x1])
            observation = _translate_hand_observation(
                observation,
                offset_x=float(x0),
                offset_y=float(y0),
            )
            observed_hand = (
                observation.hand
                if isinstance(observation, HandObservation)
                else observation
            )
            if not _hand_overlaps_neck(observed_hand, homography):
                self._roi_misses += 1
                if self._roi_misses >= 2:
                    self._roi_misses = 0
                    self._roi_cooldown_frames = ROI_CROP_COOLDOWN_FRAMES
                # Preserve output continuity when a tighter model crop misses.
                # The cooldown prevents repeated double inference on a pose
                # that is consistently better recognized in the full frame.
                observation = self.hand_extractor.extract(frame)
            else:
                self._last_hand_source = "crop"
                self._roi_misses = 0
        return observation

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
        accepted = self.board_tracker.accept_detection(
            current_frame,
            fresh,
            timestamp_s=timestamp_s,
        )
        if accepted:
            self._fret_centers = (
                None if result.fret_centers is None else result.fret_centers.copy()
            )
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
