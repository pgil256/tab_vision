"""FretCam-local optical tracking and fretboard geometry validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tabvision.types import Homography

CANONICAL_QUAD = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class TrackingSnapshot:
    homography: Homography
    status: str
    geometry_age_s: float
    detector_age_s: float
    stability: float
    flow_inliers: int = 0
    flow_inlier_ratio: float = 0.0
    flow_error_px: float = 0.0


class OpticalBoardTracker:
    """Track a validated canonical-to-image homography between YOLO passes."""

    def __init__(
        self,
        *,
        stale_after_s: float = 0.35,
        hard_expire_s: float = 0.90,
        max_features: int = 100,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.55,
        max_forward_backward_error_px: float = 1.5,
    ) -> None:
        if not 0.0 <= stale_after_s < hard_expire_s:
            raise ValueError("expected 0 <= stale_after_s < hard_expire_s")
        self.stale_after_s = stale_after_s
        self.hard_expire_s = hard_expire_s
        self.max_features = max_features
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_forward_backward_error_px = max_forward_backward_error_px
        self.reset()

    def reset(self) -> None:
        # OpenCV's RANSAC routines consume a thread-local RNG.  Re-seeding at
        # source/session boundaries makes a reset genuinely reproducible
        # instead of letting prior clips change the accepted flow geometry.
        cv2.setRNGSeed(0)
        self._homography = _missing_homography()
        self._previous_gray: np.ndarray | None = None
        self._previous_points: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._last_geometry_s: float | None = None
        self._last_detector_s: float | None = None
        self._status = "missing"
        self._stability = 0.0
        self._flow_inliers = 0
        self._flow_inlier_ratio = 0.0
        self._flow_error_px = 0.0

    @property
    def homography(self) -> Homography:
        return self._homography

    def advance(self, frame: np.ndarray, *, timestamp_s: float) -> TrackingSnapshot:
        """Advance optical flow to ``frame`` and return the current geometry."""
        gray = _gray(frame)
        shape = gray.shape
        if self._frame_shape is not None and shape != self._frame_shape:
            self._rescale_for_shape(self._frame_shape, shape)
            self._previous_gray = None
            self._previous_points = None

        tracked = False
        if (
            self._homography.confidence > 0.0
            and self._previous_gray is not None
            and self._previous_gray.shape == gray.shape
        ):
            tracked = self._advance_flow(self._previous_gray, gray, timestamp_s)

        if not tracked:
            # LK points are coordinates in ``_previous_gray``. Once a flow pass
            # fails they must not be carried forward and paired with this new
            # source image; reseed below from the current frame instead.
            self._previous_points = None
        if not tracked and self._homography.confidence > 0.0:
            age = self._age(timestamp_s, self._last_geometry_s)
            if age >= self.hard_expire_s:
                self._homography = _missing_homography()
                self._status = "missing"
                self._stability = 0.0
            else:
                self._status = "stale" if age >= self.stale_after_s else "held"
                self._stability = max(
                    0.0,
                    self._stability * (1.0 - age / max(self.hard_expire_s, 1e-9)),
                )

        self._previous_gray = gray
        self._frame_shape = shape
        if self._homography.confidence > 0.0 and (
            self._previous_points is None or len(self._previous_points) < 24
        ):
            self._previous_points = self._seed_features(gray, self._homography)
        return self.snapshot(timestamp_s)

    def accept_detection(
        self,
        frame: np.ndarray,
        fresh: Homography,
        *,
        timestamp_s: float,
    ) -> bool:
        """Atomically accept a plausible fresh detector geometry."""
        height, width = frame.shape[:2]
        if fresh.confidence <= 0.0 or not _valid_homography(
            fresh, width=width, height=height
        ):
            return False

        current_valid = self._homography.confidence > 0.0 and self._status != "missing"
        current_age = self._age(timestamp_s, self._last_geometry_s)
        if current_valid and current_age < self.hard_expire_s:
            old_quad = _project_quad(self._homography)
            fresh_quad = _project_quad(fresh)
            diagonal = math.hypot(width, height)
            rms = float(
                np.sqrt(np.mean(np.sum(np.square(old_quad - fresh_quad), axis=1)))
            )
            iou = _quad_iou(old_quad, fresh_quad)
            if not _canonical_axes_aligned(old_quad, fresh_quad):
                return False
            if rms > 0.20 * diagonal or (rms > 0.08 * diagonal and iou < 0.35):
                return False

        self._homography = Homography(
            H=np.asarray(fresh.H, dtype=np.float64).copy(),
            confidence=float(np.clip(fresh.confidence, 0.0, 1.0)),
            method=fresh.method,
        )
        self._last_geometry_s = timestamp_s
        self._last_detector_s = timestamp_s
        self._status = "detected"
        self._stability = 1.0
        self._flow_inliers = 0
        self._flow_inlier_ratio = 0.0
        self._flow_error_px = 0.0
        gray = _gray(frame)
        self._previous_gray = gray
        self._frame_shape = gray.shape
        self._previous_points = self._seed_features(gray, self._homography)
        return True

    def snapshot(self, timestamp_s: float) -> TrackingSnapshot:
        return TrackingSnapshot(
            homography=self._homography,
            status=self._status,
            geometry_age_s=self._age(timestamp_s, self._last_geometry_s),
            detector_age_s=self._age(timestamp_s, self._last_detector_s),
            stability=float(np.clip(self._stability, 0.0, 1.0)),
            flow_inliers=self._flow_inliers,
            flow_inlier_ratio=float(np.clip(self._flow_inlier_ratio, 0.0, 1.0)),
            flow_error_px=max(0.0, self._flow_error_px),
        )

    def _advance_flow(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        timestamp_s: float,
    ) -> bool:
        points = self._previous_points
        if points is None or len(points) < self.min_inliers:
            points = self._seed_features(previous_gray, self._homography)
        if points is None or len(points) < self.min_inliers:
            return False

        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if forward is None or status_f is None:
            return False
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous_gray,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if backward is None or status_b is None:
            return False

        original = points.reshape(-1, 2)
        advanced = forward.reshape(-1, 2)
        returned = backward.reshape(-1, 2)
        fb_error = np.linalg.norm(original - returned, axis=1)
        good = (
            status_f.reshape(-1).astype(bool)
            & status_b.reshape(-1).astype(bool)
            & np.all(np.isfinite(advanced), axis=1)
            & (fb_error <= self.max_forward_backward_error_px)
        )
        source = original[good]
        destination = advanced[good]
        if len(source) < self.min_inliers:
            return False
        motion, inlier_mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            2.5,
        )
        if motion is None or inlier_mask is None:
            return False
        inliers = inlier_mask.reshape(-1).astype(bool)
        count = int(np.count_nonzero(inliers))
        ratio = count / max(len(source), 1)
        if count < self.min_inliers or ratio < self.min_inlier_ratio:
            return False

        candidate_matrix = np.asarray(motion, dtype=np.float64) @ np.asarray(
            self._homography.H, dtype=np.float64
        )
        if abs(candidate_matrix[2, 2]) < 1e-9:
            return False
        candidate_matrix /= candidate_matrix[2, 2]
        candidate = Homography(
            H=candidate_matrix,
            # The detector confidence describes the board identity/fit.  Do not
            # multiply it by the flow inlier ratio on every frame: that causes
            # an otherwise healthy track to decay exponentially with FPS.
            # Flow quality is reported independently through ``stability`` and
            # is already part of the composite position confidence.
            confidence=self._homography.confidence,
            method="optical_flow",
        )
        height, width = gray.shape
        if not _valid_homography(candidate, width=width, height=height):
            return False
        if not _canonical_axes_aligned(
            _project_quad(self._homography),
            _project_quad(candidate),
        ):
            return False

        source_inliers = source[inliers]
        destination_inliers = destination[inliers]
        projected = cv2.perspectiveTransform(
            source_inliers.reshape(-1, 1, 2).astype(np.float32),
            np.asarray(motion, dtype=np.float64),
        ).reshape(-1, 2)
        error = float(
            np.median(np.linalg.norm(projected - destination_inliers, axis=1))
        )
        self._homography = candidate
        self._last_geometry_s = timestamp_s
        self._status = "tracked"
        self._stability = float(np.clip(ratio * math.exp(-error / 2.0), 0.0, 1.0))
        self._flow_inliers = count
        self._flow_inlier_ratio = ratio
        self._flow_error_px = error
        self._previous_points = destination_inliers.reshape(-1, 1, 2).astype(np.float32)
        return True

    def _seed_features(
        self, gray: np.ndarray, homography: Homography
    ) -> np.ndarray | None:
        quad = _project_quad(homography)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=6,
            mask=mask,
            blockSize=5,
        )

    def _rescale_for_shape(
        self,
        old_shape: tuple[int, int],
        new_shape: tuple[int, int],
    ) -> None:
        if self._homography.confidence <= 0.0:
            return
        old_height, old_width = old_shape
        new_height, new_width = new_shape
        scale = np.asarray(
            [
                [new_width / old_width, 0.0, 0.0],
                [0.0, new_height / old_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._homography = Homography(
            H=scale @ np.asarray(self._homography.H, dtype=np.float64),
            confidence=self._homography.confidence,
            method=self._homography.method,
        )

    @staticmethod
    def _age(timestamp_s: float, updated_s: float | None) -> float:
        if updated_s is None:
            return math.inf
        return max(0.0, timestamp_s - updated_s)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected BGR frame, got shape {frame.shape}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _missing_homography() -> Homography:
    return Homography(
        H=np.eye(3, dtype=np.float64),
        confidence=0.0,
        method="missing",
    )


def _project_quad(homography: Homography) -> np.ndarray:
    matrix = np.asarray(homography.H, dtype=np.float64)
    return cv2.perspectiveTransform(
        CANONICAL_QUAD.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)


def _valid_homography(
    homography: Homography,
    *,
    width: int,
    height: int,
) -> bool:
    matrix = np.asarray(homography.H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False
    quad = _project_quad(homography)
    if not np.all(np.isfinite(quad)) or not cv2.isContourConvex(
        np.round(quad).astype(np.int32)
    ):
        return False
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    frame_area = max(float(width * height), 1.0)
    if area < 0.005 * frame_area or area > 1.5 * frame_area:
        return False
    diagonal = math.hypot(width, height)
    center = np.mean(quad, axis=0)
    return bool(
        -0.15 * diagonal <= center[0] <= width + 0.15 * diagonal
        and -0.15 * diagonal <= center[1] <= height + 0.15 * diagonal
    )


def _quad_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return 0.0 if union <= 0.0 else float(intersection) / union


def _canonical_axes_aligned(first: np.ndarray, second: np.ndarray) -> bool:
    """Reject a detector quad whose ordered canonical axes have flipped."""

    def axes(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = (quad[0] + quad[3]) / 2.0
        right = (quad[1] + quad[2]) / 2.0
        top = (quad[0] + quad[1]) / 2.0
        bottom = (quad[2] + quad[3]) / 2.0
        return right - left, bottom - top

    old_x, old_y = axes(np.asarray(first, dtype=np.float64))
    new_x, new_y = axes(np.asarray(second, dtype=np.float64))
    for old_axis, new_axis in ((old_x, new_x), (old_y, new_y)):
        denominator = float(np.linalg.norm(old_axis) * np.linalg.norm(new_axis))
        if denominator <= 1e-9:
            return False
        if float(np.dot(old_axis, new_axis)) / denominator <= 0.0:
            return False
    return True


def align_detection_homography(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    homography: Homography,
) -> tuple[Homography, float, float] | None:
    """Warp delayed detector geometry from its source frame into the target."""
    source_gray = _gray(source_frame)
    target_gray = _gray(target_frame)
    if source_gray.shape != target_gray.shape:
        return None
    quad = _project_quad(homography)
    mask = np.zeros(source_gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    points = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=120,
        qualityLevel=0.01,
        minDistance=6,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 8:
        return None
    forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
        source_gray,
        target_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if forward is None or status_f is None:
        return None
    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
        target_gray,
        source_gray,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if backward is None or status_b is None:
        return None
    original = points.reshape(-1, 2)
    destination = forward.reshape(-1, 2)
    returned = backward.reshape(-1, 2)
    good = (
        status_f.reshape(-1).astype(bool)
        & status_b.reshape(-1).astype(bool)
        & np.all(np.isfinite(destination), axis=1)
        & (np.linalg.norm(original - returned, axis=1) <= 1.5)
    )
    original = original[good]
    destination = destination[good]
    if len(original) < 8:
        return None
    motion, mask_inliers = cv2.findHomography(
        original,
        destination,
        cv2.RANSAC,
        2.5,
    )
    if motion is None or mask_inliers is None:
        return None
    inliers = mask_inliers.reshape(-1).astype(bool)
    count = int(np.count_nonzero(inliers))
    ratio = count / max(len(original), 1)
    if count < 8 or ratio < 0.55:
        return None
    matrix = np.asarray(motion, dtype=np.float64) @ np.asarray(
        homography.H, dtype=np.float64
    )
    if abs(matrix[2, 2]) < 1e-9:
        return None
    matrix /= matrix[2, 2]
    aligned = Homography(
        H=matrix,
        confidence=float(np.clip(homography.confidence * ratio, 0.0, 1.0)),
        method=homography.method,
    )
    height, width = target_gray.shape
    if not _valid_homography(aligned, width=width, height=height):
        return None
    projected = cv2.perspectiveTransform(
        original[inliers].reshape(-1, 1, 2).astype(np.float32),
        np.asarray(motion, dtype=np.float64),
    ).reshape(-1, 2)
    error = float(np.median(np.linalg.norm(projected - destination[inliers], axis=1)))
    return aligned, ratio, error


__all__ = [
    "OpticalBoardTracker",
    "TrackingSnapshot",
    "align_detection_homography",
]
