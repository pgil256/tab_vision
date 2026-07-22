"""F2 live-frame detection chain built from TabVision's vision library."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import RULE_OF_18_RATIO, calibrate_board
from tabvision.video.fretboard.tracker import smooth_homography_track
from tabvision.video.guitar.yolo_backend import OBBPredictions, YoloOBBBackend
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS, HandSample
from tabvision.video.hand.mediapipe_backend import MediaPipeHandBackend
from tabvision.video.hand.neck_anchor import HandNeckAnchor, compute_neck_anchor

Point = tuple[float, float]
BoardCalibrator = Callable[
    [OBBPredictions, GuitarConfig], tuple[Homography, np.ndarray | None]
]


class Detector(Protocol):
    def predict_all(self, frame: np.ndarray) -> OBBPredictions: ...


class HandExtractor(Protocol):
    def extract(self, frame: np.ndarray) -> HandSample | None: ...

    def close(self) -> None: ...


class MediaPipeHandExtractor:
    """Small adapter that retains landmarks for both HUD and anchor output.

    TabVision's public ``detect_anchor`` intentionally returns only the coarse
    anchor. FretCam also needs the hand marker for its future HUD, so this
    quarantined adapter calls the backend's existing landmark extractor and
    then routes that exact ``HandSample`` through ``compute_neck_anchor``.
    """

    def __init__(self, backend: MediaPipeHandBackend | None = None) -> None:
        self.backend = backend or MediaPipeHandBackend()

    def extract(self, frame: np.ndarray) -> HandSample | None:
        return self.backend._extract_fretting_hand(frame)

    def close(self) -> None:
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
    anchor: HandNeckAnchor
    stage_latency: StageLatency

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


class DetectionChain:
    """Stateful 2 Hz board detector plus per-frame hand/anchor inference."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        hand_extractor: HandExtractor | None = None,
        guitar_config: GuitarConfig | None = None,
        detector_hz: float = 2.0,
        tracker_alpha: float = 0.3,
        min_lock_confidence: float = 0.2,
        calibrator: BoardCalibrator = calibrate_board,
    ) -> None:
        if detector_hz <= 0.0:
            raise ValueError("detector_hz must be positive")
        if not 0.0 < tracker_alpha <= 1.0:
            raise ValueError("tracker_alpha must be in (0, 1]")
        self.detector = detector or YoloOBBBackend()
        self.hand_extractor = hand_extractor or MediaPipeHandExtractor()
        self.guitar_config = guitar_config or GuitarConfig()
        self.detector_interval_s = 1.0 / detector_hz
        self.tracker_alpha = tracker_alpha
        self.min_lock_confidence = min_lock_confidence
        self.calibrator = calibrator
        self.reset()

    def reset(self) -> None:
        """Clear temporal state when a camera or replay clip changes."""
        self._last_detection_s: float | None = None
        self._homography = _empty_homography()
        self._fret_centers: np.ndarray | None = None

    def close(self) -> None:
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
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"expected BGR frame with shape (H, W, 3), got {frame.shape}"
            )
        now_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(now_s):
            raise ValueError("timestamp_s must be finite")

        total_started = time.perf_counter()
        detector_ms = 0.0
        homography_ms = 0.0
        detector_ran = self._should_detect(now_s)
        if detector_ran:
            detector_started = time.perf_counter()
            predictions = self.detector.predict_all(frame)
            detector_ms = (time.perf_counter() - detector_started) * 1000.0

            homography_started = time.perf_counter()
            fresh_homography, fresh_centers = self.calibrator(
                predictions, self.guitar_config
            )
            self._update_board(fresh_homography, fresh_centers)
            homography_ms = (time.perf_counter() - homography_started) * 1000.0
            self._last_detection_s = now_s

        hand_started = time.perf_counter()
        hand = self.hand_extractor.extract(frame)
        hand_ms = (time.perf_counter() - hand_started) * 1000.0

        anchor_started = time.perf_counter()
        anchor = compute_neck_anchor(hand, self._homography, self.guitar_config)
        anchor_ms = (time.perf_counter() - anchor_started) * 1000.0

        latency = StageLatency(
            detector_ms=detector_ms,
            homography_ms=homography_ms,
            hand_ms=hand_ms,
            anchor_ms=anchor_ms,
            total_ms=(time.perf_counter() - total_started) * 1000.0,
        )
        locked = self._homography.confidence >= self.min_lock_confidence
        return FrameDetection(
            timestamp_s=now_s,
            detector_ran=detector_ran,
            neck_locked=locked,
            fret_map_locked=self._fret_centers is not None,
            homography_confidence=float(self._homography.confidence),
            homography_method=self._homography.method,
            neck_quad=_neck_quad(self._homography) if locked else (),
            fret_ticks=(
                _fret_ticks(self._homography, self._fret_centers) if locked else ()
            ),
            hand_points=_hand_points(hand),
            anchor=anchor,
            stage_latency=latency,
        )

    def _should_detect(self, timestamp_s: float) -> bool:
        if self._last_detection_s is None:
            return True
        # A timestamp reset means a new/restarted source; reacquire immediately.
        return (
            timestamp_s < self._last_detection_s
            or timestamp_s - self._last_detection_s >= self.detector_interval_s
        )

    def _update_board(
        self,
        fresh: Homography,
        fret_centers: np.ndarray | None,
    ) -> None:
        if self._homography.confidence <= 0.0:
            self._homography = fresh
        else:
            self._homography = smooth_homography_track(
                [self._homography, fresh], alpha=self.tracker_alpha
            )[-1]
        if fret_centers is not None:
            self._fret_centers = np.asarray(fret_centers, dtype=np.float64).copy()


def process_frame(
    frame: np.ndarray,
    *,
    chain: DetectionChain,
    timestamp_s: float | None = None,
) -> FrameDetection:
    """Functional entry point around a caller-owned stateful chain."""
    return chain.process_frame(frame, timestamp_s=timestamp_s)


__all__ = [
    "DetectionChain",
    "FrameDetection",
    "FretTick",
    "HandPoint",
    "MediaPipeHandExtractor",
    "StageLatency",
    "process_frame",
]
