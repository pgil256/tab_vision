"""Non-UI FretCam adapter for TabVision pipeline orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import cast

import cv2
import numpy as np

from fretcam.detection import DetectionChain, FrameDetection
from fretcam.position import EstimatorConfig, PositionEstimator
from fretcam.processing import build_hand_search_hint
from tabvision.types import GuitarConfig
from tabvision.video.position import (
    PositionObservationState,
    PositionWindowObservation,
)

ChainFactory = Callable[..., DetectionChain]
EstimatorFactory = Callable[[EstimatorConfig], PositionEstimator]


class FretCamPositionAnalyzer:
    """Run FretCam's stabilized position loop over an offline media clock."""

    def __init__(
        self,
        config: GuitarConfig | None = None,
        *,
        chain_factory: ChainFactory = DetectionChain,
        estimator_factory: EstimatorFactory = PositionEstimator,
        min_confidence: float = 0.20,
        max_frame_width: int = 640,
        max_frame_height: int = 480,
    ) -> None:
        if not 0.20 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.20, 1]")
        if max_frame_width <= 0 or max_frame_height <= 0:
            raise ValueError("maximum frame dimensions must be positive")
        self.config = config or GuitarConfig()
        self._chain_factory = chain_factory
        self._estimator_factory = estimator_factory
        self.min_confidence = min_confidence
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height

    def analyze(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        stride: int = 1,
    ) -> list[PositionWindowObservation]:
        """Return reliable position windows while preserving media timestamps."""
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")

        chain = self._chain_factory(
            guitar_config=self.config,
            detector_hz=2.0,
            background_detector=False,
            crop_hand=True,
        )
        try:
            estimator = self._estimator_factory(
                EstimatorConfig(max_fret=self.config.max_fret)
            )
            observations: list[PositionWindowObservation] = []
            for frame_index, (timestamp_s, frame) in enumerate(frames):
                if frame_index % stride:
                    continue
                timestamp = float(timestamp_s)
                if not math.isfinite(timestamp):
                    raise ValueError("frame timestamp_s must be finite")

                detection = chain.process_frame(
                    self._fit_frame(frame),
                    timestamp_s=timestamp,
                )
                position_fret, confidence = self._position_observation(detection)
                estimate = estimator.update(
                    index_fret=position_fret,
                    vision_confidence=confidence,
                    timestamp_s=detection.timestamp_s,
                )

                hint_setter = getattr(chain, "set_hand_search_hint", None)
                if hint_setter is not None:
                    hint_setter(build_hand_search_hint(detection, estimate))

                estimate_timestamp = float(estimate.timestamp_s)
                estimate_confidence = float(estimate.confidence)
                if (
                    estimate.state not in {"locked", "holding"}
                    or estimate.position is None
                    or not math.isfinite(estimate_timestamp)
                    or not math.isfinite(estimate_confidence)
                    or estimate_confidence < self.min_confidence
                ):
                    continue
                observations.append(
                    PositionWindowObservation(
                        timestamp_s=estimate_timestamp,
                        position=int(estimate.position),
                        window_frets=tuple(int(fret) for fret in estimate.window_frets),
                        confidence=estimate_confidence,
                        state=cast(PositionObservationState, estimate.state),
                    )
                )
            return observations
        finally:
            chain.close()

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

    @staticmethod
    def _position_observation(
        detection: FrameDetection,
    ) -> tuple[float | None, float]:
        if detection.composite_available or detection.position_fret is not None:
            return detection.position_fret, detection.observation_confidence
        return (
            detection.index_fret,
            detection.anchor.confidence if detection.neck_locked else 0.0,
        )


__all__ = ["FretCamPositionAnalyzer"]
