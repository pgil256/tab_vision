"""Non-UI FretCam adapter for TabVision pipeline orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import cast

import cv2
import numpy as np

from fretcam.capo import CapoDetector
from fretcam.detection import DetectionChain, FrameDetection
from fretcam.position import EstimatorConfig, PositionEstimator
from fretcam.processing import build_hand_search_hint
from tabvision.types import GuitarConfig
from tabvision.video.position import (
    FingerContactObservation,
    PositionObservationState,
    PositionWindowObservation,
    SessionCapoObservation,
    VideoObservations,
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
        return list(self.analyze_all(frames, stride=stride).windows)

    def analyze_all(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        stride: int = 1,
    ) -> VideoObservations:
        """Return position windows *and* finger contacts from one traversal.

        Frame decoding and inference dominate the cost, so both evidence types
        come out of the same pass. The two are gated differently on purpose:
        windows require the estimator's ``locked``/``holding`` lock, contacts
        do not, because the lock criterion is a display requirement rather than
        an evidence one.
        """
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
            contacts: list[FingerContactObservation] = []
            # A capo is static, so this accumulates over the whole session and
            # costs nothing beyond reading pixels the traversal already decoded.
            capo_detector = CapoDetector()
            for frame_index, (timestamp_s, frame) in enumerate(frames):
                if frame_index % stride:
                    continue
                timestamp = float(timestamp_s)
                if not math.isfinite(timestamp):
                    raise ValueError("frame timestamp_s must be finite")

                fitted = self._fit_frame(frame)
                detection = chain.process_frame(fitted, timestamp_s=timestamp)
                capo_detector.observe(
                    fitted,
                    detection.fret_ticks,
                    neck_quad=detection.neck_quad,
                    body_joint_fret=detection.body_joint_fret,
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

                contact_record = self._contact_observation(detection)
                if contact_record is not None:
                    contacts.append(contact_record)

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
            capo = capo_detector.estimate()
            return VideoObservations(
                windows=tuple(observations),
                contacts=tuple(contacts),
                capo=SessionCapoObservation(
                    fret=capo.fret,
                    confidence=capo.confidence,
                    frames_observed=capo.frames_observed,
                    reason=capo.reason,
                ),
            )
        finally:
            chain.close()

    def _contact_observation(
        self,
        detection: FrameDetection,
    ) -> FingerContactObservation | None:
        """Project one frame's finger contacts into TabVision's convention.

        FretCam numbers strings the way tab notation does — 1 = high E — while
        ``TabEvent.string_idx`` is zero-based from the low E, so the map is
        ``n_strings - string``. Getting this backwards is not a crash; it is a
        silent likelihood ratio of exactly 1.00 (see
        ``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``).
        """
        n_strings = self.config.n_strings
        positions: set[tuple[int, int]] = set()
        for contact in detection.finger_contacts:
            if contact.string is None or not contact.visible:
                continue
            string_idx = n_strings - int(contact.string)
            fret = int(contact.fret)
            if not 0 <= string_idx < n_strings or not 0 <= fret <= self.config.max_fret:
                continue
            positions.add((string_idx, fret))
        if not positions:
            return None

        timestamp = float(detection.timestamp_s)
        confidence = float(detection.observation_confidence)
        if not math.isfinite(timestamp) or not math.isfinite(confidence):
            return None
        return FingerContactObservation(
            timestamp_s=timestamp,
            positions=tuple(sorted(positions)),
            confidence=min(1.0, max(0.0, confidence)),
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
