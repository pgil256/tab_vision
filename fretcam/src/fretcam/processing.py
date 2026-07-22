"""In-memory JPEG to live-HUD response processing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from fretcam.detection import DetectionChain
from fretcam.guidance import assess_guidance
from fretcam.position import PositionEstimator


class FrameProcessor(Protocol):
    def warmup(self) -> None: ...

    def reset(self) -> None: ...

    def process_jpeg(self, payload: bytes) -> dict[str, object]: ...

    def close(self) -> None: ...


FrameProcessorFactory = Callable[[], FrameProcessor]


class HudFrameProcessor:
    """Own one stateful detection and position chain per WebSocket session."""

    def __init__(
        self,
        *,
        chain: DetectionChain | None = None,
        estimator: PositionEstimator | None = None,
    ) -> None:
        self.chain = chain or DetectionChain(detector_hz=2.0)
        self.estimator = estimator or PositionEstimator()

    def warmup(self) -> None:
        """Pay one-time model initialization before the server reports ready."""
        self.chain.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=0.0
        )
        self.reset()

    def reset(self) -> None:
        self.chain.reset()
        self.estimator.reset()

    def process_jpeg(self, payload: bytes) -> dict[str, object]:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("payload is not a decodable colour JPEG")

        timestamp_s = time.monotonic()
        detection = self.chain.process_frame(frame, timestamp_s=timestamp_s)
        estimate = self.estimator.update(
            index_fret=detection.index_fret,
            vision_confidence=(
                detection.anchor.confidence if detection.neck_locked else 0.0
            ),
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
            "frame": {"width": width, "height": height},
            "detection": detection.as_dict(),
            "position": estimate.as_dict(),
            "guidance": guidance.as_dict(),
            "server_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def close(self) -> None:
        self.chain.close()


__all__ = ["FrameProcessor", "FrameProcessorFactory", "HudFrameProcessor"]
