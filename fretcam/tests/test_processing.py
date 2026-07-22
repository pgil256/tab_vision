from __future__ import annotations

import cv2
import numpy as np

from fretcam.detection import FrameDetection, FretTick, HandPoint, StageLatency
from fretcam.position import EstimatorConfig, PositionEstimator
from fretcam.processing import HudFrameProcessor
from tabvision.video.hand.neck_anchor import HandNeckAnchor


class FakeChain:
    def __init__(self) -> None:
        self.closed = False
        self.frame_shape: tuple[int, ...] | None = None

    def process_frame(self, frame: np.ndarray, *, timestamp_s: float) -> FrameDetection:
        self.frame_shape = frame.shape
        return FrameDetection(
            timestamp_s=timestamp_s,
            detector_ran=True,
            neck_locked=True,
            fret_map_locked=True,
            homography_confidence=0.8,
            homography_method="fixture",
            neck_quad=(
                (10.0, 10.0),
                (90.0, 10.0),
                (90.0, 40.0),
                (10.0, 40.0),
            ),
            fret_ticks=(FretTick(5, (50.0, 10.0), (50.0, 40.0)),),
            hand_points=(HandPoint("index", 50.0, 25.0),),
            index_fret=5.0,
            anchor=HandNeckAnchor(5.2, 4.0, 8.0, 0.72, "fixture"),
            stage_latency=StageLatency(1.0, 1.0, 1.0, 1.0, 4.0),
            index_fret_raw=5.2,
        )

    def close(self) -> None:
        self.closed = True


def _jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((50, 100, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_processor_emits_complete_json_hud_contract_and_closes() -> None:
    chain = FakeChain()
    processor = HudFrameProcessor(
        chain=chain,  # type: ignore[arg-type]
        estimator=PositionEstimator(EstimatorConfig(hysteresis_frames=1)),
    )

    result = processor.process_jpeg(_jpeg())
    processor.close()

    assert result["type"] == "hud"
    assert result["frame"] == {"width": 100, "height": 50}
    assert result["detection"]["neck_locked"] is True  # type: ignore[index]
    assert result["detection"]["index_fret"] == 5.0  # type: ignore[index]
    assert result["detection"]["index_fret_raw"] == 5.2  # type: ignore[index]
    assert result["position"]["label"] == "Position V"  # type: ignore[index]
    assert result["guidance"]["code"] == "locked"  # type: ignore[index]
    assert result["server_ms"] > 0
    assert chain.frame_shape == (50, 100, 3)
    assert chain.closed


def test_processor_rejects_non_jpeg_payload() -> None:
    processor = HudFrameProcessor(chain=FakeChain())  # type: ignore[arg-type]

    try:
        try:
            processor.process_jpeg(b"not a jpeg")
        except ValueError as exc:
            assert "decodable" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("invalid JPEG was accepted")
    finally:
        processor.close()
