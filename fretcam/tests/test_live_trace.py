from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fretcam.diagnostic_capture import (
    DiagnosticCaptureError,
    FailureExpectation,
    LocalCaptureSession,
)
from fretcam.live_trace import TraceError, compare_trace, load_trace


def _jpeg(marker: bytes) -> bytes:
    return b"\xff\xd8" + marker + b"\xff\xd9"


def _hud(
    timestamp_s: float,
    *,
    position: int = 5,
    blockers: tuple[str, ...] = (),
    geometry_status: str = "tracked",
    detector_requested: bool = False,
    hand_search_source: str = "last_hand_crop",
    server_ms: float = 8.0,
) -> dict[str, object]:
    return {
        "type": "hud",
        "detection": {
            "timestamp_s": timestamp_s,
            "neck_locked": True,
            "fret_map_locked": True,
            "geometry_status": geometry_status,
            "homography_method": "fixture",
            "geometry_stability": 0.9,
            "detector_ran": False,
            "detector_requested": detector_requested,
            "detector_pending": detector_requested,
            "hand_detector_ran": True,
            "hand_source": "temporal_detector",
            "hand_search_source": hand_search_source,
            "hand_refresh_reason": "deadline",
            "hand_detector_interval_ms": 100.0,
            "hand_schedule_mode": (
                "locked" if hand_search_source == "last_hand_crop" else "recovering"
            ),
            "hand_pose_predicted": False,
            "confidence_factors": {"blockers": list(blockers)},
            "stage_latency": {"total_ms": server_ms - 1.0},
        },
        "position": {
            "state": "locked",
            "position": position,
            "reason": "stable",
            "confidence": 0.8,
        },
        "server_ms": server_ms,
    }


def _saved_trace(tmp_path: Path) -> tuple[Path, tuple[bytes, ...]]:
    payloads = (_jpeg(b"one"), _jpeg(b"two"))
    session = LocalCaptureSession(root=tmp_path / "diagnostics")
    session.start_trace(
        session_metadata={"scenario": "position-v"},
        replay_controls=({"type": "settings", "player_handedness": "right"},),
    )
    for index, payload in enumerate(payloads):
        timestamp = 50.0 + index * 0.1
        session.record_frame(
            payload,
            _hud(timestamp, detector_requested=index == 1),
            observed_at_s=100.0 + index * 0.1,
            processor_timestamp_s=timestamp,
            client_metadata={
                "sequence": index + 10,
                "source_width": 1280,
                "source_height": 720,
                "inference_width": 640,
                "inference_height": 360,
                "jpeg_quality": 0.72,
            },
            server_metadata={"received_order": index + 1},
        )
    return session.save_trace(confirm=True), payloads


class FakeTraceProcessor:
    def __init__(self, *, divergent: bool = False) -> None:
        self.divergent = divergent
        self.warmed = False
        self.reset_count = 0
        self.closed = False
        self.timestamps: list[float] = []
        self.controls: list[dict[str, object]] = []
        self.events: list[str] = []

    def warmup(self) -> None:
        self.warmed = True

    def reset(self) -> None:
        self.reset_count += 1
        self.events.append("reset")

    def handle_control(self, message: dict[str, object]) -> dict[str, object]:
        self.controls.append(message)
        self.events.append("control")
        return {"type": "control", "status": "applied"}

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float,
    ) -> dict[str, object]:
        self.timestamps.append(timestamp_s)
        index = len(self.timestamps) - 1
        assert payload in {_jpeg(b"one"), _jpeg(b"two")}
        if not self.divergent:
            return _hud(
                timestamp_s,
                detector_requested=index == 1,
                server_ms=999.0,
            )
        return _hud(
            timestamp_s,
            position=6,
            blockers=("finger_conflict",),
            geometry_status="stale",
            detector_requested=True,
            hand_search_source="neck_recovery",
            server_ms=999.0,
        )

    def close(self) -> None:
        self.closed = True


def test_loader_verifies_exact_packets_metadata_and_hud(tmp_path: Path) -> None:
    package, payloads = _saved_trace(tmp_path)

    trace = load_trace(package)

    assert trace.package_path == package.resolve()
    assert trace.session == {"scenario": "position-v"}
    assert trace.replay_controls == (
        {"type": "settings", "player_handedness": "right"},
    )
    assert tuple(frame.payload for frame in trace.frames) == payloads
    assert [frame.relative_timestamp_s for frame in trace.frames] == [
        pytest.approx(0.0),
        pytest.approx(0.1),
    ]
    assert [frame.processor_timestamp_s for frame in trace.frames] == [50.0, 50.1]
    assert trace.frames[0].client_metadata["inference_width"] == 640
    assert trace.frames[1].server_metadata["received_order"] == 2
    assert trace.frames[0].live_hud["position"]["position"] == 5  # type: ignore[index]


def test_browser_trace_verifies_exact_jpeg_dimensions_and_complete_context(
    tmp_path: Path,
) -> None:
    ok, encoded = cv2.imencode(".jpg", np.zeros((2, 3, 3), dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    hud = _hud(5.0)
    hud["frame"] = {"width": 3, "height": 2}
    session = LocalCaptureSession(root=tmp_path / "diagnostics")
    session.start_trace(
        session_metadata={"source": "browser_live", "hud_version": 2},
        replay_controls=({"type": "settings", "player_handedness": "right"},),
    )
    session.record_frame(
        payload,
        hud,
        observed_at_s=5.0,
        client_metadata={
            "sequence": 9,
            "session_offset_ms": 100.0,
            "source_width": 1280,
            "source_height": 720,
            "inference_width": 3,
            "inference_height": 2,
            "jpeg_quality": 0.72,
            "payload_bytes": len(payload),
        },
    )
    package = session.save_trace(confirm=True)

    loaded = load_trace(package)
    assert loaded.frames[0].client_metadata["inference_width"] == 3

    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["client"]["inference_width"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TraceError, match="dimensions differ"):
        load_trace(package)


def test_comparator_injects_timestamps_controls_and_ignores_latency(
    tmp_path: Path,
) -> None:
    package, _ = _saved_trace(tmp_path)
    processor = FakeTraceProcessor()

    result = compare_trace(
        package,
        processor_factory=lambda: processor,
        pace=False,
    )

    assert result["frames"] == 2
    assert result["matched_frames"] == 2
    assert result["mismatched_frames"] == 0
    assert result["divergence_counts"] == {}
    assert processor.warmed
    assert processor.reset_count == 1
    assert processor.timestamps == [50.0, 50.1]
    assert processor.controls == [{"type": "settings", "player_handedness": "right"}]
    assert processor.events[:2] == ["control", "reset"]
    assert processor.closed
    assert all(frame["differences"] == [] for frame in result["frame_results"])


def test_comparator_reports_position_blocker_geometry_detector_and_hand_divergence(
    tmp_path: Path,
) -> None:
    package, _ = _saved_trace(tmp_path)
    processor = FakeTraceProcessor(divergent=True)

    result = compare_trace(
        package,
        processor_factory=lambda: processor,
        pace=False,
    )

    assert result["mismatched_frames"] == 2
    counts = result["divergence_counts"]
    assert counts["position.position"] == 2
    assert counts["detection.confidence_factors.blockers"] == 2
    assert counts["detection.geometry_status"] == 2
    assert counts["detection.hand_search_source"] == 2
    assert counts["detection.hand_schedule_mode"] == 2
    assert counts["detection.detector_requested"] == 1
    fields = {
        difference["field"]
        for frame in result["frame_results"]
        for difference in frame["differences"]
    }
    assert "server_ms" not in fields
    assert "detection.stage_latency.total_ms" not in fields
    assert processor.closed


def test_loader_rejects_tampered_frame_hash(tmp_path: Path) -> None:
    package, _ = _saved_trace(tmp_path)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    frame_path = package / Path(*manifest["frames"][0]["path"].split("/"))
    frame_path.write_bytes(_jpeg(b"tampered"))

    with pytest.raises(TraceError, match="byte count differs|hash differs"):
        load_trace(package)


def test_loader_rejects_failure_packages_even_with_valid_exact_jpegs(
    tmp_path: Path,
) -> None:
    session = LocalCaptureSession(root=tmp_path / "diagnostics")
    session.set_failure_buffer(True)
    session.record_frame(
        _jpeg(b"failure"),
        _hud(2.0),
        observed_at_s=2.0,
    )
    package = session.mark_failure(
        FailureExpectation(position=5, pressing_fingers=("index",)),
        confirm=True,
    )

    with pytest.raises(TraceError, match="cannot be replayed"):
        load_trace(package)


def test_loader_enforces_immutable_diagnostics_policy(tmp_path: Path) -> None:
    package, _ = _saved_trace(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy"]["evaluation_allowed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiagnosticCaptureError, match="policy"):
        load_trace(package)


def test_comparator_closes_processor_when_timestamp_injection_is_unsupported(
    tmp_path: Path,
) -> None:
    package, _ = _saved_trace(tmp_path)

    class LegacyProcessor(FakeTraceProcessor):
        def process_jpeg(self, payload: bytes) -> dict[str, object]:  # type: ignore[override]
            return _hud(0.0)

    processor = LegacyProcessor()
    with pytest.raises(TraceError, match="timestamp_s"):
        compare_trace(
            package,
            processor_factory=lambda: processor,  # type: ignore[arg-type]
            pace=False,
        )
    assert processor.closed
