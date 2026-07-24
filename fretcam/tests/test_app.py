from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from fretcam.app import create_app
from fretcam.benchmark import SYNTHETIC_JPEG, run_loopback_benchmark
from fretcam.diagnostic_capture import BufferLimits, LocalCaptureSession


class EchoRoundTripTest(unittest.TestCase):
    def test_synthetic_jpeg_round_trip_over_websocket(self) -> None:
        self.assertTrue(SYNTHETIC_JPEG.startswith(b"\xff\xd8"))
        self.assertTrue(SYNTHETIC_JPEG.endswith(b"\xff\xd9"))

        metrics = run_loopback_benchmark(rounds=3, warmup=1)

        self.assertEqual(metrics.rounds, 3)
        self.assertEqual(metrics.payload_bytes, len(SYNTHETIC_JPEG))
        self.assertGreater(metrics.median_ms, 0)
        self.assertGreaterEqual(metrics.p95_ms, metrics.median_ms)


class FakeProcessor:
    def __init__(self) -> None:
        self.closed = False
        self.warmed_up = False
        self.reset_calls = 0
        self.controls: list[dict[str, object]] = []

    def warmup(self) -> None:
        self.warmed_up = True

    def reset(self) -> None:
        self.reset_calls += 1

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float | None = None,
    ) -> dict[str, object]:
        return {
            "type": "hud",
            "payload_bytes": len(payload),
            "detection": {"timestamp_s": timestamp_s},
            "server_ms": 1.0,
        }

    def handle_control(self, message: dict[str, object]) -> dict[str, object]:
        self.controls.append(message)
        return {"type": "control", "status": "applied"}

    def close(self) -> None:
        self.closed = True


class HudWebSocketTest(unittest.TestCase):
    def test_browser_exposes_live_fretboard_and_position_readouts(self) -> None:
        with TestClient(create_app(echo_mode=True)) as client:
            page = client.get("/")
            script = client.get("/static/client.js")

        self.assertEqual(page.status_code, 200)
        self.assertIn('id="fretboard-status"', page.text)
        self.assertIn('id="position-status"', page.text)
        self.assertIn("The green border follows the detected fretboard", page.text)
        self.assertIn('<link rel="icon" href="data:,"', page.text)
        self.assertIn('id="camera"', page.text)
        self.assertIn('id="preview"', page.text)
        self.assertIn('id="inference"', page.text)
        self.assertIn('id="camera-select"', page.text)
        self.assertIn('id="player-handedness"', page.text)
        self.assertIn('id="mirror-preview"', page.text)
        self.assertIn('id="calibrate"', page.text)
        self.assertIn('id="calibrate-two-point"', page.text)
        self.assertIn('id="continue-calibration"', page.text)
        self.assertIn('id="calibration-upper-position"', page.text)
        self.assertIn('id="export-diagnostics"', page.text)
        self.assertIn('id="diagnostics"', page.text)
        self.assertIn('id="local-accuracy-tools"', page.text)
        self.assertIn('id="trace-toggle"', page.text)
        self.assertIn('id="failure-buffer"', page.text)
        self.assertIn('id="mark-failure"', page.text)
        self.assertIn("Normal camera use saves no images", page.text)
        self.assertIn(
            "never training, evaluation, tuning, or release evidence", page.text
        )
        self.assertIn("updateLiveReadouts(detection, position)", script.text)
        self.assertIn("fitInferenceSize", script.text)
        self.assertIn("MAX_DIAGNOSTIC_SAMPLES = 300", script.text)
        self.assertIn("geometry_status", script.text)
        self.assertIn("devicechange", script.text)
        self.assertIn("populateCameras().catch", script.text)
        self.assertIn("sessionGeneration !== encodeGeneration", script.text)
        self.assertIn("socket !== encodeSocket", script.text)
        self.assertIn('type: "frame_context"', script.text)
        self.assertIn('type: "failure_mark"', script.text)
        self.assertIn('type: "trace_save"', script.text)
        self.assertIn("handednessSelect.disabled = traceActive", script.text)
        self.assertIn("enabled && !traceActive", script.text)
        self.assertNotIn('<video id="camera" hidden', page.text)

    def test_binary_frame_returns_json_and_closes_session_processor(self) -> None:
        processor = FakeProcessor()
        with TestClient(create_app(processor_factory=lambda: processor)) as client:
            self.assertEqual(client.get("/health").json()["mode"], "hud")
            with client.websocket_connect("/ws") as websocket:
                websocket.send_bytes(SYNTHETIC_JPEG)
                response = websocket.receive_json()

        self.assertEqual(response["type"], "hud")
        self.assertEqual(response["payload_bytes"], len(SYNTHETIC_JPEG))
        self.assertTrue(processor.warmed_up)
        self.assertEqual(processor.reset_calls, 1)
        self.assertTrue(processor.closed)

    def test_ordinary_live_socket_never_constructs_optional_capture_state(
        self,
    ) -> None:
        processor = FakeProcessor()
        capture_calls = 0

        def unexpected_capture() -> LocalCaptureSession:
            nonlocal capture_calls
            capture_calls += 1
            raise AssertionError("ordinary live mode must not initialize capture")

        with TestClient(
            create_app(
                processor_factory=lambda: processor,
                capture_factory=unexpected_capture,
            )
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_bytes(SYNTHETIC_JPEG)
                self.assertEqual(websocket.receive_json()["type"], "hud")

        self.assertEqual(capture_calls, 0)

    def test_json_control_message_is_dispatched_without_consuming_frame_slot(
        self,
    ) -> None:
        processor = FakeProcessor()
        message = {"type": "settings", "player_handedness": "left"}
        with TestClient(create_app(processor_factory=lambda: processor)) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(message)
                response = websocket.receive_json()

        self.assertEqual(response, {"type": "control", "status": "applied"})
        self.assertEqual(processor.controls, [message])

    def test_invalid_control_message_returns_recoverable_error(self) -> None:
        processor = FakeProcessor()
        with TestClient(create_app(processor_factory=lambda: processor)) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_text("not-json")
                response = websocket.receive_json()
                websocket.send_bytes(SYNTHETIC_JPEG)
                recovered = websocket.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(recovered["type"], "hud")

    def test_exact_trace_is_opt_in_and_saves_the_received_jpeg_and_context(
        self,
    ) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: LocalCaptureSession(root=root),
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    started = websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "frame_context",
                            "sequence": 7,
                            "session_offset_ms": 125.0,
                            "source_width": 1280,
                            "source_height": 720,
                            "inference_width": 640,
                            "inference_height": 360,
                            "jpeg_quality": 0.72,
                            "payload_bytes": len(SYNTHETIC_JPEG),
                        }
                    )
                    websocket.send_json({"type": "trace_save", "confirm_save": True})
                    saved = websocket.receive_json()

            self.assertEqual(started["status"], "trace_started")
            self.assertEqual(saved["status"], "trace_saved")
            self.assertNotIn("path", saved)
            package = root / "traces" / saved["package_id"]
            manifest = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8")
            )
            frame = manifest["frames"][0]
            self.assertEqual(frame["client"]["sequence"], 7)
            self.assertEqual(
                (package / frame["path"]).read_bytes(),
                SYNTHETIC_JPEG,
            )

    def test_trace_rejects_other_loopback_origins_before_creating_capture(
        self,
    ) -> None:
        processor = FakeProcessor()
        capture_calls = 0

        def capture_factory() -> LocalCaptureSession:
            nonlocal capture_calls
            capture_calls += 1
            raise AssertionError("cross-origin request must be rejected first")

        with TestClient(
            create_app(
                processor_factory=lambda: processor,
                capture_factory=capture_factory,
            )
        ) as client:
            with client.websocket_connect(
                "/ws",
                headers={"origin": "http://testserver:9999"},
            ) as websocket:
                websocket.send_json({"type": "trace_start"})
                rejected = websocket.receive_json()

        self.assertEqual(rejected["type"], "error")
        self.assertEqual(rejected["scope"], "capture")
        self.assertEqual(capture_calls, 0)

    def test_frozen_trace_consumes_later_context_and_remains_saveable(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            capture = LocalCaptureSession(
                root=root,
                trace_limits=BufferLimits(
                    duration_s=10.0,
                    max_frames=1,
                    max_bytes=1024,
                ),
            )
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    websocket.receive_json()
                    for sequence in (1, 2):
                        websocket.send_bytes(SYNTHETIC_JPEG)
                        websocket.receive_json()
                        websocket.send_json(
                            {
                                "type": "frame_context",
                                "sequence": sequence,
                                "session_offset_ms": sequence * 10.0,
                                "source_width": 1,
                                "source_height": 1,
                                "inference_width": 1,
                                "inference_height": 1,
                                "jpeg_quality": 0.72,
                                "payload_bytes": len(SYNTHETIC_JPEG),
                            }
                        )
                    websocket.send_json({"type": "trace_save", "confirm_save": True})
                    saved = websocket.receive_json()

            self.assertEqual(saved["status"], "trace_saved")
            manifest = json.loads(
                (root / "traces" / saved["package_id"] / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["frames"][0]["client"]["sequence"], 1)

    def test_invalid_frame_context_aborts_and_discards_capture(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            capture = LocalCaptureSession(root=root)
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "frame_context",
                            "sequence": 1,
                            "session_offset_ms": 10.0,
                            "source_width": 1,
                            "source_height": 1,
                            "inference_width": 1,
                            "inference_height": 1,
                            "jpeg_quality": 0.72,
                            "payload_bytes": len(SYNTHETIC_JPEG) + 1,
                        }
                    )
                    rejected = websocket.receive_json()

            self.assertEqual(rejected["type"], "error")
            self.assertEqual(rejected["scope"], "capture")
            self.assertFalse(rejected["capture"]["trace_enabled"])
            self.assertFalse(capture.status().trace_enabled)
            self.assertEqual(capture.status().trace_frames, 0)
            self.assertFalse(root.exists())

    def test_missing_frame_context_stops_capture_before_next_packet(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            capture = LocalCaptureSession(
                root=Path(directory) / "diagnostics",
            )
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    recovered = websocket.receive_json()

            self.assertEqual(recovered["type"], "hud")
            self.assertIn(
                "previous frame context was missing", recovered["capture_warning"]
            )
            self.assertFalse(capture.status().trace_enabled)
            self.assertEqual(capture.status().trace_frames, 0)

    def test_trace_blocks_inference_controls_until_saved_or_cancelled(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            capture = LocalCaptureSession(root=Path(directory) / "diagnostics")
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    websocket.receive_json()
                    websocket.send_json(
                        {"type": "settings", "player_handedness": "left"}
                    )
                    rejected = websocket.receive_json()
                    websocket.send_json({"type": "trace_cancel"})
                    websocket.receive_json()

        self.assertEqual(rejected["type"], "error")
        self.assertEqual(rejected["scope"], "capture")
        self.assertEqual(processor.controls, [])

    def test_failed_trace_write_is_recoverable_without_losing_buffer(self) -> None:
        class RetryCapture(LocalCaptureSession):
            fail_next_write = True

            def _write_package(self, **kwargs: object) -> Path:
                if self.fail_next_write:
                    self.fail_next_write = False
                    raise OSError("simulated disk failure")
                return super()._write_package(**kwargs)  # type: ignore[arg-type]

        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            capture = RetryCapture(root=Path(directory) / "diagnostics")
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "trace_start"})
                    websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "frame_context",
                            "sequence": 1,
                            "session_offset_ms": 10.0,
                            "source_width": 1,
                            "source_height": 1,
                            "inference_width": 1,
                            "inference_height": 1,
                            "jpeg_quality": 0.72,
                            "payload_bytes": len(SYNTHETIC_JPEG),
                        }
                    )
                    websocket.send_json({"type": "trace_save", "confirm_save": True})
                    failed = websocket.receive_json()
                    self.assertTrue(capture.status().trace_enabled)
                    self.assertEqual(capture.status().trace_frames, 1)
                    websocket.send_json({"type": "trace_save", "confirm_save": True})
                    saved = websocket.receive_json()

        self.assertEqual(failed["scope"], "capture")
        self.assertIn("save failed", failed["message"])
        self.assertEqual(saved["status"], "trace_saved")

    def test_disconnect_discards_unsaved_failure_buffer(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            capture = LocalCaptureSession(root=root)
            with TestClient(
                create_app(
                    processor_factory=lambda: processor,
                    capture_factory=lambda: capture,
                )
            ) as client:
                with client.websocket_connect("/ws") as websocket:
                    websocket.send_json({"type": "failure_buffer", "enabled": True})
                    websocket.receive_json()
                    websocket.send_bytes(SYNTHETIC_JPEG)
                    websocket.receive_json()
                    self.assertEqual(capture.status().failure_frames, 1)

            self.assertFalse(capture.status().failure_enabled)
            self.assertEqual(capture.status().failure_frames, 0)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
