from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from fretcam.app import create_app
from fretcam.benchmark import SYNTHETIC_JPEG, run_loopback_benchmark


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

    def warmup(self) -> None:
        self.warmed_up = True

    def reset(self) -> None:
        self.reset_calls += 1

    def process_jpeg(self, payload: bytes) -> dict[str, object]:
        return {
            "type": "hud",
            "payload_bytes": len(payload),
            "server_ms": 1.0,
        }

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
        self.assertIn("updateLiveReadouts(detection, position)", script.text)
        self.assertIn('context.strokeStyle = "#59ff88"', script.text)

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


if __name__ == "__main__":
    unittest.main()
