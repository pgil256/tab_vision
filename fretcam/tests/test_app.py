from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
