from __future__ import annotations

import unittest

import numpy as np

from fretcam.detection import DetectionChain, _fret_wire_xs, process_frame
from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import RULE_OF_18_RATIO
from tabvision.video.guitar.yolo_backend import OBBPredictions
from tabvision.video.hand.fingertip_to_fret import FingerSample, HandSample


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict_all(self, _frame: np.ndarray) -> OBBPredictions:
        self.calls += 1
        return OBBPredictions()


class FakeHandExtractor:
    def __init__(self, hand: HandSample | None) -> None:
        self.hand = hand
        self.calls = 0
        self.closed = False

    def extract(self, _frame: np.ndarray) -> HandSample | None:
        self.calls += 1
        return self.hand

    def close(self) -> None:
        self.closed = True


def _hand() -> HandSample:
    fingers = {
        name: FingerSample(name, (x, 25.0), 0.0, 0.8)
        for name, x in zip(
            ("index", "middle", "ring", "pinky"),
            (35.0, 40.0, 45.0, 50.0),
            strict=True,
        )
    }
    return HandSample(
        wrist_xy=(30.0, 25.0),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.9,
        fingers=fingers,
    )


def _calibrator(
    _predictions: OBBPredictions, cfg: GuitarConfig
) -> tuple[Homography, np.ndarray]:
    homography = Homography(
        H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
        confidence=0.8,
        method="fixture",
    )
    x0, scale = 0.0, 1.3
    frets = np.arange(cfg.max_fret + 1, dtype=np.float64) + 0.5
    centers = x0 + scale * (1.0 - np.power(RULE_OF_18_RATIO, frets))
    return homography, centers


class DetectionChainTest(unittest.TestCase):
    def test_detector_runs_at_two_hz_while_hand_runs_every_frame(self) -> None:
        detector = FakeDetector()
        hands = FakeHandExtractor(_hand())
        chain = DetectionChain(
            detector=detector,
            hand_extractor=hands,
            detector_hz=2.0,
            calibrator=_calibrator,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)

        first = process_frame(frame, chain=chain, timestamp_s=0.0)
        tracked = process_frame(frame, chain=chain, timestamp_s=0.1)
        reacquired = process_frame(frame, chain=chain, timestamp_s=0.5)

        self.assertTrue(first.detector_ran)
        self.assertFalse(tracked.detector_ran)
        self.assertTrue(reacquired.detector_ran)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(hands.calls, 3)
        self.assertTrue(tracked.neck_locked)
        self.assertEqual(len(first.neck_quad), 4)
        self.assertEqual(len(first.hand_points), 5)
        self.assertEqual(len(first.fret_ticks), 26)
        self.assertAlmostEqual(first.anchor.center_fret, 9.6)

    def test_missing_hand_returns_zero_confidence_anchor(self) -> None:
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(None),
            calibrator=_calibrator,
        )
        result = chain.process_frame(
            np.zeros((50, 100, 3), dtype=np.uint8), timestamp_s=0
        )

        self.assertTrue(result.neck_locked)
        self.assertEqual(result.anchor.confidence, 0.0)
        self.assertEqual(result.hand_points, ())

    def test_reset_forces_reacquisition(self) -> None:
        detector = FakeDetector()
        chain = DetectionChain(
            detector=detector,
            hand_extractor=FakeHandExtractor(_hand()),
            calibrator=_calibrator,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        chain.process_frame(frame, timestamp_s=10.0)
        chain.reset()
        after_reset = chain.process_frame(frame, timestamp_s=10.1)

        self.assertTrue(after_reset.detector_ran)
        self.assertEqual(detector.calls, 2)

    def test_invalid_frame_is_rejected(self) -> None:
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(_hand()),
            calibrator=_calibrator,
        )
        with self.assertRaisesRegex(ValueError, "BGR frame"):
            chain.process_frame(np.zeros((50, 100), dtype=np.uint8))


class FretWireProjectionTest(unittest.TestCase):
    def test_rule_of_18_centers_recover_wire_locations(self) -> None:
        origin, scale = 0.02, 1.4
        centers = origin + scale * (
            1.0 - RULE_OF_18_RATIO ** (np.arange(10, dtype=np.float64) + 0.5)
        )

        wires = _fret_wire_xs(centers)
        expected = origin + scale * (
            1.0 - RULE_OF_18_RATIO ** np.arange(11, dtype=np.float64)
        )

        np.testing.assert_allclose(wires, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
