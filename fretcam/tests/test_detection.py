from __future__ import annotations

import unittest

import numpy as np

from fretcam.detection import (
    DetectionChain,
    HandObservation,
    _fret_cell_from_canonical_x,
    _fret_wire_xs,
    compute_index_fret,
    compute_index_fret_raw,
    compute_position_anchor,
    process_frame,
)
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
    def __init__(self, hand: HandObservation | HandSample | None) -> None:
        self.hand = hand
        self.calls = 0
        self.closed = False

    def extract(self, _frame: np.ndarray) -> HandObservation | HandSample | None:
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


def _hand_at(x: float, y: float) -> HandSample:
    fingers = {
        name: FingerSample(name, (x, y), 0.0, 0.8)
        for name in ("index", "middle", "ring", "pinky")
    }
    return HandSample(
        wrist_xy=(x, y),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.9,
        fingers=fingers,
    )


def _image_point(canonical_x: float, canonical_y: float = 0.5) -> tuple[float, float]:
    return (100.0 * canonical_x, 50.0 * canonical_y)


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
        self.assertAlmostEqual(first.anchor.center_fret, 5.869500639052957)
        self.assertEqual(first.anchor.method, "mediapipe_calibrated_fret_map")
        self.assertIsNotNone(first.index_fret)
        self.assertEqual(first.index_fret, 6.0)
        self.assertAlmostEqual(first.index_fret_raw or 0.0, 5.932, places=3)

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
        self.assertIsNone(result.index_fret)
        self.assertEqual(result.hand_points, ())

    def test_off_neck_hand_is_suppressed_before_position_lock(self) -> None:
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(_hand_at(50.0, 55.0)),
            calibrator=_calibrator,
        )

        result = chain.process_frame(
            np.zeros((50, 100, 3), dtype=np.uint8), timestamp_s=0
        )

        self.assertTrue(result.neck_locked)
        self.assertIsNone(result.index_fret)
        self.assertEqual(result.anchor.confidence, 0.0)
        self.assertEqual(result.hand_points, ())

    def test_lone_index_overlap_does_not_admit_a_picking_hand(self) -> None:
        outside = _hand_at(50.0, 55.0)
        picking_hand = HandSample(
            wrist_xy=outside.wrist_xy,
            wrist_z=outside.wrist_z,
            is_left_hand=outside.is_left_hand,
            confidence=outside.confidence,
            fingers={
                **outside.fingers,
                "index": FingerSample("index", (50.0, 25.0), 0.0, 0.8),
            },
        )
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(picking_hand),
            calibrator=_calibrator,
        )

        result = chain.process_frame(
            np.zeros((50, 100, 3), dtype=np.uint8), timestamp_s=0
        )

        self.assertIsNone(result.index_fret)
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

    def test_just_past_wire_deadband_favors_fret_behind_the_wire(self) -> None:
        cfg = GuitarConfig()
        _, centers = _calibrator(OBBPredictions(), cfg)
        wires = _fret_wire_xs(centers)
        next_cell_width = wires[2] - wires[1]

        just_past = _fret_cell_from_canonical_x(
            float(wires[1] + 0.10 * next_cell_width), cfg, centers
        )
        clearly_inside_next = _fret_cell_from_canonical_x(
            float(wires[1] + 0.45 * next_cell_width), cfg, centers
        )

        self.assertEqual(just_past and just_past[0], 1)
        self.assertEqual(clearly_inside_next and clearly_inside_next[0], 2)

    def test_wire_deadband_scales_with_local_high_fret_width(self) -> None:
        cfg = GuitarConfig()
        _, centers = _calibrator(OBBPredictions(), cfg)
        wires = _fret_wire_xs(centers)
        next_cell_width = wires[10] - wires[9]

        just_past = _fret_cell_from_canonical_x(
            float(wires[9] + 0.10 * next_cell_width), cfg, centers
        )
        clearly_inside_next = _fret_cell_from_canonical_x(
            float(wires[9] + 0.45 * next_cell_width), cfg, centers
        )

        self.assertEqual(just_past and just_past[0], 9)
        self.assertEqual(clearly_inside_next and clearly_inside_next[0], 10)

    def test_descending_wire_axis_preserves_behind_wire_semantics(self) -> None:
        cfg = GuitarConfig()
        _, centers = _calibrator(OBBPredictions(), cfg)
        centers = centers[::-1]
        wires = _fret_wire_xs(centers)
        next_cell_width = wires[2] - wires[1]

        just_past = _fret_cell_from_canonical_x(
            float(wires[1] + 0.10 * next_cell_width), cfg, centers
        )

        self.assertEqual(just_past and just_past[0], 1)


class PositionAnchorGeometryTest(unittest.TestCase):
    def test_index_fret_uses_one_based_physical_cell_number(self) -> None:
        cfg = GuitarConfig()
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )
        centers = np.linspace(0.1, 0.9, cfg.max_fret + 1)
        hand = HandSample(
            wrist_xy=(10.0, 25.0),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={"index": FingerSample("index", (10.0, 25.0), 0.0, 0.8)},
        )

        index_fret = compute_index_fret(hand, homography, cfg, centers)

        self.assertAlmostEqual(index_fret or 0.0, 1.0)

    def test_extended_index_axis_corrects_barre_tip_in_next_cell(self) -> None:
        cfg = GuitarConfig()
        homography, centers = _calibrator(OBBPredictions(), cfg)
        wires = _fret_wire_xs(centers)
        second_width = wires[2] - wires[1]
        barre_x = float(wires[1] - 0.05 * (wires[1] - wires[0]))
        tip_x = float(wires[1] + 0.50 * second_width)
        hand = HandSample(
            wrist_xy=_image_point(barre_x),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={
                "index": FingerSample("index", _image_point(tip_x, 0.9), 0.0, 0.95)
            },
        )
        index_axis = (
            _image_point(barre_x, -0.2),
            _image_point(barre_x, 0.1),
            _image_point(barre_x, 0.5),
            _image_point(tip_x, 0.9),
        )

        contact_fret = compute_index_fret(
            hand,
            homography,
            cfg,
            centers,
            index_axis_xy=index_axis,
        )
        tip_coordinate = compute_index_fret_raw(hand, homography, cfg, centers)

        self.assertEqual(contact_fret, 1.0)
        self.assertGreater(tip_coordinate or 0.0, 1.5)

    def test_extended_index_not_across_neck_continues_to_use_tip(self) -> None:
        cfg = GuitarConfig()
        homography, centers = _calibrator(OBBPredictions(), cfg)
        wires = _fret_wire_xs(centers)
        second_width = wires[2] - wires[1]
        barre_x = float(wires[1] - 0.05 * (wires[1] - wires[0]))
        tip_x = float(wires[1] + 0.50 * second_width)
        hand = HandSample(
            wrist_xy=_image_point(barre_x),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={"index": FingerSample("index", _image_point(tip_x), 0.0, 0.95)},
        )

        contact_fret = compute_index_fret(
            hand,
            homography,
            cfg,
            centers,
            index_axis_xy=(_image_point(barre_x),) * 3 + (_image_point(tip_x),),
        )

        self.assertEqual(contact_fret, 2.0)

    def test_curled_index_continues_to_use_its_tip_contact(self) -> None:
        cfg = GuitarConfig()
        homography, centers = _calibrator(OBBPredictions(), cfg)
        wires = _fret_wire_xs(centers)
        second_width = wires[2] - wires[1]
        barre_x = float(wires[1] - 0.05 * (wires[1] - wires[0]))
        tip_x = float(wires[1] + 0.10 * second_width)
        hand = HandSample(
            wrist_xy=_image_point(barre_x),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={
                "index": FingerSample("index", _image_point(tip_x, 0.9), 0.0, 0.7)
            },
        )

        contact_fret = compute_index_fret(
            hand,
            homography,
            cfg,
            centers,
            index_axis_xy=(
                _image_point(barre_x, -0.2),
                _image_point(barre_x, 0.1),
                _image_point(barre_x, 0.5),
                _image_point(tip_x, 0.9),
            ),
        )

        self.assertEqual(contact_fret, 2.0)

    def test_descending_fret_map_preserves_fret_identity(self) -> None:
        hand = HandSample(
            wrist_xy=(50.0, 25.0),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={
                name: FingerSample(name, (50.0, 25.0), 0.0, 0.8)
                for name in ("index", "middle", "ring", "pinky")
            },
        )
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )
        centers = np.linspace(0.9, 0.1, 25)

        anchor = compute_position_anchor(hand, homography, GuitarConfig(), centers)

        self.assertAlmostEqual(anchor.center_fret, 12.0)
        self.assertEqual(anchor.method, "mediapipe_calibrated_fret_map")

    def test_fallback_maps_body_joint_to_fret_twelve_not_twenty_four(self) -> None:
        hand = HandSample(
            wrist_xy=(100.0, 25.0),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=1.0,
            fingers={
                name: FingerSample(name, (100.0, 25.0), 0.0, 0.8)
                for name in ("index", "middle", "ring", "pinky")
            },
        )
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )

        anchor = compute_position_anchor(hand, homography, GuitarConfig(), None)

        self.assertAlmostEqual(anchor.center_fret, 12.0)
        self.assertEqual(anchor.method, "mediapipe_rule18_fret12_fallback")

    def test_index_outside_neck_cross_axis_is_rejected(self) -> None:
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )

        index_fret = compute_index_fret(
            _hand_at(50.0, 55.0), homography, GuitarConfig(), None
        )

        self.assertIsNone(index_fret)

    def test_calibrated_coordinates_beyond_fret_cell_boundaries_are_rejected(
        self,
    ) -> None:
        cfg = GuitarConfig()
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )
        centers = np.linspace(0.1, 0.9, cfg.max_fret + 1)

        before_nut = compute_index_fret(_hand_at(5.0, 25.0), homography, cfg, centers)
        beyond_last_fret = compute_index_fret(
            _hand_at(95.0, 25.0), homography, cfg, centers
        )

        self.assertIsNone(before_nut)
        self.assertIsNone(beyond_last_fret)

    def test_anchor_uses_only_landmarks_that_are_on_the_neck(self) -> None:
        cfg = GuitarConfig()
        homography = Homography(
            H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=1.0,
            method="fixture",
        )
        hand = _hand_at(50.0, 55.0)
        hand = HandSample(
            wrist_xy=hand.wrist_xy,
            wrist_z=hand.wrist_z,
            is_left_hand=hand.is_left_hand,
            confidence=hand.confidence,
            fingers={
                **hand.fingers,
                "index": FingerSample("index", (50.0, 25.0), 0.0, 0.8),
            },
        )

        anchor = compute_position_anchor(hand, homography, cfg, None)

        expected = np.log1p(-0.5 * (1.0 - RULE_OF_18_RATIO**12)) / np.log(
            RULE_OF_18_RATIO
        )
        self.assertAlmostEqual(anchor.center_fret, expected)
        self.assertGreater(anchor.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
