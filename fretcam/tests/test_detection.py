from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np

from fretcam.detection import (
    DetectionChain,
    HandObservation,
    MediaPipeHandExtractor,
    _fret_cell_from_canonical_x,
    _fret_wire_xs,
    compute_index_fret,
    compute_index_fret_raw,
    compute_position_anchor,
    process_frame,
    solve_hand_position,
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
        self.reset_calls = 0
        self.closed = False

    def extract(self, _frame: np.ndarray) -> HandObservation | HandSample | None:
        self.calls += 1
        return self.hand

    def close(self) -> None:
        self.closed = True

    def reset(self) -> None:
        self.reset_calls += 1


class SelectableFakeHandExtractor(FakeHandExtractor):
    def __init__(self, hand: HandObservation | HandSample | None) -> None:
        super().__init__(hand)
        self.player_handedness = "right"
        self.shapes: list[tuple[int, ...]] = []

    def extract(self, frame: np.ndarray) -> HandObservation | HandSample | None:
        self.shapes.append(frame.shape)
        return super().extract(frame)

    def set_player_handedness(self, value: str) -> None:
        self.player_handedness = value


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


def _physical_fret_x(fret: int) -> float:
    cfg = GuitarConfig()
    _, centers = _calibrator(OBBPredictions(), cfg)
    wires = _fret_wire_xs(centers)
    return float((wires[fret - 1] + wires[fret]) / 2.0)


def _position_hand(
    position: int,
    *,
    include_index: bool = True,
    index_fret: int | None = None,
    index_curl: float = 0.70,
) -> HandSample:
    names = ("index", "middle", "ring", "pinky")
    fingers = {}
    for offset, name in enumerate(names):
        if name == "index" and not include_index:
            continue
        fret = position + offset
        curl = 0.70
        if name == "index":
            fret = position if index_fret is None else index_fret
            curl = index_curl
        fingers[name] = FingerSample(
            name,
            _image_point(_physical_fret_x(fret), 0.2 + offset * 0.2),
            0.0,
            curl,
        )
    return HandSample(
        wrist_xy=_image_point(_physical_fret_x(position), 0.5),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.95,
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


def _missing_calibrator(
    _predictions: OBBPredictions, _cfg: GuitarConfig
) -> tuple[Homography, None]:
    return (
        Homography(
            H=np.eye(3, dtype=np.float64),
            confidence=0.0,
            method="missing",
        ),
        None,
    )


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
        self.assertTrue(first.composite_available)

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
        hands = FakeHandExtractor(_hand())
        chain = DetectionChain(
            detector=detector,
            hand_extractor=hands,
            calibrator=_calibrator,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        chain.process_frame(frame, timestamp_s=10.0)
        chain.reset()
        after_reset = chain.process_frame(frame, timestamp_s=10.1)

        self.assertTrue(after_reset.detector_ran)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(hands.reset_calls, 1)

    def test_tracking_only_reset_retains_warmed_hand_backend(self) -> None:
        hands = FakeHandExtractor(_hand())
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        chain.process_frame(frame, timestamp_s=0.0)

        chain.reset_tracking()
        result = chain.process_frame(frame, timestamp_s=0.1)

        self.assertTrue(result.detector_ran)
        self.assertEqual(hands.reset_calls, 0)

    def test_handedness_change_preserves_warmed_models_and_board_geometry(
        self,
    ) -> None:
        detector = FakeDetector()
        hands = SelectableFakeHandExtractor(_hand())
        chain = DetectionChain(
            detector=detector,
            hand_extractor=hands,
            calibrator=_calibrator,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        chain.process_frame(frame, timestamp_s=0.0)

        chain.set_player_handedness("left")
        after_change = chain.process_frame(frame, timestamp_s=0.1)
        chain.set_player_handedness("left")

        self.assertEqual(hands.player_handedness, "left")
        self.assertEqual(hands.reset_calls, 0)
        self.assertEqual(hands.shapes[-1], frame.shape)
        self.assertTrue(after_change.neck_locked)
        self.assertFalse(after_change.detector_ran)
        self.assertEqual(detector.calls, 1)

    def test_background_detector_never_blocks_the_live_response_path(self) -> None:
        class BlockingDetector:
            def __init__(self) -> None:
                self.calls = 0
                self.started = Event()
                self.release = Event()
                self.finished = Event()

            def predict_all(self, _frame: np.ndarray) -> OBBPredictions:
                self.calls += 1
                self.started.set()
                self.release.wait(timeout=2.0)
                self.finished.set()
                return OBBPredictions()

        detector = BlockingDetector()
        chain = DetectionChain(
            detector=detector,
            hand_extractor=FakeHandExtractor(None),
            calibrator=_missing_calibrator,
            background_detector=True,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        caller = ThreadPoolExecutor(max_workers=1)
        try:
            response = caller.submit(
                chain.process_frame,
                frame,
                timestamp_s=0.0,
            ).result(timeout=0.25)
            self.assertFalse(response.neck_locked)
            self.assertTrue(detector.started.wait(timeout=1.0))
            self.assertEqual(detector.calls, 1)

            detector.release.set()
            self.assertTrue(detector.finished.wait(timeout=1.0))
            chain.process_frame(frame, timestamp_s=0.2)

            # A rejected completed job keeps its retry timestamp; it does not
            # trigger a synchronous second YOLO pass in the consuming frame.
            self.assertEqual(detector.calls, 1)
        finally:
            detector.release.set()
            caller.shutdown(wait=True, cancel_futures=True)
            chain.close()

    def test_background_detector_can_acquire_a_texture_poor_static_board(
        self,
    ) -> None:
        detector = FakeDetector()
        chain = DetectionChain(
            detector=detector,
            hand_extractor=FakeHandExtractor(None),
            calibrator=_calibrator,
            background_detector=True,
        )
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        try:
            first = chain.process_frame(frame, timestamp_s=0.0)
            assert chain._detector_future is not None
            chain._detector_future.result(timeout=1.0)
            acquired = chain.process_frame(frame.copy(), timestamp_s=1.0)

            self.assertFalse(first.neck_locked)
            self.assertTrue(acquired.neck_locked)
            self.assertEqual(acquired.geometry_status, "detected")
            self.assertEqual(detector.calls, 1)
        finally:
            chain.close()

    def test_tracking_reset_does_not_wait_for_an_obsolete_detector_job(self) -> None:
        class BlockingDetector:
            def __init__(self) -> None:
                self.started = Event()
                self.release = Event()

            def predict_all(self, _frame: np.ndarray) -> OBBPredictions:
                self.started.set()
                self.release.wait(timeout=2.0)
                return OBBPredictions()

        detector = BlockingDetector()
        chain = DetectionChain(
            detector=detector,
            hand_extractor=FakeHandExtractor(None),
            calibrator=_missing_calibrator,
            background_detector=True,
        )
        caller = ThreadPoolExecutor(max_workers=1)
        try:
            chain.process_frame(
                np.zeros((50, 100, 3), dtype=np.uint8),
                timestamp_s=0.0,
            )
            self.assertTrue(detector.started.wait(timeout=1.0))
            caller.submit(chain.reset_tracking).result(timeout=0.25)
        finally:
            detector.release.set()
            caller.shutdown(wait=True, cancel_futures=True)
            chain.close()

    def test_invalid_frame_is_rejected(self) -> None:
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(_hand()),
            calibrator=_calibrator,
        )
        with self.assertRaisesRegex(ValueError, "BGR frame"):
            chain.process_frame(np.zeros((50, 100), dtype=np.uint8))

    def test_failed_hand_crop_falls_back_to_full_frame_without_a_dropout(
        self,
    ) -> None:
        class CropThenFullExtractor:
            def __init__(self) -> None:
                self.shapes: list[tuple[int, ...]] = []
                self.responses = iter((_hand(), _hand(), None, _hand()))

            def extract(self, frame: np.ndarray) -> HandObservation | HandSample | None:
                self.shapes.append(frame.shape)
                return next(self.responses)

            def close(self) -> None:
                return None

        hands = CropThenFullExtractor()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=True,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        chain.process_frame(frame, timestamp_s=0.0)
        chain.process_frame(frame, timestamp_s=0.1)
        result = chain.process_frame(frame, timestamp_s=0.2)

        self.assertEqual(len(hands.shapes), 4)
        self.assertEqual(hands.shapes[0], frame.shape)
        self.assertEqual(hands.shapes[1], frame.shape)
        self.assertLess(hands.shapes[2][0], frame.shape[0])
        self.assertLess(hands.shapes[2][1], frame.shape[1])
        self.assertEqual(hands.shapes[3], frame.shape)
        self.assertTrue(result.hand_points)


class HandednessSelectionTest(unittest.TestCase):
    @staticmethod
    def _landmarks(spread: float) -> list[SimpleNamespace]:
        landmarks = [SimpleNamespace(x=0.5) for _ in range(21)]
        for index, x in zip((8, 12, 16, 20), (0.5, 0.5 + spread, 0.5, 0.5)):
            landmarks[index] = SimpleNamespace(x=x)
        return landmarks

    @staticmethod
    def _handedness(name: str, score: float) -> list[SimpleNamespace]:
        return [SimpleNamespace(category_name=name, score=score)]

    def test_right_handed_player_preserves_first_right_label_selector(self) -> None:
        extractor = MediaPipeHandExtractor(
            backend=SimpleNamespace(),  # type: ignore[arg-type]
            player_handedness="right",
        )

        selected = extractor._select_hand(
            [self._landmarks(0.1), self._landmarks(0.2)],
            [self._handedness("Right", 0.1), self._handedness("Right", 0.99)],
        )

        self.assertEqual(selected, 0)

    def test_left_handed_player_selects_mirrored_label(self) -> None:
        extractor = MediaPipeHandExtractor(
            backend=SimpleNamespace(),  # type: ignore[arg-type]
            player_handedness="left",
        )

        selected = extractor._select_hand(
            [self._landmarks(0.1), self._landmarks(0.2)],
            [self._handedness("Right", 0.9), self._handedness("Left", 0.8)],
        )

        self.assertEqual(selected, 1)


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


class MultiFingerPositionSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = GuitarConfig()
        self.homography, self.centers = _calibrator(OBBPredictions(), self.cfg)

    def _solve(
        self,
        hand: HandSample,
        *,
        homography: Homography | None = None,
        finger_axes_xy: dict[str, tuple[tuple[float, float], ...]] | None = None,
        freshness: float = 1.0,
        geometry_stability: float = 1.0,
    ):
        board = self.homography if homography is None else homography
        anchor = compute_position_anchor(hand, board, self.cfg, self.centers)
        return solve_hand_position(
            hand,
            board,
            self.cfg,
            self.centers,
            anchor,
            finger_axes_xy=finger_axes_xy,
            freshness=freshness,
            geometry_stability=geometry_stability,
        )

    def test_coherent_four_finger_shape_solves_classical_position(self) -> None:
        position, contacts, factors = self._solve(_position_hand(5))

        self.assertEqual(position, 5.0)
        self.assertEqual(
            [(contact.name, contact.fret) for contact in contacts],
            [("index", 5), ("middle", 6), ("ring", 7), ("pinky", 8)],
        )
        self.assertGreater(factors.finger_agreement, 0.75)
        self.assertGreater(factors.combined, 0.20)
        self.assertNotIn("finger_conflict", factors.blockers)

    def test_playable_stretch_keeps_the_index_defined_position(self) -> None:
        base = _position_hand(5)
        fingers = {
            name: FingerSample(
                name,
                _image_point(_physical_fret_x(fret), 0.2 + index * 0.2),
                0.0,
                0.70,
            )
            for index, (name, fret) in enumerate(
                zip(
                    ("index", "middle", "ring", "pinky"),
                    (5, 7, 8, 10),
                    strict=True,
                )
            )
        }
        stretched = HandSample(
            wrist_xy=base.wrist_xy,
            wrist_z=base.wrist_z,
            is_left_hand=base.is_left_hand,
            confidence=base.confidence,
            fingers=fingers,
        )

        position, _, factors = self._solve(stretched)

        self.assertAlmostEqual(position or 0.0, 5.0, places=6)
        self.assertGreater(factors.combined, 0.20)
        self.assertNotIn("finger_conflict", factors.blockers)

    def test_solver_exposes_a_bounded_continuous_calibration_residual(self) -> None:
        base = _position_hand(5)
        fingers = {}
        for index, (name, fret) in enumerate(
            zip(
                ("index", "middle", "ring", "pinky"),
                (5, 6, 7, 8),
                strict=True,
            )
        ):
            center = _physical_fret_x(fret)
            next_center = _physical_fret_x(fret + 1)
            shifted = center + 0.20 * (next_center - center)
            fingers[name] = FingerSample(
                name,
                _image_point(shifted, 0.2 + index * 0.2),
                0.0,
                0.70,
            )
        shifted_hand = HandSample(
            wrist_xy=base.wrist_xy,
            wrist_z=base.wrist_z,
            is_left_hand=base.is_left_hand,
            confidence=base.confidence,
            fingers=fingers,
        )

        position, _, factors = self._solve(shifted_hand)

        self.assertIsNotNone(position)
        self.assertGreater(position or 0.0, 5.0)
        self.assertLess(position or 0.0, 5.45)
        self.assertGreater(factors.combined, 0.20)

    def test_hovering_index_does_not_override_three_curled_fingers(self) -> None:
        hand = _position_hand(5, index_fret=14, index_curl=0.99)

        position, contacts, factors = self._solve(hand)

        index = next(contact for contact in contacts if contact.name == "index")
        self.assertLess(index.weight, 0.05)
        self.assertEqual(position, 5.0)
        self.assertGreater(factors.combined, 0.20)

    def test_missing_index_still_solves_from_middle_ring_and_pinky(self) -> None:
        position, contacts, factors = self._solve(
            _position_hand(5, include_index=False)
        )

        self.assertEqual(position, 5.0)
        self.assertEqual(
            [contact.name for contact in contacts],
            ["middle", "ring", "pinky"],
        )
        self.assertEqual(factors.on_neck, 1.0)
        self.assertGreater(factors.combined, 0.20)

    def test_conflicting_contacts_reduce_confidence_and_add_diagnostic(
        self,
    ) -> None:
        coherent_hand = _position_hand(5)
        conflicting_hand = HandSample(
            wrist_xy=coherent_hand.wrist_xy,
            wrist_z=coherent_hand.wrist_z,
            is_left_hand=coherent_hand.is_left_hand,
            confidence=coherent_hand.confidence,
            fingers={
                name: FingerSample(
                    name,
                    _image_point(_physical_fret_x(fret), 0.2 + offset * 0.2),
                    0.0,
                    0.70,
                )
                for offset, (name, fret) in enumerate(
                    zip(
                        ("index", "middle", "ring", "pinky"),
                        (5, 9, 13, 17),
                        strict=True,
                    )
                )
            },
        )

        _, _, coherent = self._solve(coherent_hand)
        _, _, conflicting = self._solve(conflicting_hand)

        self.assertLess(conflicting.finger_agreement, coherent.finger_agreement)
        self.assertLess(conflicting.combined, coherent.combined)
        self.assertLess(conflicting.combined, 0.20)
        self.assertIn("finger_conflict", conflicting.blockers)
        self.assertIn("low_confidence", conflicting.blockers)

    def test_extended_index_barre_uses_axis_contact_for_position(self) -> None:
        hand = _position_hand(5, index_fret=6, index_curl=0.95)
        barre_x = _physical_fret_x(5)
        axis = tuple(
            _image_point(barre_x, canonical_y) for canonical_y in (0.05, 0.50, 0.95)
        )

        position, contacts, factors = self._solve(
            hand,
            finger_axes_xy={"index": axis},
        )

        index = next(contact for contact in contacts if contact.name == "index")
        self.assertTrue(index.barre)
        self.assertEqual(index.fret, 5)
        self.assertAlmostEqual(index.raw_fret, 6.0, places=1)
        self.assertEqual(position, 5.0)
        self.assertGreaterEqual(factors.support_sufficiency, 0.85)

    def test_lone_barre_abstains_when_the_detected_hand_is_boundary_clipped(
        self,
    ) -> None:
        base = _position_hand(5, index_curl=0.95)
        fingers = {
            name: FingerSample(
                name,
                (finger.tip_xy if name != "pinky" else (finger.tip_xy[0], 60.0)),
                finger.tip_z,
                0.95 if name == "index" else 0.99,
            )
            for name, finger in base.fingers.items()
        }
        hand = HandSample(
            wrist_xy=base.wrist_xy,
            wrist_z=base.wrist_z,
            is_left_hand=base.is_left_hand,
            confidence=base.confidence,
            fingers=fingers,
        )
        barre_x = _physical_fret_x(5)
        axis = tuple(
            _image_point(barre_x, canonical_y) for canonical_y in (0.05, 0.50, 0.95)
        )

        position, _, factors = self._solve(
            hand,
            finger_axes_xy={"index": axis},
        )

        self.assertEqual(position, 5.0)
        self.assertEqual(factors.on_neck, 0.75)
        self.assertEqual(factors.combined, 0.0)
        self.assertIn("boundary_clipped", factors.blockers)
        self.assertIn("low_confidence", factors.blockers)

    def test_geometry_quality_factors_monotonically_reduce_confidence(self) -> None:
        hand = _position_hand(5)
        _, _, healthy = self._solve(hand)
        _, _, aged = self._solve(hand, freshness=0.25)
        _, _, unstable = self._solve(hand, geometry_stability=0.40)
        low_board = Homography(
            H=self.homography.H,
            confidence=0.40,
            method=self.homography.method,
        )
        _, _, weak_board = self._solve(hand, homography=low_board)

        self.assertLess(aged.combined, healthy.combined)
        self.assertLess(unstable.combined, healthy.combined)
        self.assertLess(weak_board.combined, healthy.combined)
        self.assertEqual(aged.freshness, 0.25)
        self.assertEqual(unstable.stability, 0.40)
        self.assertEqual(weak_board.board, 0.40)


if __name__ == "__main__":
    unittest.main()
