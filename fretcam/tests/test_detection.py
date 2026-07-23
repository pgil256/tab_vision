from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np

from fretcam.detection import (
    DetectionChain,
    HandObservation,
    MediaPipeHandExtractor,
    _fret_cell_from_canonical_x,
    _fret_wire_xs,
    _hand_overlaps_neck,
    _has_outward_longitudinal_wrist,
    compute_finger_contacts,
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


def _landmark_observation() -> HandObservation:
    points = np.asarray(
        [
            (100, 180),
            (90, 160),
            (80, 145),
            (70, 135),
            (60, 125),
            (80, 140),
            (78, 115),
            (78, 95),
            (78, 78),
            (95, 135),
            (95, 105),
            (95, 82),
            (95, 62),
            (110, 138),
            (112, 110),
            (112, 90),
            (112, 73),
            (125, 145),
            (130, 122),
            (132, 105),
            (134, 90),
        ],
        dtype=np.float64,
    )
    points = (points - np.asarray((50.0, 55.0))) * 0.35
    z = np.linspace(0.0, -0.08, 21)
    indices = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    hand = HandSample(
        wrist_xy=tuple(points[0]),  # type: ignore[arg-type]
        wrist_z=float(z[0]),
        is_left_hand=True,
        confidence=0.9,
        fingers={
            name: FingerSample(
                name,
                tuple(points[axis[-1]]),  # type: ignore[arg-type]
                float(z[axis[-1]]),
                0.8,
            )
            for name, axis in indices.items()
        },
    )
    return HandObservation(
        hand=hand,
        finger_axes_xy={
            name: tuple(
                (float(points[index, 0]), float(points[index, 1])) for index in axis
            )
            for name, axis in indices.items()
        },
        finger_quality={name: 1.0 for name in indices},
        handedness_label="Right",
        handedness_score=0.9,
        landmarks_xy=tuple((float(point[0]), float(point[1])) for point in points),
        landmarks_z=tuple(float(value) for value in z),
        joint_quality=(1.0,) * 21,
    )


class TimestampedLandmarkExtractor:
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def extract_candidates(
        self,
        _frame: np.ndarray,
        *,
        timestamp_s: float,
        use_video: bool = True,
    ) -> tuple[HandObservation, ...]:
        del use_video
        self.timestamps.append(timestamp_s)
        return (_landmark_observation(),)

    def close(self) -> None:
        return None


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


def _geometry_fixture(
    body_joint_fret: int,
    *,
    include_strings: bool = True,
) -> tuple[np.ndarray, Homography, np.ndarray]:
    cfg = GuitarConfig()
    homography = Homography(
        H=np.asarray(
            ((600.0, 0.0, 20.0), (0.0, 120.0, 20.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        confidence=0.9,
        method="geometry_fixture",
    )
    body_fraction = 1.0 - RULE_OF_18_RATIO**body_joint_fret
    cell_frets = np.arange(cfg.max_fret + 1, dtype=np.float64) + 0.5
    centers = (1.0 - np.power(RULE_OF_18_RATIO, cell_frets)) / body_fraction
    wire_frets = np.arange(centers.size + 1, dtype=np.float64)
    wires = (1.0 - np.power(RULE_OF_18_RATIO, wire_frets)) / body_fraction
    frame = np.zeros((160, 640, 3), dtype=np.uint8)
    for wire in wires[(wires >= 0.0) & (wires <= 1.0)]:
        x = round(20.0 + 600.0 * float(wire))
        cv2.line(frame, (x, 20), (x, 140), (230, 230, 230), 2)
    if include_strings:
        for index in range(cfg.n_strings):
            y = round(20.0 + 120.0 * index / (cfg.n_strings - 1))
            cv2.line(frame, (20, y), (620, y), (180, 180, 180), 1)
    return frame, homography, centers


def _fixed_geometry_calibrator(
    homography: Homography,
    centers: np.ndarray,
):
    def calibrate(
        _predictions: OBBPredictions,
        _cfg: GuitarConfig,
    ) -> tuple[Homography, np.ndarray]:
        return homography, centers.copy()

    return calibrate


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

    def test_routine_detector_refresh_preserves_trusted_live_fret_axis(self) -> None:
        frame, homography, centers = _geometry_fixture(14)
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(None),
            detector_hz=2.0,
            calibrator=_fixed_geometry_calibrator(homography, centers),
            crop_hand=False,
        )

        chain.process_frame(frame, timestamp_s=0.0)
        chain.process_frame(frame, timestamp_s=0.21)
        trusted = chain._geometry_fret_centers
        self.assertIsNotNone(trusted)
        assert trusted is not None
        trusted = trusted.copy()

        refreshed = chain.process_frame(frame, timestamp_s=0.50)

        self.assertTrue(refreshed.detector_ran)
        self.assertIsNotNone(chain._geometry_fret_centers)
        np.testing.assert_allclose(chain._geometry_fret_centers, trusted, atol=5e-4)
        np.testing.assert_allclose(chain._fret_centers, trusted, atol=5e-4)

    def test_live_axis_without_independent_string_support_is_not_adopted(
        self,
    ) -> None:
        frame, homography, centers = _geometry_fixture(
            10,
            include_strings=False,
        )
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(None),
            detector_hz=0.1,
            calibrator=_fixed_geometry_calibrator(homography, centers),
            crop_hand=False,
        )

        first = chain.process_frame(frame, timestamp_s=0.0)
        second = chain.process_frame(frame, timestamp_s=0.21)

        self.assertEqual(first.string_refinement_support, 0.0)
        self.assertEqual(second.string_refinement_support, 0.0)
        self.assertIsNone(chain._geometry_fret_centers)
        self.assertEqual(chain._pending_body_joint_count, 0)

    def test_strong_repeated_geometry_adopts_common_body_joint_axes(self) -> None:
        for body_joint_fret in (12, 14, 17, 18, 19):
            with self.subTest(body_joint_fret=body_joint_fret):
                frame, homography, centers = _geometry_fixture(body_joint_fret)
                chain = DetectionChain(
                    detector=FakeDetector(),
                    hand_extractor=FakeHandExtractor(None),
                    detector_hz=0.1,
                    calibrator=_fixed_geometry_calibrator(homography, centers),
                    crop_hand=False,
                )

                chain.process_frame(frame, timestamp_s=0.0)
                result = chain.process_frame(frame, timestamp_s=0.21)

                self.assertGreater(result.string_refinement_support, 0.08)
                self.assertEqual(result.body_joint_fret, body_joint_fret)
                self.assertIsNotNone(chain._geometry_fret_centers)

    def test_detector_geometry_discontinuity_resets_trusted_live_axis(self) -> None:
        frame, homography, centers = _geometry_fixture(14)
        shifted = Homography(
            H=np.asarray(homography.H, dtype=np.float64).copy(),
            confidence=homography.confidence,
            method=homography.method,
        )
        shifted.H[0, 2] += 70.0
        calibrations = [(homography, centers), (shifted, centers)]

        def sequential_calibrator(
            _predictions: OBBPredictions,
            _cfg: GuitarConfig,
        ) -> tuple[Homography, np.ndarray]:
            selected_homography, selected_centers = calibrations.pop(0)
            return selected_homography, selected_centers.copy()

        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(None),
            detector_hz=2.0,
            calibrator=sequential_calibrator,
            crop_hand=False,
        )
        chain.process_frame(frame, timestamp_s=0.0)
        chain.process_frame(frame, timestamp_s=0.21)
        self.assertIsNotNone(chain._geometry_fret_centers)

        refreshed = chain.process_frame(frame, timestamp_s=0.50)

        self.assertTrue(refreshed.detector_ran)
        self.assertIsNone(chain._geometry_fret_centers)
        self.assertLessEqual(chain._pending_body_joint_count, 1)

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

    def test_failed_tight_crop_falls_back_to_full_neck_without_a_dropout(
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
        self.assertNotEqual(hands.shapes[0], frame.shape)
        self.assertNotEqual(hands.shapes[1], frame.shape)
        self.assertNotEqual(hands.shapes[2], frame.shape)
        self.assertGreaterEqual(hands.shapes[3][0], hands.shapes[2][0])
        self.assertGreaterEqual(hands.shapes[3][1], hands.shapes[2][1])
        self.assertTrue(result.hand_points)

    def test_neck_search_starts_immediately_then_alternates_full_frame(self) -> None:
        class AcquireOnFullFrame:
            def __init__(self) -> None:
                self.shapes: list[tuple[int, ...]] = []

            def extract(self, frame: np.ndarray) -> HandSample | None:
                self.shapes.append(frame.shape)
                return None if len(self.shapes) == 1 else _hand()

            def close(self) -> None:
                return None

        hands = AcquireOnFullFrame()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=True,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        first = chain.process_frame(frame, timestamp_s=0.0)
        second = chain.process_frame(frame, timestamp_s=0.1)

        self.assertFalse(first.hand_points)
        self.assertNotEqual(hands.shapes[0], frame.shape)
        self.assertEqual(hands.shapes[1], frame.shape)
        self.assertTrue(second.hand_points)

    def test_neck_geometry_selects_best_hand_before_handedness(self) -> None:
        class TwoHands:
            player_handedness = "right"

            def extract_candidates(
                self,
                _frame: np.ndarray,
                *,
                timestamp_s: float,
            ) -> tuple[HandObservation, HandObservation]:
                del timestamp_s
                return (
                    HandObservation(
                        hand=_hand_at(300.0, 150.0),
                        finger_axes_xy={},
                        handedness_label="Right",
                        handedness_score=0.99,
                    ),
                    HandObservation(
                        hand=_hand(),
                        finger_axes_xy={},
                        handedness_label="Left",
                        handedness_score=0.2,
                    ),
                )

            def close(self) -> None:
                return None

        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=TwoHands(),
            calibrator=_calibrator,
            crop_hand=False,
        )

        result = chain.process_frame(
            np.zeros((200, 400, 3), dtype=np.uint8),
            timestamp_s=0.0,
        )

        self.assertTrue(result.hand_points)
        self.assertLess(max(point.x for point in result.hand_points), 100.0)

    def test_temporal_landmarks_survive_a_brief_detector_miss(self) -> None:
        class OneDetection:
            def __init__(self) -> None:
                self.responses = iter((_landmark_observation(), None, None))

            def extract(
                self,
                _frame: np.ndarray,
            ) -> HandObservation | None:
                return next(self.responses)

            def close(self) -> None:
                return None

        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=OneDetection(),
            calibrator=_calibrator,
            crop_hand=False,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        detected = chain.process_frame(frame, timestamp_s=0.0)
        held = chain.process_frame(frame, timestamp_s=0.10)
        expired = chain.process_frame(frame, timestamp_s=0.19)

        self.assertEqual(detected.hand_source, "temporal_detector")
        self.assertTrue(detected.hand_points)
        self.assertEqual(held.hand_source, "temporal_held")
        self.assertTrue(held.hand_points)
        self.assertEqual(expired.hand_source, "none")
        self.assertFalse(expired.hand_points)

    def test_optical_tracking_runs_between_scheduled_mediapipe_updates(self) -> None:
        class CountingLandmarks:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, _frame: np.ndarray) -> HandObservation:
                self.calls += 1
                return _landmark_observation()

            def close(self) -> None:
                return None

        hands = CountingLandmarks()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=False,
            hand_detector_hz=5.0,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        detected = chain.process_frame(frame, timestamp_s=0.0)
        tracked = chain.process_frame(frame, timestamp_s=0.05)

        self.assertEqual(hands.calls, 1)
        self.assertEqual(detected.hand_source, "temporal_detector")
        self.assertEqual(tracked.hand_source, "temporal_held")
        self.assertTrue(tracked.hand_points)

    def test_hand_detector_10hz_is_drift_free_at_20fps(self) -> None:
        hands = TimestampedLandmarkExtractor()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=False,
            hand_detector_hz=10.0,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        results = [
            chain.process_frame(frame, timestamp_s=2.0 + index / 20.0)
            for index in range(21)
        ]

        self.assertEqual(
            hands.timestamps,
            [2.0 + index / 10.0 for index in range(11)],
        )
        self.assertTrue(
            all(result.hand_source == "temporal_detector" for result in results[::2])
        )
        self.assertTrue(
            all(
                result.hand_source in {"temporal_held", "temporal_optical_flow"}
                for result in results[1::2]
            )
        )

    def test_hand_detector_deadline_keeps_phase_after_a_late_frame(self) -> None:
        hands = TimestampedLandmarkExtractor()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=False,
            hand_detector_hz=10.0,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        for timestamp_s in (2.0, 2.05, 2.11, 2.15, 2.20, 2.25, 2.30):
            chain.process_frame(frame, timestamp_s=timestamp_s)

        self.assertEqual(hands.timestamps, [2.0, 2.11, 2.20, 2.30])

    def test_hand_detector_resets_on_between_update_clock_regression(self) -> None:
        hands = TimestampedLandmarkExtractor()
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=hands,
            calibrator=_calibrator,
            crop_hand=False,
            hand_detector_hz=10.0,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        results = [
            chain.process_frame(frame, timestamp_s=timestamp_s)
            for timestamp_s in (10.0, 10.05, 10.02, 10.07, 10.12)
        ]

        self.assertEqual(hands.timestamps, [10.0, 10.02, 10.12])
        self.assertEqual(results[2].hand_source, "temporal_detector")
        self.assertTrue(results[2].hand_points)

    def test_session_calibration_changes_contacts_and_fret_ticks(self) -> None:
        hand = _position_hand(5)
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(hand),
            calibrator=_calibrator,
            crop_hand=False,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        baseline = chain.process_frame(frame, timestamp_s=0.0)
        chain.set_position_calibration(scale=1.0, offset=1.0)
        calibrated = chain.process_frame(frame, timestamp_s=0.1)

        self.assertEqual(baseline.position_fret, 5.0)
        self.assertGreater(calibrated.position_fret or 0.0, 5.7)
        self.assertLess(calibrated.position_fret or 0.0, 6.1)
        self.assertEqual(
            [contact.fret for contact in calibrated.finger_contacts],
            [6, 7, 8, 9],
        )
        self.assertNotEqual(baseline.fret_ticks, calibrated.fret_ticks)


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

    def test_missing_mediapipe_visibility_is_neutral_not_zero_quality(self) -> None:
        landmark = SimpleNamespace(
            x=0.5,
            y=0.5,
            visibility=None,
            presence=None,
        )

        quality = MediaPipeHandExtractor._joint_landmark_quality(
            landmark,
        )

        self.assertAlmostEqual(quality, 1.0)

    def test_variable_crops_use_image_mode_not_video_tracker_state(self) -> None:
        class Landmarker:
            def __init__(self) -> None:
                self.video_calls = 0
                self.image_calls = 0

            def detect_for_video(
                self,
                _image: object,
                _timestamp_ms: int,
            ) -> SimpleNamespace:
                self.video_calls += 1
                return SimpleNamespace(hand_landmarks=[], handedness=[])

            def detect(self, _image: object) -> SimpleNamespace:
                self.image_calls += 1
                return SimpleNamespace(hand_landmarks=[], handedness=[])

            def close(self) -> None:
                return None

        landmarker = Landmarker()
        backend = SimpleNamespace(
            _load=lambda: landmarker,
            close=lambda: None,
        )
        extractor = MediaPipeHandExtractor(backend=backend)  # type: ignore[arg-type]
        frame = np.zeros((50, 100, 3), dtype=np.uint8)

        extractor.extract_candidates(frame, timestamp_s=1.0, use_video=True)
        extractor.extract_candidates(frame, timestamp_s=1.1, use_video=False)

        self.assertEqual(landmarker.video_calls, 1)
        self.assertEqual(landmarker.image_calls, 1)


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

    def test_longitudinal_wrist_rejects_picking_hand_at_either_neck_end(
        self,
    ) -> None:
        homography, _ = _calibrator(OBBPredictions(), GuitarConfig())

        def edge_hand(tip_x: float, wrist_x: float) -> HandSample:
            return HandSample(
                wrist_xy=_image_point(wrist_x, -0.8),
                wrist_z=0.0,
                is_left_hand=True,
                confidence=0.95,
                fingers={
                    name: FingerSample(
                        name,
                        _image_point(tip_x, 0.2 + index * 0.2),
                        0.0,
                        0.8,
                    )
                    for index, name in enumerate(("index", "middle", "ring", "pinky"))
                },
            )

        body_end = edge_hand(0.10, -0.10)
        nut_end = edge_hand(0.90, 1.10)

        self.assertTrue(_has_outward_longitudinal_wrist(body_end, homography))
        self.assertTrue(_has_outward_longitudinal_wrist(nut_end, homography))
        self.assertFalse(_hand_overlaps_neck(body_end, homography))
        self.assertFalse(_hand_overlaps_neck(nut_end, homography))

    def test_lateral_fretting_wrist_remains_valid_near_position_one(self) -> None:
        homography, _ = _calibrator(OBBPredictions(), GuitarConfig())
        hand = HandSample(
            wrist_xy=_image_point(0.60, 1.70),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=0.95,
            fingers={
                name: FingerSample(
                    name,
                    _image_point(tip_x, 0.2 + index * 0.2),
                    0.0,
                    0.8,
                )
                for index, (name, tip_x) in enumerate(
                    zip(
                        ("index", "middle", "ring", "pinky"),
                        (0.90, 0.82, 0.75, 0.66),
                        strict=True,
                    )
                )
            },
        )

        self.assertFalse(_has_outward_longitudinal_wrist(hand, homography))
        self.assertTrue(_hand_overlaps_neck(hand, homography))

    def test_neck_selection_skips_longitudinal_picking_hand(self) -> None:
        homography, _ = _calibrator(OBBPredictions(), GuitarConfig())
        picking = HandSample(
            wrist_xy=_image_point(-0.10, -0.8),
            wrist_z=0.0,
            is_left_hand=True,
            confidence=0.99,
            fingers={
                name: FingerSample(
                    name,
                    _image_point(0.10, 0.2 + index * 0.2),
                    0.0,
                    0.8,
                )
                for index, name in enumerate(("index", "middle", "ring", "pinky"))
            },
        )
        fretting = _position_hand(5)
        chain = DetectionChain(
            detector=FakeDetector(),
            hand_extractor=FakeHandExtractor(None),
            calibrator=_calibrator,
        )

        selected = chain._select_neck_hand((picking, fretting), homography)

        self.assertIs(selected, fretting)


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
        self.assertTrue(index.visible)
        self.assertFalse(index.pressing)
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

    def test_distal_segment_produces_finger_pad_contact(self) -> None:
        hand = _position_hand(5)
        index = hand.fingers["index"]
        axis = (
            _image_point(_physical_fret_x(5), 0.20),
            _image_point(_physical_fret_x(5), 0.20),
            _image_point(_physical_fret_x(5) - 0.01, 0.20),
            index.tip_xy,
        )

        contacts = compute_finger_contacts(
            hand,
            self.homography,
            self.cfg,
            self.centers,
            finger_axes_xy={"index": axis},
        )

        contact = next(item for item in contacts if item.name == "index")
        self.assertEqual(contact.contact_source, "finger_pad")
        self.assertTrue(contact.visible)
        self.assertTrue(contact.pressing)

    def test_partial_and_multiple_finger_barres_are_reported(self) -> None:
        base = _position_hand(5)
        fingers = {
            name: FingerSample(
                finger.name,
                finger.tip_xy,
                finger.tip_z,
                0.95 if name in {"index", "ring"} else finger.curl_ratio,
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
        index_x = _physical_fret_x(5)
        ring_x = _physical_fret_x(7)
        axes = {
            "index": tuple(_image_point(index_x, y) for y in (0.05, 0.50, 0.95)),
            "ring": tuple(_image_point(ring_x, y) for y in (0.20, 0.50, 0.80)),
        }

        contacts = compute_finger_contacts(
            hand,
            self.homography,
            self.cfg,
            self.centers,
            finger_axes_xy=axes,
        )
        by_name = {contact.name: contact for contact in contacts}

        self.assertTrue(by_name["index"].barre)
        self.assertEqual(by_name["index"].barre_strings, (1, 2, 3, 4, 5, 6))
        self.assertTrue(by_name["ring"].barre)
        self.assertEqual(len(by_name["ring"].barre_strings), 2)
        self.assertTrue(by_name["ring"].pressing)

    def test_lone_non_index_partial_barre_cannot_lock(self) -> None:
        base = _position_hand(5)
        fingers = {
            name: FingerSample(
                finger.name,
                finger.tip_xy,
                finger.tip_z,
                0.95 if name == "ring" else 0.99,
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
        ring_x = _physical_fret_x(7)
        ring_axis = tuple(
            _image_point(ring_x, canonical_y) for canonical_y in (0.20, 0.50, 0.80)
        )

        position, contacts, factors = self._solve(
            hand,
            finger_axes_xy={"ring": ring_axis},
        )

        ring = next(contact for contact in contacts if contact.name == "ring")
        self.assertTrue(ring.barre)
        self.assertTrue(ring.pressing)
        self.assertEqual(position, 5.0)
        self.assertEqual(factors.support_sufficiency, 0.0)
        self.assertEqual(factors.combined, 0.0)
        self.assertIn("too_few_contacts", factors.blockers)
        self.assertIn("low_confidence", factors.blockers)

    def test_fast_finger_motion_reduces_contact_evidence(self) -> None:
        hand = _position_hand(5)

        steady = compute_finger_contacts(
            hand,
            self.homography,
            self.cfg,
            self.centers,
            finger_stillness={"index": 1.0},
        )
        moving = compute_finger_contacts(
            hand,
            self.homography,
            self.cfg,
            self.centers,
            finger_stillness={"index": 0.0},
        )
        steady_index = next(item for item in steady if item.name == "index")
        moving_index = next(item for item in moving if item.name == "index")

        self.assertEqual(moving_index.motion_score, 0.0)
        self.assertLess(moving_index.pressing_score, steady_index.pressing_score)
        self.assertLess(moving_index.weight, steady_index.weight)

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
