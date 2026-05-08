"""Unit tests for ``tabvision.video.fretboard.keypoint``.

The geometry — corner ordering, homography construction — is a pure
function and exercised here with synthetic OBB inputs that don't need
ultralytics or a trained model. Real inference is deferred to gated
integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.types import Homography
from tabvision.video.fretboard.keypoint import (
    _homography_from_quad,
    _obb_to_corners,
    _order_corners_by_neck_anatomy,
    predictions_to_homography,
)
from tabvision.video.guitar.yolo_backend import (
    CLASS_FRET,
    CLASS_NECK,
    CLASS_NUT,
    OBBDetection,
    OBBPredictions,
)

# ----- corner geometry -----


def test_obb_to_corners_axis_aligned():
    """A non-rotated OBB yields rectangular corners around (cx, cy)."""
    obb = OBBDetection(
        class_name=CLASS_NECK, cx=100.0, cy=50.0, w=80.0, h=20.0,
        rotation_deg=0.0, confidence=0.9,
    )
    corners = _obb_to_corners(obb)
    # Pre-rotation order: (+w/2,+h/2), (-w/2,+h/2), (-w/2,-h/2), (+w/2,-h/2)
    expected = np.array(
        [
            [140.0, 60.0],
            [60.0, 60.0],
            [60.0, 40.0],
            [140.0, 40.0],
        ]
    )
    assert np.allclose(corners, expected)


def test_obb_to_corners_rotated_90_deg():
    """Rotation by +90° swaps the role of w and h relative to image axes."""
    obb = OBBDetection(
        class_name=CLASS_NECK, cx=0.0, cy=0.0, w=80.0, h=20.0,
        rotation_deg=90.0, confidence=0.9,
    )
    corners = _obb_to_corners(obb)
    # Original (+40, +10) rotated 90° -> (-10, +40).  All 4 should be
    # consistent with (±40, ±10) -> (∓10, ±40).
    assert np.allclose(corners[0], [-10.0, 40.0], atol=1e-9)
    assert np.allclose(corners[1], [-10.0, -40.0], atol=1e-9)
    assert np.allclose(corners[2], [10.0, -40.0], atol=1e-9)
    assert np.allclose(corners[3], [10.0, 40.0], atol=1e-9)


# ----- corner ordering -----


def _make_horizontal_neck_corners(
    headstock_left_x: float = 50.0,
    body_right_x: float = 450.0,
    top_y: float = 100.0,
    bottom_y: float = 140.0,
) -> np.ndarray:
    """Build OBB corners for a perfectly horizontal neck with the
    headstock on the left of the image, output of _obb_to_corners() for
    an OBB rotated 0°. Corners ordered as the function emits them:
    (+w/2,+h/2), (-w/2,+h/2), (-w/2,-h/2), (+w/2,-h/2)."""
    return np.array(
        [
            [body_right_x, bottom_y],     # (+,+) -> body, bottom
            [headstock_left_x, bottom_y], # (-,+) -> nut, bottom
            [headstock_left_x, top_y],    # (-,-) -> nut, top
            [body_right_x, top_y],        # (+,-) -> body, top
        ],
        dtype=np.float64,
    )


def test_order_corners_uses_nut_to_disambiguate_short_edge():
    """With nut_xy provided, the ordering correctly identifies the nut side."""
    corners = _make_horizontal_neck_corners()
    nut_xy = (50.0, 120.0)  # at the headstock-left short edge
    ordered = _order_corners_by_neck_anatomy(corners, nut_xy)
    # ordered = [top_left, top_right, bottom_right, bottom_left]
    # top_left = nut side + smaller y
    assert ordered.shape == (4, 2)
    assert ordered[0].tolist() == [50.0, 100.0]   # top_left: nut, top
    assert ordered[1].tolist() == [450.0, 100.0]  # top_right: body, top
    assert ordered[2].tolist() == [450.0, 140.0]  # bottom_right: body, bot
    assert ordered[3].tolist() == [50.0, 140.0]   # bottom_left: nut, bot


def test_order_corners_handles_swapped_nut_to_the_right():
    """If the nut is on the right (unusual framing), short-edge identification flips."""
    corners = _make_horizontal_neck_corners()
    nut_xy = (450.0, 120.0)  # on the body-right short edge -> that's "nut"
    ordered = _order_corners_by_neck_anatomy(corners, nut_xy)
    # nut side is now at x=450, body side at x=50.
    assert ordered[0].tolist() == [450.0, 100.0]  # top_left = nut/top = (450, top)
    assert ordered[1].tolist() == [50.0, 100.0]   # top_right = body/top = (50, top)
    assert ordered[2].tolist() == [50.0, 140.0]
    assert ordered[3].tolist() == [450.0, 140.0]


def test_order_corners_falls_back_to_smaller_x_when_nut_missing():
    corners = _make_horizontal_neck_corners()
    ordered = _order_corners_by_neck_anatomy(corners, nut_xy=None)
    # No nut: heuristic picks smaller-x short edge (=headstock at 50).
    assert ordered[0].tolist() == [50.0, 100.0]
    assert ordered[3].tolist() == [50.0, 140.0]


# ----- homography construction -----


def test_homography_maps_unit_square_to_quad():
    """[0,1]² corners must map to the supplied quad corners."""
    quad = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]])
    H = _homography_from_quad(quad)  # noqa: N806
    src = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
    projected = (H @ src.T).T
    projected = projected[:, :2] / projected[:, 2:]
    assert np.allclose(projected, quad, atol=1e-6)


def test_homography_round_trip_with_rotation():
    """A rotated quad produces an invertible homography that round-trips."""
    quad = np.array([[100.0, 200.0], [300.0, 180.0], [310.0, 260.0], [110.0, 280.0]])
    H = _homography_from_quad(quad)  # noqa: N806
    H_inv = np.linalg.inv(H)  # noqa: N806
    projected = (H @ np.array([[0.5, 0.5, 1.0]]).T).T
    projected = projected[:, :2] / projected[:, 2:]
    back = (H_inv @ np.array([[projected[0, 0], projected[0, 1], 1.0]]).T).T
    back = back[:, :2] / back[:, 2:]
    assert np.allclose(back, [[0.5, 0.5]], atol=1e-6)


# ----- predictions_to_homography (integration of the pieces) -----


def _neck(cx, cy, w, h, rot=0.0, conf=0.9):
    return OBBDetection(class_name=CLASS_NECK, cx=cx, cy=cy, w=w, h=h,
                        rotation_deg=rot, confidence=conf)


def _nut(cx, cy, w=8.0, h=60.0, rot=0.0, conf=0.7):
    return OBBDetection(class_name=CLASS_NUT, cx=cx, cy=cy, w=w, h=h,
                        rotation_deg=rot, confidence=conf)


def _fret(cx, cy, w=8.0, h=40.0, rot=0.0, conf=0.5):
    return OBBDetection(class_name=CLASS_FRET, cx=cx, cy=cy, w=w, h=h,
                        rotation_deg=rot, confidence=conf)


def test_predictions_no_neck_returns_zero_confidence_identity():
    """Without a neck detection there's nothing to anchor the fretboard plane on."""
    preds = OBBPredictions(frets=[_fret(100, 100)], neck=[], nut=[])
    homog = predictions_to_homography(preds)
    assert isinstance(homog, Homography)
    assert homog.method == "keypoint"
    assert homog.confidence == 0.0
    assert np.allclose(homog.H, np.eye(3))


def test_predictions_with_neck_only_returns_valid_homography():
    """Just a neck is enough to produce a homography; nut and frets refine confidence."""
    preds = OBBPredictions(neck=[_neck(250, 200, 400, 60, conf=0.8)])
    homog = predictions_to_homography(preds)
    assert homog.method == "keypoint"
    assert homog.confidence == pytest.approx(0.8)
    # Map (0.5, 0.5) -> should land near the OBB centre.
    pt = np.array([[0.5, 0.5, 1.0]])
    proj = (homog.H @ pt.T).T
    proj = proj[:, :2] / proj[:, 2:]
    assert np.allclose(proj, [[250.0, 200.0]], atol=1e-3)


def test_predictions_with_neck_nut_and_frets_boosts_confidence():
    """Nut and >=4 frets each add 0.05 to confidence."""
    preds = OBBPredictions(
        neck=[_neck(250, 200, 400, 60, conf=0.7)],
        nut=[_nut(50, 200, conf=0.5)],
        frets=[_fret(100 + 30 * i, 200) for i in range(5)],
    )
    homog = predictions_to_homography(preds)
    assert homog.confidence == pytest.approx(0.7 + 0.05 + 0.05)


def test_predictions_confidence_clamped_to_one():
    """Confidence boosts cap at 1.0 even if neck conf is already 0.99."""
    preds = OBBPredictions(
        neck=[_neck(250, 200, 400, 60, conf=0.99)],
        nut=[_nut(50, 200, conf=0.5)],
        frets=[_fret(100 + 30 * i, 200) for i in range(5)],
    )
    homog = predictions_to_homography(preds)
    assert homog.confidence <= 1.0


def test_predictions_canonical_origin_maps_to_nut_when_nut_detected():
    """The canonical (0, 0) corner should land on the nut side, near the nut detection."""
    # Neck centred at (250, 200), w=400 => extends x ∈ [50, 450]. Nut on left at x=50.
    preds = OBBPredictions(
        neck=[_neck(250, 200, 400, 60, conf=0.8)],
        nut=[_nut(50, 200, conf=0.5)],
    )
    homog = predictions_to_homography(preds)
    # Map (0, 0.5) — on the nut-side, mid-string — should land near (50, 200).
    pt = np.array([[0.0, 0.5, 1.0]])
    proj = (homog.H @ pt.T).T
    proj = proj[:, :2] / proj[:, 2:]
    assert proj[0, 0] == pytest.approx(50.0, abs=1.0)
    assert proj[0, 1] == pytest.approx(200.0, abs=1.0)


def test_predictions_canonical_top_maps_to_smaller_image_y():
    """Canonical y=0 (top, high-E) maps to the smaller image-y edge."""
    preds = OBBPredictions(
        neck=[_neck(250, 200, 400, 60, conf=0.8)],
        nut=[_nut(50, 200, conf=0.5)],
    )
    homog = predictions_to_homography(preds)
    # (0, 0): nut-side, top
    pt_top = np.array([[0.0, 0.0, 1.0]])
    pt_bot = np.array([[0.0, 1.0, 1.0]])
    proj_top = (homog.H @ pt_top.T).T
    proj_top = proj_top[:, :2] / proj_top[:, 2:]
    proj_bot = (homog.H @ pt_bot.T).T
    proj_bot = proj_bot[:, :2] / proj_bot[:, 2:]
    assert proj_top[0, 1] < proj_bot[0, 1]
