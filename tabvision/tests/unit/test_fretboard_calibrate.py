"""Unit tests for ``tabvision.video.fretboard.calibrate`` (chunk-6 WS1).

Pure-numpy geometry: a hand-built homography + synthetic fret/nut OBBs placed at
known rule-of-18 canonical positions, so the calibration can be checked against
ground truth with no YOLO / video.
"""

# ruff: noqa: N803, N806 — H is the math-convention name for the homography matrix

from __future__ import annotations

import numpy as np
import pytest

from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import (
    RULE_OF_18_RATIO,
    calibrate_fret_xs,
    fit_fret_map,
    nut_at_high_canonical_x,
    project_to_canonical,
)
from tabvision.video.guitar.yolo_backend import OBBDetection, OBBPredictions

# ----- helpers -----


def _make_homography(
    nut_x: float = 100.0,
    body_x: float = 500.0,
    top_y: float = 200.0,
    bottom_y: float = 320.0,
) -> Homography:
    """Horizontal-fretboard homography: canonical [0,1]² → pixel space."""
    import cv2

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array(
        [[nut_x, top_y], [body_x, top_y], [body_x, bottom_y], [nut_x, bottom_y]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return Homography(H=H.astype(np.float64), confidence=0.9, method="keypoint")


def _canon_to_px(H: Homography, x: float, y: float) -> tuple[float, float]:
    proj = H.H @ np.array([x, y, 1.0])
    return float(proj[0] / proj[2]), float(proj[1] / proj[2])


def _fret_obb(H: Homography, canon_x: float, *, conf: float = 0.5) -> OBBDetection:
    cx, cy = _canon_to_px(H, canon_x, 0.5)
    return OBBDetection("fret", cx=cx, cy=cy, w=4.0, h=80.0, rotation_deg=0.0, confidence=conf)


def _synthetic_preds(
    H: Homography,
    *,
    x0: float,
    b: float,
    k_lo: int,
    k_hi: int,
    with_nut: bool = True,
    nut_x: float | None = None,
) -> OBBPredictions:
    """Fret OBBs at rule-of-18 wire positions ``x0 + b·(1 − r^k)`` for k in range."""
    frets = [_fret_obb(H, x0 + b * (1.0 - RULE_OF_18_RATIO**k)) for k in range(k_lo, k_hi + 1)]
    nut = []
    if with_nut:
        nx = x0 if nut_x is None else nut_x
        cx, cy = _canon_to_px(H, nx, 0.5)
        nut = [OBBDetection("nut", cx=cx, cy=cy, w=4.0, h=80.0, rotation_deg=0.0, confidence=0.6)]
    return OBBPredictions(frets=frets, neck=[], nut=nut)


# ----- project_to_canonical -----


def test_project_to_canonical_roundtrips():
    H = _make_homography()
    canon = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.8]])
    px = np.array([_canon_to_px(H, x, y) for x, y in canon])
    back = project_to_canonical(H, px)
    np.testing.assert_allclose(back, canon, atol=1e-9)


def test_project_to_canonical_rejects_bad_shape():
    H = _make_homography()
    with pytest.raises(ValueError, match="N, 2"):
        project_to_canonical(H, np.zeros((3,)))


# ----- nut_at_high_canonical_x (spacing decay) -----


def test_nut_at_low_x_when_gaps_shrink_toward_high_x():
    # Gaps shrink with x (nut at low x, body at high x) — the canonical case.
    xs = np.array([0.05, 0.15, 0.23, 0.29, 0.33, 0.36])
    assert nut_at_high_canonical_x(xs) is False


def test_nut_at_high_x_when_gaps_grow_toward_high_x():
    # Gaps grow with x (nut at high x): a mirror of the above.
    xs = np.array([0.05, 0.08, 0.12, 0.18, 0.26, 0.36])
    assert nut_at_high_canonical_x(xs) is True


def test_nut_spacing_too_few_wires_defaults_low():
    assert nut_at_high_canonical_x(np.array([0.1])) is False


# ----- fit_fret_map -----


def test_fit_fret_map_recovers_known_rule_of_18_scale():
    cfg = GuitarConfig(max_fret=19)
    x0, b = 0.02, 1.4
    wires = np.array([x0 + b * (1.0 - RULE_OF_18_RATIO**k) for k in range(1, 11)])
    fret_xs = fit_fret_map(wires, x0, cfg.max_fret)
    assert fret_xs is not None
    assert fret_xs.shape == (cfg.max_fret + 1,)
    expected = x0 + b * (1.0 - RULE_OF_18_RATIO ** (np.arange(cfg.max_fret + 1) + 0.5))
    np.testing.assert_allclose(fret_xs, expected, atol=1e-6)


def test_fit_fret_map_rejects_growing_gap_wires():
    # Gaps that GROW nut→body are physically impossible for a fretboard (frets
    # compress toward the body), so the fit must reject them.
    cfg = GuitarConfig(max_fret=19)
    wires = np.array([0.05, 0.12, 0.22, 0.35, 0.51, 0.70])  # accelerating gaps
    assert fit_fret_map(wires, 0.0, cfg.max_fret) is None


def test_fit_fret_map_too_few_wires_returns_none():
    assert fit_fret_map(np.array([0.1, 0.2, 0.28]), 0.0, 19) is None


def test_fit_fret_map_is_monotone():
    cfg = GuitarConfig(max_fret=19)
    x0, b = 0.0, 1.5
    wires = np.array([x0 + b * (1.0 - RULE_OF_18_RATIO**k) for k in range(2, 9)])
    fret_xs = fit_fret_map(wires, x0, cfg.max_fret)
    assert fret_xs is not None
    assert np.all(np.diff(fret_xs) > 0)


# ----- calibrate_fret_xs (end to end) -----


def test_calibrate_fret_xs_recovers_map_from_obbs():
    cfg = GuitarConfig(max_fret=19)
    H = _make_homography()
    x0, b = 0.02, 1.4
    preds = _synthetic_preds(H, x0=x0, b=b, k_lo=1, k_hi=10)
    fret_xs = calibrate_fret_xs(preds, H, cfg)
    assert fret_xs is not None
    expected = x0 + b * (1.0 - RULE_OF_18_RATIO ** (np.arange(cfg.max_fret + 1) + 0.5))
    np.testing.assert_allclose(fret_xs, expected, atol=2e-3)


def test_calibrate_fret_xs_handles_flipped_orientation():
    """Nut at high canonical x (body at x≈0): the recovered map is monotone
    *decreasing* but still anchored at the nut."""
    cfg = GuitarConfig(max_fret=19)
    H = _make_homography()
    x0, b = 0.98, -1.4  # nut near x=1, frets march toward x=0
    preds = _synthetic_preds(H, x0=x0, b=b, k_lo=1, k_hi=10, nut_x=x0)
    fret_xs = calibrate_fret_xs(preds, H, cfg)
    assert fret_xs is not None
    assert np.all(np.diff(fret_xs) < 0)  # decreasing toward the body
    assert fret_xs[0] == pytest.approx(x0 + b * (1.0 - RULE_OF_18_RATIO**0.5), abs=1e-2)


def test_calibrate_fret_xs_too_few_frets_returns_none():
    cfg = GuitarConfig(max_fret=19)
    H = _make_homography()
    preds = _synthetic_preds(H, x0=0.02, b=1.4, k_lo=1, k_hi=2)  # only 2 wires
    assert calibrate_fret_xs(preds, H, cfg) is None


def test_calibrate_fret_xs_zero_confidence_homography_returns_none():
    cfg = GuitarConfig(max_fret=19)
    H = _make_homography()
    preds = _synthetic_preds(H, x0=0.02, b=1.4, k_lo=1, k_hi=10)
    dead = Homography(H=H.H, confidence=0.0, method="keypoint")
    assert calibrate_fret_xs(preds, dead, cfg) is None


def test_calibrate_fret_xs_no_frets_returns_none():
    cfg = GuitarConfig(max_fret=19)
    H = _make_homography()
    assert calibrate_fret_xs(OBBPredictions(frets=[], neck=[], nut=[]), H, cfg) is None
