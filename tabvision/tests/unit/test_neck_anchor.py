"""Unit tests for coarse hand-on-neck region estimation."""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.types import GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import FingerSample, HandSample
from tabvision.video.hand.neck_anchor import compute_neck_anchor


def _make_homography(
    nut_x: float = 100.0,
    body_x: float = 500.0,
    top_y: float = 200.0,
    bottom_y: float = 320.0,
) -> Homography:
    import cv2

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array(
        [[nut_x, top_y], [body_x, top_y], [body_x, bottom_y], [nut_x, bottom_y]],
        dtype=np.float32,
    )
    return Homography(
        H=cv2.getPerspectiveTransform(src, dst).astype(np.float64),
        confidence=0.9,
        method="unit",
    )


def _project(homography: Homography, canon_x: float, canon_y: float = 0.5) -> tuple[float, float]:
    pt = homography.H @ np.array([canon_x, canon_y, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def _hand_at_frets(
    homography: Homography, frets: dict[str, float], max_fret: int = 12
) -> HandSample:
    fingers: dict[str, FingerSample] = {}
    for name in ("index", "middle", "ring", "pinky"):
        fret = frets.get(name, 5.0)
        fingers[name] = FingerSample(
            name=name,
            tip_xy=_project(homography, fret / max_fret),
            tip_z=0.0,
            curl_ratio=1.0,
        )
    return HandSample(
        wrist_xy=_project(homography, 4.0 / max_fret),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.8,
        fingers=fingers,
    )


def test_compute_neck_anchor_estimates_hand_fret_span():
    cfg = GuitarConfig(max_fret=12)
    homography = _make_homography()
    hand = _hand_at_frets(
        homography, {"index": 3.0, "middle": 4.0, "ring": 5.0, "pinky": 6.0}
    )

    anchor = compute_neck_anchor(hand, homography, cfg)

    assert anchor.center_fret == pytest.approx(4.0, abs=0.25)
    assert anchor.min_fret <= 3.0
    assert anchor.max_fret >= 6.0
    assert anchor.contains(5)
    assert not anchor.contains(10)
    assert anchor.confidence > 0.4


def test_compute_neck_anchor_returns_zero_confidence_without_homography():
    cfg = GuitarConfig(max_fret=12)
    homography = Homography(H=np.eye(3, dtype=np.float64), confidence=0.0, method="none")

    anchor = compute_neck_anchor(None, homography, cfg)

    assert anchor.confidence == 0.0
    assert anchor.center_fret == 0.0
