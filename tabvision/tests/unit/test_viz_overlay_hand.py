"""Unit tests for the Phase 4 hand-fingering overlay script.

Covers the pure-Python heat-map LUT + the end-to-end render path with
a stubbed MediaPipe backend so no model weights / MediaPipe install
are needed for CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

# ruff: noqa: E402, I001
from scripts.viz.overlay_hand import (
    HEATMAP_LUT_BGR,
    _heat_colour,
)
from scripts.viz import overlay_hand
from tabvision.types import FrameFingering, GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS


# ----- _heat_colour -----


def test_heat_colour_endpoints_match_lut():
    assert _heat_colour(0.0) == HEATMAP_LUT_BGR[0]
    # The interpolation formula returns LUT[-1] only for t==1.0 exactly.
    assert _heat_colour(1.0) == HEATMAP_LUT_BGR[-1]


def test_heat_colour_interpolates_between_lut_entries():
    """Halfway between LUT[0] and LUT[1] yields the channel-wise mean."""
    n = len(HEATMAP_LUT_BGR) - 1
    half_t = 0.5 / n
    a = HEATMAP_LUT_BGR[0]
    b = HEATMAP_LUT_BGR[1]
    expected = tuple(int(a[i] + 0.5 * (b[i] - a[i])) for i in range(3))
    assert _heat_colour(half_t) == expected


def test_heat_colour_clamps_negative_input():
    assert _heat_colour(-0.5) == HEATMAP_LUT_BGR[0]


# ----- end-to-end render with stub backends -----


def _identity_homography() -> Homography:
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array([[40, 60], [120, 60], [120, 90], [40, 90]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)  # noqa: N806
    return Homography(H=H.astype(np.float64), confidence=0.85, method="keypoint")


def _make_fingering(cfg: GuitarConfig) -> FrameFingering:
    """A FrameFingering with a clear peak at (string=2, fret=5) for the index finger."""
    n_fingers = len(FRETTING_FINGERS)
    logits = np.full(
        (n_fingers, cfg.n_strings, cfg.max_fret + 1), -10.0, dtype=np.float64,
    )
    logits[0, 2, 5] = 0.0
    return FrameFingering(t=0.0, finger_pos_logits=logits, homography_confidence=0.8)


def _write_synthetic_video(path: Path, n_frames: int = 4) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(fourcc and path), fourcc, 30.0, (160, 120))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_render_overlay_hand_writes_output_with_stub_backends(tmp_path):
    """Stub fretboard + hand backends so the test runs without MediaPipe."""
    cfg = GuitarConfig(max_fret=12)
    homog = _identity_homography()
    fingering = _make_fingering(cfg)

    class _FakeFretboard:
        name = "keypoint"

        def detect(self, _frame, _bbox):
            return homog

    class _FakeHand:
        def __init__(self, *_args, **_kw):
            pass

        def detect(self, _frame, _h, _cfg):
            return fingering

        def close(self):
            pass

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_synthetic_video(src, n_frames=4)

    with (
        patch.object(overlay_hand, "KeypointFretboardBackend", _FakeFretboard),
        patch.object(overlay_hand, "GeometricFretboardBackend", _FakeFretboard),
        patch.object(overlay_hand, "MediaPipeHandBackend", _FakeHand),
    ):
        stats = overlay_hand.render_overlay(
            src, out,
            cfg=cfg,
            stride=1,
            show_progress=False,
        )

    assert out.exists()
    assert stats["frames_written"] == 4
    assert stats["hand_detected_frames"] == 4


def test_render_overlay_hand_handles_no_hand_detected(tmp_path):
    """When the hand backend returns confidence=0, the renderer still
    writes valid frames (with HUD only) and counts hand_detected_frames=0."""
    cfg = GuitarConfig(max_fret=12)
    homog = _identity_homography()
    n_fingers = len(FRETTING_FINGERS)
    empty_logits = np.full(
        (n_fingers, cfg.n_strings, cfg.max_fret + 1), -10.0, dtype=np.float64,
    )
    empty = FrameFingering(t=0.0, finger_pos_logits=empty_logits, homography_confidence=0.0)

    class _FakeFretboard:
        name = "keypoint"

        def detect(self, _frame, _bbox):
            return homog

    class _FakeHand:
        def __init__(self, *_args, **_kw):
            pass

        def detect(self, _frame, _h, _cfg):
            return empty

        def close(self):
            pass

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_synthetic_video(src, n_frames=4)

    with (
        patch.object(overlay_hand, "KeypointFretboardBackend", _FakeFretboard),
        patch.object(overlay_hand, "GeometricFretboardBackend", _FakeFretboard),
        patch.object(overlay_hand, "MediaPipeHandBackend", _FakeHand),
    ):
        stats = overlay_hand.render_overlay(
            src, out, cfg=cfg, stride=1, show_progress=False,
        )

    assert out.exists()
    assert stats["frames_written"] == 4
    assert stats["hand_detected_frames"] == 0
