"""Unit tests for the Phase 3 debug-overlay scripts.

These exercise the rendering pipelines (annotate-a-frame helpers) on a
small synthetic video / fake detections so they run in CI without the
trained YOLO checkpoint or any video dataset. Real inference paths
(``render_overlay`` end-to-end with a real backend) are integration
territory and gated to ``-m guitar_eval`` / ``-m fretboard_eval``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # opencv-python is in [vision] extras

# Imports below are deferred past the importorskip above so the test
# module collects cleanly when opencv isn't installed.
# ruff: noqa: E402, I001
from scripts.viz.overlay_fretboard import _FRET_X_CANON, _project
from scripts.viz.overlay_fretboard import _draw as fb_draw
from scripts.viz.overlay_guitar import _draw_predictions, _obb_corners
from tabvision.types import Homography
from tabvision.video.guitar.yolo_backend import (
    CLASS_FRET,
    CLASS_NECK,
    CLASS_NUT,
    OBBDetection,
    OBBPredictions,
)

# ----- overlay_guitar -----


def _det(class_name, cx, cy, w, h, rot=0.0, conf=0.8) -> OBBDetection:
    return OBBDetection(
        class_name=class_name, cx=cx, cy=cy, w=w, h=h, rotation_deg=rot, confidence=conf
    )


def test_overlay_guitar_draws_neck_fret_and_nut():
    """All three classes should produce visible pixels on a clean frame."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    preds = OBBPredictions(
        neck=[_det(CLASS_NECK, 160, 120, 200, 30)],
        frets=[_det(CLASS_FRET, 100 + 20 * i, 120, 6, 25, conf=0.7) for i in range(3)],
        nut=[_det(CLASS_NUT, 80, 120, 4, 30)],
    )
    out = _draw_predictions(frame, preds)
    # Each colour should appear at least once. Frame started fully black,
    # so any non-zero pixel comes from drawing.
    assert (out != 0).any()
    # Green channel non-zero only where neck colour was drawn (BGR (0,255,0)).
    assert (out[..., 1] == 255).any()


def test_overlay_guitar_handles_empty_predictions():
    """No detections must not crash and must leave the frame mostly intact."""
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    out = _draw_predictions(frame, OBBPredictions())
    # Legend text overwrites a few pixels but the bulk of the frame should
    # be untouched; check that the centre remains the original colour.
    assert (out[120, 160] == 80).all()


def test_obb_corners_local_helper_matches_keypoint_module():
    """The viz helper duplicates the geometry; verify it agrees with the
    canonical implementation in video.fretboard.keypoint."""
    from tabvision.video.fretboard.keypoint import _obb_to_corners

    obb = _det(CLASS_NECK, 100.0, 50.0, 80.0, 20.0, rot=15.0)
    a = _obb_corners(obb)
    b = _obb_to_corners(obb)
    assert np.allclose(a, b)


# ----- overlay_fretboard -----


def _has_color(image: np.ndarray, bgr: tuple[int, int, int]) -> bool:
    """True if any pixel of ``image`` is exactly the given BGR colour."""
    b, g, r = bgr
    mask = (image[..., 0] == b) & (image[..., 1] == g) & (image[..., 2] == r)
    return bool(mask.any())


def _identity_homography(method: str = "keypoint") -> Homography:
    """A homography that maps canonical [0,1]² -> a fixed quad in image px."""
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array([[40, 60], [280, 50], [290, 180], [50, 200]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)  # noqa: N806
    return Homography(H=H.astype(np.float64), confidence=0.85, method=method)


def test_overlay_fretboard_draws_quad_and_string_grid():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    homog = _identity_homography()
    out = fb_draw(frame, homog, "keypoint", draw_fret_grid=False)
    # Magenta quadrilateral pixels should exist (BGR (255, 0, 255) — both B and R == 255).
    magenta_mask = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert magenta_mask.any()
    # The HUD writes anti-aliased white text in the lower-left; check that
    # something bright (>200) landed in the bottom-left band.
    hud_region = out[-30:, :180]
    assert (hud_region > 200).any()


def test_overlay_fretboard_draws_fret_lines_when_enabled():
    """With the fret grid enabled, the rendered output has substantially more
    grey-toned pixels (anti-aliased fret lines + fret-number labels) than
    without. Comparing counts is more robust than checking for an exact
    colour, since LINE_AA blends the line colour with the background."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    homog = _identity_homography()
    out_no = fb_draw(frame, homog, "keypoint", draw_fret_grid=False)
    out_yes = fb_draw(frame, homog, "keypoint", draw_fret_grid=True)

    def _grey_count(img: np.ndarray) -> int:
        # All three channels equal and non-zero -> a grey-toned pixel.
        m = (
            (img[..., 0] == img[..., 1])
            & (img[..., 1] == img[..., 2])
            & (img[..., 0] > 0)
        )
        return int(m.sum())

    # 12 fret guides + 12 number labels add hundreds of grey pixels relative
    # to the no-grid baseline (which has only HUD text edges).
    assert _grey_count(out_yes) > _grey_count(out_no) + 500


def test_overlay_fretboard_zero_confidence_falls_back_to_no_detection_hud():
    """A zero-confidence homography draws only the HUD message — no quad."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    homog = Homography(H=np.eye(3, dtype=np.float64), confidence=0.0, method="keypoint")
    out = fb_draw(frame, homog, "keypoint", draw_fret_grid=True)
    # No magenta quad pixels.
    magenta = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert not magenta.any()


def test_fret_x_positions_in_unit_interval_and_increasing():
    """The fret-x guides cover frets 1..12 monotonically inside [0, 1]."""
    assert len(_FRET_X_CANON) == 12
    assert _FRET_X_CANON[0] > 0.0
    assert _FRET_X_CANON[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(_FRET_X_CANON[i] < _FRET_X_CANON[i + 1] for i in range(11))


def test_project_points_through_identity_homography():
    """Projecting through cv2-built identity-square homography reproduces the dst."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dst = np.array([[10.0, 20.0], [110.0, 25.0], [115.0, 75.0], [15.0, 80.0]])
    H = cv2.getPerspectiveTransform(  # noqa: N806
        src.astype(np.float32), dst.astype(np.float32),
    )
    homog = Homography(H=H.astype(np.float64), confidence=1.0, method="keypoint")
    proj = _project(homog, src)
    assert np.allclose(proj, dst, atol=1e-4)


# ----- end-to-end smoke (synthetic video) -----


def _write_synthetic_video(path: Path, n_frames: int = 8) -> None:
    """Write a tiny black-frame mp4 we can stream through the renderer."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (160, 120))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_render_overlay_guitar_writes_output_with_fake_backend(tmp_path):
    """End-to-end render with a fake YOLO backend that emits one neck + one fret."""
    from scripts.viz import overlay_guitar

    # Build a stub backend whose predict_all() returns synthetic predictions.
    fake_preds = OBBPredictions(
        neck=[_det(CLASS_NECK, 80, 60, 100, 20)],
        frets=[_det(CLASS_FRET, 60, 60, 5, 15, conf=0.6)],
    )

    class _FakeBackend:
        def __init__(self, *_args, **_kw):
            pass

        def predict_all(self, _frame):
            return fake_preds

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_synthetic_video(src, n_frames=4)

    with patch.object(overlay_guitar, "YoloOBBBackend", _FakeBackend):
        stats = overlay_guitar.render_overlay(
            src, out,
            checkpoint=tmp_path / "ignored.pt",
            stride=1,
            show_progress=False,
        )

    assert out.exists()
    assert stats["frames_written"] == 4
    assert stats["detections_run"] == 4


def test_render_overlay_fretboard_writes_output_with_fake_backend(tmp_path):
    """End-to-end fretboard overlay using a stub backend."""
    from scripts.viz import overlay_fretboard

    homog = _identity_homography()

    class _FakeBackend:
        name = "keypoint"

        def __init__(self, *_args, **_kw):
            pass

        def detect(self, _frame, _bbox):
            return homog

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_synthetic_video(src, n_frames=4)

    with patch.object(overlay_fretboard, "_build_backends",
                      lambda _name, _cp: (_FakeBackend(), None)):
        stats = overlay_fretboard.render_overlay(
            src, out,
            backend_name="keypoint",
            stride=1,
            show_progress=False,
        )

    assert out.exists()
    assert stats["frames_written"] == 4
    # No fallback was needed because primary confidence (0.85) is above default 0.3.
    assert stats["fallback_used"] == 0
