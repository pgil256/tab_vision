"""Unit tests for tabvision.video.guitar.tracker."""

import pytest

from tabvision.types import GuitarBBox
from tabvision.video.guitar.tracker import smooth_track


def _box(x, y, w=100, h=200, conf=0.9, rot=0.0) -> GuitarBBox:
    return GuitarBBox(x=x, y=y, w=w, h=h, confidence=conf, rotation_deg=rot)


def test_smooth_track_returns_per_frame_box():
    track = smooth_track([_box(0, 0), _box(10, 5), _box(20, 10)], fps=30.0)
    assert len(track.boxes) == 3
    assert track.fps == 30.0


def test_smooth_track_averages_with_alpha():
    """alpha=0.5 mid-blend should land halfway between old and new."""
    track = smooth_track([_box(0, 0), _box(10, 0)], fps=30.0, alpha=0.5)
    assert track.boxes[1].x == pytest.approx(5.0, abs=1e-6)


def test_smooth_track_handles_none_with_decay():
    """A None frame after a real detection holds the last box with decayed confidence."""
    track = smooth_track([_box(0, 0, conf=0.9), None, _box(0, 0, conf=0.9)], fps=30.0, alpha=1.0)
    assert track.boxes[1].x == 0.0
    assert track.boxes[1].confidence < 0.9


def test_smooth_track_handles_leading_none():
    """A None at the very start emits a zero-confidence stub box."""
    track = smooth_track([None, _box(10, 0)], fps=30.0)
    assert track.boxes[0].confidence == 0.0


def test_alpha_must_be_in_range():
    with pytest.raises(ValueError):
        smooth_track([_box(0, 0)], fps=30.0, alpha=0.0)
    with pytest.raises(ValueError):
        smooth_track([_box(0, 0)], fps=30.0, alpha=1.5)


def test_stability_px_is_zero_for_static_box():
    """A perfectly static detection sequence should report ~0 stability."""
    track = smooth_track([_box(100, 50)] * 5, fps=30.0)
    assert track.stability_px == pytest.approx(0.0, abs=1e-9)
