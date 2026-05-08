"""Unit tests for the video-stage orchestrators.

Each function is exercised with a recording fake backend that captures
its inputs — no real model weights or video frames required.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.types import (
    FrameFingering,
    GuitarBBox,
    GuitarConfig,
    GuitarTrack,
    Homography,
)
from tabvision.video.fretboard import track_fretboard
from tabvision.video.guitar import detect_guitar
from tabvision.video.hand import track_hand

# ---------- helpers ----------


def _frame(idx: int) -> np.ndarray:
    """Tiny stub frame; content doesn't matter for orchestrator tests."""
    return np.full((4, 4, 3), idx, dtype=np.uint8)


def _frames(n: int):
    return [(i / 30.0, _frame(i)) for i in range(n)]


def _bbox(conf: float = 0.9) -> GuitarBBox:
    return GuitarBBox(x=0.0, y=0.0, w=10.0, h=10.0, confidence=conf)


def _homography(conf: float = 0.9) -> Homography:
    return Homography(H=np.eye(3), confidence=conf, method="fake")


def _fingering(t: float = 0.0) -> FrameFingering:
    return FrameFingering(
        t=t,
        finger_pos_logits=np.ones((4, 6, 25), dtype=np.float64),
        homography_confidence=0.9,
    )


# ---------- detect_guitar ----------


class _RecordingGuitarBackend:
    name = "recording_guitar"

    def __init__(self, returns):
        self._returns = list(returns)
        self.calls: list[np.ndarray] = []

    def detect(self, frame):
        self.calls.append(frame)
        return self._returns.pop(0)


def test_detect_guitar_calls_backend_per_frame():
    n = 5
    backend = _RecordingGuitarBackend([_bbox() for _ in range(n)])
    track = detect_guitar(_frames(n), backend, fps=30.0)
    assert len(backend.calls) == n
    assert isinstance(track, GuitarTrack)
    assert len(track.boxes) == n
    assert track.fps == 30.0


def test_detect_guitar_passes_none_to_smoother():
    """``None`` from a backend should propagate (becomes confidence-decayed last box)."""
    n = 3
    backend = _RecordingGuitarBackend([_bbox(), None, _bbox()])
    track = detect_guitar(_frames(n), backend, fps=30.0)
    assert len(track.boxes) == n
    # Middle frame keeps the previous detection with decayed confidence.
    assert track.boxes[1].confidence < track.boxes[0].confidence


# ---------- track_fretboard ----------


class _RecordingFretboardBackend:
    name = "recording_fretboard"

    def __init__(self, returns=None):
        self._returns = list(returns) if returns is not None else None
        self.calls: list[tuple[np.ndarray, GuitarBBox]] = []

    def detect(self, frame, guitar_box):
        self.calls.append((frame, guitar_box))
        if self._returns:
            return self._returns.pop(0)
        return _homography()


def test_track_fretboard_walks_frames_and_boxes_in_lockstep():
    n = 3
    boxes = [_bbox() for _ in range(n)]
    track = GuitarTrack(boxes=boxes, fps=30.0, stability_px=0.0)
    backend = _RecordingFretboardBackend()
    homs = track_fretboard(_frames(n), track, backend)
    assert len(homs) == n
    assert len(backend.calls) == n
    for (frame, bbox), expected_box in zip(backend.calls, boxes, strict=True):
        assert bbox is expected_box
        assert frame.shape == (4, 4, 3)


def test_track_fretboard_skips_frames_with_zero_confidence_bbox():
    """A confidence==0 bbox short-circuits the backend and emits a degenerate H."""
    boxes = [_bbox(0.9), _bbox(0.0), _bbox(0.9)]
    track = GuitarTrack(boxes=boxes, fps=30.0, stability_px=0.0)
    backend = _RecordingFretboardBackend()
    homs = track_fretboard(_frames(3), track, backend)
    assert len(homs) == 3
    assert len(backend.calls) == 2  # zero-conf frame skipped
    assert homs[1].confidence == 0.0
    assert homs[1].method == "skipped"


# ---------- track_hand ----------


class _RecordingHandBackend:
    name = "recording_hand"

    def __init__(self):
        self.calls: list[tuple[np.ndarray, Homography, GuitarConfig]] = []

    def detect(self, frame, H, cfg):  # noqa: N803 — H is the math name
        self.calls.append((frame, H, cfg))
        return _fingering(t=0.0)


def test_track_hand_stamps_timestamps():
    n = 4
    frames = _frames(n)
    homs = [_homography() for _ in range(n)]
    backend = _RecordingHandBackend()
    cfg = GuitarConfig()
    fings = track_hand(frames, homs, backend, cfg)
    assert len(fings) == n
    for ff, (t, _) in zip(fings, frames, strict=True):
        assert ff.t == t


def test_track_hand_skips_zero_confidence_homography():
    """Zero-confidence H short-circuits the backend, emits empty logits."""
    homs = [_homography(0.9), _homography(0.0), _homography(0.9)]
    backend = _RecordingHandBackend()
    cfg = GuitarConfig()
    fings = track_hand(_frames(3), homs, backend, cfg)
    assert len(fings) == 3
    assert len(backend.calls) == 2  # middle skipped
    # Middle fingering has zero homography_confidence and uniform-zero logits.
    assert fings[1].homography_confidence == 0.0
    assert (fings[1].finger_pos_logits == 0).all()
    # Timestamp still set for the skipped frame.
    assert fings[1].t == pytest.approx(1 / 30.0)
