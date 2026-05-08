"""Unit tests for ``scripts.annotate.frames``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

# ruff: noqa: E402, I001
from scripts.annotate.frames import (
    ClipMeta,
    encode_jpeg,
    evenly_spaced_frame_indices,
    probe_clip,
    read_frame,
    representative_frame_idx,
)


def _write_test_video(path: Path, n_frames: int = 30, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    try:
        for i in range(n_frames):
            # Bright unique value per frame so we can confirm read_frame is seeking.
            shade = int(10 + (i * 8) % 240)
            frame = np.full((120, 160, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


# ----- evenly_spaced_frame_indices -----


def test_evenly_spaced_indices_distributes_across_clip():
    assert evenly_spaced_frame_indices(100, 5) == [0, 25, 50, 74, 99]


def test_evenly_spaced_indices_caps_at_clip_length():
    assert evenly_spaced_frame_indices(3, 10) == [0, 1, 2]


def test_evenly_spaced_indices_empty_inputs():
    assert evenly_spaced_frame_indices(0, 5) == []
    assert evenly_spaced_frame_indices(10, 0) == []


# ----- representative_frame_idx -----


def test_representative_frame_idx_picks_about_1p5s_in():
    meta = ClipMeta(fps=30.0, n_frames=300, width=640, height=480)
    assert representative_frame_idx(meta) == 45


def test_representative_frame_idx_clamps_to_last_frame_for_short_clips():
    meta = ClipMeta(fps=30.0, n_frames=10, width=640, height=480)
    assert representative_frame_idx(meta) == 9


def test_representative_frame_idx_handles_zero_fps():
    meta = ClipMeta(fps=0.0, n_frames=100, width=640, height=480)
    # target_s * 0 = 0 → returns frame 0.
    assert representative_frame_idx(meta) == 0


# ----- probe / read -----


def test_probe_clip_returns_metadata(tmp_path):
    p = tmp_path / "vid.mp4"
    _write_test_video(p, n_frames=30, fps=30.0)
    meta = probe_clip(p)
    assert meta.n_frames == 30
    assert meta.fps == pytest.approx(30.0)
    assert meta.width == 160
    assert meta.height == 120
    assert meta.duration_s == pytest.approx(1.0)


def test_probe_clip_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_clip(tmp_path / "missing.mp4")


def test_read_frame_returns_bgr_array(tmp_path):
    p = tmp_path / "vid.mp4"
    _write_test_video(p, n_frames=10)
    f = read_frame(p, 5)
    assert f.shape == (120, 160, 3)
    assert f.dtype == np.uint8


def test_read_frame_raises_for_out_of_range_index(tmp_path):
    p = tmp_path / "vid.mp4"
    _write_test_video(p, n_frames=10)
    with pytest.raises(IndexError):
        read_frame(p, 9999)


def test_encode_jpeg_round_trip(tmp_path):
    """encode_jpeg → cv2.imdecode reproduces the frame approximately."""
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    blob = encode_jpeg(frame, quality=95)
    arr = np.frombuffer(blob, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape
    # JPEG is lossy; allow small tolerance in pixel diff.
    assert np.abs(decoded.astype(int) - frame.astype(int)).max() < 10
