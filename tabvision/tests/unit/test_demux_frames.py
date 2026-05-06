"""Unit tests for the demux frame iterator.

Skipped when opencv-python or ffmpeg aren't available — the demux stack
needs both. The fixture clip lives at ``tabvision/data/fixtures/test_a440.mp4``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tabvision.demux import demux

pytest.importorskip("cv2", reason="opencv-python needed for frame iteration")

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("ffmpeg/ffprobe not on PATH", allow_module_level=True)

FIXTURE = Path(__file__).parent.parent.parent / "data" / "fixtures" / "test_a440.mp4"

if not FIXTURE.exists():
    pytest.skip(f"fixture missing: {FIXTURE}", allow_module_level=True)


def _read_all_frames():
    result = demux(FIXTURE)
    return result, list(result.frame_iterator)


def test_frame_iterator_yields_at_least_one_frame():
    _result, frames = _read_all_frames()
    assert len(frames) > 0


def test_frames_are_bgr_uint8_arrays():
    _result, frames = _read_all_frames()
    t, frame = frames[0]
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3, f"expected 3 channels (BGR), got {frame.shape[2]}"


def test_first_frame_timestamp_is_zero():
    _result, frames = _read_all_frames()
    t0, _ = frames[0]
    assert t0 == 0.0


def test_timestamps_monotonic_and_match_fps():
    """Timestamps should be strictly increasing and step by ~1/fps."""
    result, frames = _read_all_frames()
    timestamps = [t for t, _ in frames]
    diffs = np.diff(timestamps)
    assert (diffs > 0).all(), "timestamps must be strictly increasing"
    expected_dt = 1.0 / result.fps
    # Allow tiny float drift; iterator computes t = i / fps so error is ULP-level.
    assert np.allclose(diffs, expected_dt, rtol=1e-6, atol=1e-9)


def test_total_frames_consistent_with_duration():
    """Frame count should be within ±1 frame of fps × duration_s."""
    result, frames = _read_all_frames()
    expected = result.duration_s * result.fps
    assert abs(len(frames) - expected) <= 1, f"expected ~{expected:.1f} frames, got {len(frames)}"


def test_iterator_is_single_pass():
    """The generator from a single demux() call exhausts after one walk."""
    result = demux(FIXTURE)
    first_pass = list(result.frame_iterator)
    second_pass = list(result.frame_iterator)
    assert len(first_pass) > 0
    assert second_pass == []  # generator already exhausted


def test_bad_path_raises():
    """Non-existent file → InvalidInputError before any frames are produced."""
    from tabvision.errors import InvalidInputError

    with pytest.raises(InvalidInputError):
        demux(Path("/no/such/video.mp4"))
