"""Demuxing — see SPEC.md §3.3, §8.

Public entrypoint: ``demux(video_path) -> DemuxResult``.

Wraps ffmpeg to extract mono 22050 Hz audio plus a per-frame iterator.
The audio path is fully eager (loaded into memory); the frame iterator is
lazy.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError, InvalidInputError
from tabvision.types import DemuxResult

DEFAULT_SAMPLE_RATE = 22050  # Basic Pitch's native rate


def demux(video_path: str | Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> DemuxResult:
    """Extract audio + frames from a video file.

    Args:
        video_path: Path to mp4/mov video.
        sample_rate: Audio sample rate. 22050 Hz by default (Basic Pitch).

    Returns:
        DemuxResult with the audio loaded into memory and a frame iterator
        that lazily yields ``(timestamp_s, frame_bgr)`` tuples.
    """
    path = Path(video_path)
    if not path.exists():
        raise InvalidInputError(f"video file not found: {path}")
    if not shutil.which("ffmpeg"):
        raise BackendError("ffmpeg not on PATH; required by tabvision.demux")
    if not shutil.which("ffprobe"):
        raise BackendError("ffprobe not on PATH; required by tabvision.demux")

    duration_s, fps = _probe_metadata(path)
    wav = _extract_audio(path, sample_rate)
    frames = _frame_iterator(path, fps)

    return DemuxResult(
        wav=wav,
        sample_rate=sample_rate,
        duration_s=duration_s,
        fps=fps,
        frame_iterator=frames,
    )


def _probe_metadata(path: Path) -> tuple[float, float]:
    """Return (duration_s, fps) via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration:format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BackendError(f"ffprobe failed: {proc.stderr.strip()}")

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    fps = 0.0
    duration_s = 0.0
    for line in lines:
        if "/" in line:
            num, den = line.split("/")
            denom = float(den) or 1.0
            fps = float(num) / denom
        else:
            try:
                duration_s = float(line)
            except ValueError:
                pass
    return duration_s, fps


def _extract_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode audio to mono float32 at the requested sample rate."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise BackendError(f"ffmpeg audio decode failed: {proc.stderr.decode().strip()}")
    if not proc.stdout:
        raise BackendError("ffmpeg returned empty audio stream")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _frame_iterator(path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Lazy frame iterator yielding ``(timestamp_s, frame_bgr)`` tuples.

    Decodes via :class:`cv2.VideoCapture`. Timestamps are computed as
    ``frame_index / fps`` rather than ``CAP_PROP_POS_MSEC`` because the
    latter is unreliable on variable-frame-rate sources (notably iPhone
    HEVC). The caller passed the ffprobe-derived ``fps`` to
    :func:`demux`, so it's the source of truth here.

    Releases the underlying capture when the generator is exhausted or
    garbage-collected.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendError(
            "frame iteration requires opencv-python; install with pip install '.[vision]'"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise BackendError(f"cv2.VideoCapture could not open {path}")

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps if fps > 0 else 0.0
            yield t, frame
            frame_idx += 1
    finally:
        cap.release()


__all__ = ["demux", "DEFAULT_SAMPLE_RATE"]
