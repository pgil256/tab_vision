"""Demuxing — see SPEC.md §3.3, §8.

Public entrypoint: ``demux(video_path) -> DemuxResult``.

    Wraps ffmpeg to extract the first audio stream as mono 22050 Hz audio plus
    a per-frame iterator.
The audio path is fully eager (loaded into memory); the frame iterator is
lazy.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterator, Sequence
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
    # MediaRecorder WebM (and other live/streaming containers) often omit the
    # duration from the header, so ffprobe reports N/A and _probe_metadata
    # returns 0.0. Recover it from the decoded audio length.
    if duration_s <= 0.0 and wav.size:
        duration_s = wav.size / float(sample_rate)
    frames = _frame_iterator(path, fps)

    return DemuxResult(
        wav=wav,
        sample_rate=sample_rate,
        duration_s=duration_s,
        fps=fps,
        frame_iterator=frames,
    )


def _as_float(value: str) -> float | None:
    """Parse a float, returning None for non-numeric tokens (e.g. ``N/A``)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_metadata(path: Path) -> tuple[float, float]:
    """Return (duration_s, fps) via ffprobe.

    Robust to audio-only inputs (no ``v:0`` stream, so no frame rate) and to
    containers that report ``N/A`` for frame rate or duration — notably the
    live/streaming WebM that browser ``MediaRecorder`` produces, which carries
    no duration in its header. ``fps`` falls back to ``0.0`` (the frame
    iterator and downstream video stack treat that as "no usable frames").
    """
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

    fps = 0.0
    duration_s = 0.0
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line == "N/A":
            continue
        if "/" in line:
            # r_frame_rate is a "num/den" fraction; ignore it if either side
            # is non-numeric or the denominator is zero (e.g. "0/0", "N/A").
            num, _, den = line.partition("/")
            n, d = _as_float(num), _as_float(den)
            if n is not None and d:
                fps = n / d
        else:
            v = _as_float(line)
            if v is not None:
                duration_s = v
    return duration_s, fps


def _extract_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode the first audio stream to mono float32 at the requested rate."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
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


def _run_ffprobe_json(
    path: Path,
    show_entries: str,
    *,
    select_streams: str | None = None,
) -> dict[str, object] | None:
    """Return one ffprobe JSON document, or ``None`` when probing is unavailable.

    Timeline probing is an accuracy enhancement rather than a hard demux
    requirement.  Keeping failures recoverable lets the frame iterator retain
    its deterministic constant-frame-rate clock on unusual/older ffprobe
    builds.
    """
    cmd = ["ffprobe", "-v", "error"]
    if select_streams is not None:
        cmd.extend(["-select_streams", select_streams])
    cmd.extend(["-show_entries", show_entries, "-of", "json", str(path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        value = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _finite_float(value: object) -> float | None:
    """Parse one finite ffprobe timestamp."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _probe_frame_clock(path: Path) -> tuple[tuple[float | None, ...], float]:
    """Return video frame PTS values on the decoded-audio time axis.

    ``ffmpeg`` writes the selected audio stream to a headerless PCM array, so
    sample zero corresponds to that stream's first decoded timestamp rather
    than to container time zero.  Subtracting the audio stream ``start_time``
    from each video frame's ``best_effort_timestamp_time`` puts both signals
    on the same clock, including files whose audio and video streams start at
    different non-zero times.

    The second return value is the stream-offset-aware start for the CFR
    fallback.  ``None`` entries deliberately preserve ffprobe frame indexes so
    an individual missing PTS cannot shift every subsequent decoded frame.
    """
    streams_doc = _run_ffprobe_json(path, "stream=codec_type,start_time")
    audio_start_s: float | None = None
    video_start_s: float | None = None
    audio_seen = False
    video_seen = False
    if streams_doc is not None:
        streams = streams_doc.get("streams")
        if isinstance(streams, list):
            for stream in streams:
                if not isinstance(stream, dict):
                    continue
                codec_type = stream.get("codec_type")
                start_s = _finite_float(stream.get("start_time"))
                if codec_type == "audio" and not audio_seen:
                    audio_start_s = start_s
                    audio_seen = True
                elif codec_type == "video" and not video_seen:
                    video_start_s = start_s
                    video_seen = True

    frames_doc = _run_ffprobe_json(
        path,
        "frame=best_effort_timestamp_time",
        select_streams="v:0",
    )
    raw_timestamps: list[float | None] = []
    if frames_doc is not None:
        frames = frames_doc.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                raw_timestamps.append(
                    _finite_float(frame.get("best_effort_timestamp_time"))
                    if isinstance(frame, dict)
                    else None
                )

    first_video_pts = next((value for value in raw_timestamps if value is not None), None)
    if audio_start_s is None:
        # Without an audio timestamp the cross-stream offset is unknowable.
        # Preserve historical behaviour by treating the first video timestamp
        # (or video stream start) as zero instead of leaking container time.
        audio_origin_s = (
            video_start_s
            if video_start_s is not None
            else first_video_pts
            if first_video_pts is not None
            else 0.0
        )
    else:
        audio_origin_s = audio_start_s

    timestamps_s = tuple(
        None if value is None else value - audio_origin_s for value in raw_timestamps
    )
    if video_start_s is not None:
        fallback_start_s = video_start_s - audio_origin_s
    else:
        fallback_start_s = next(
            (value for value in timestamps_s if value is not None),
            0.0,
        )
    return timestamps_s, fallback_start_s


def _resolve_frame_timestamp(
    frame_idx: int,
    *,
    timestamps_s: Sequence[float | None],
    fallback_start_s: float,
    fps: float,
    previous_s: float | None,
) -> float:
    """Choose an indexed PTS without letting one repair outrun later VFR PTS."""
    timestamp_s = timestamps_s[frame_idx] if frame_idx < len(timestamps_s) else None
    if timestamp_s is not None and (previous_s is None or timestamp_s > previous_s):
        return timestamp_s

    next_valid: tuple[int, float] | None = None
    for next_idx in range(frame_idx + 1, len(timestamps_s)):
        candidate = timestamps_s[next_idx]
        if candidate is None:
            continue
        if previous_s is None or candidate > previous_s:
            next_valid = next_idx, candidate
            break

    finite_fps = fps > 0.0 and math.isfinite(fps)
    if previous_s is not None and next_valid is not None:
        next_idx, next_s = next_valid
        intervals = next_idx - frame_idx + 1
        return previous_s + (next_s - previous_s) / intervals

    if previous_s is None and next_valid is not None:
        next_idx, next_s = next_valid
        if finite_fps:
            return next_s - (next_idx - frame_idx) / fps
        # With no usable frame rate, stay strictly before the future PTS.
        value = next_s
        for _ in range(next_idx - frame_idx):
            value = math.nextafter(value, -math.inf)
        return value

    if finite_fps:
        step_s = 1.0 / fps
        if previous_s is not None:
            return previous_s + step_s
        return fallback_start_s + frame_idx * step_s
    if previous_s is not None:
        return math.nextafter(previous_s, math.inf)
    return fallback_start_s


def _frame_iterator(path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Lazy frame iterator yielding ``(timestamp_s, frame_bgr)`` tuples.

    Decodes via :class:`cv2.VideoCapture`. Timestamps come from ffprobe's
    per-frame ``best_effort_timestamp_time`` and are normalized to decoded
    audio sample zero.  Missing or unusable PTS values fall back to a
    stream-offset-aware ``frame_index / fps`` clock.  We deliberately avoid
    ``CAP_PROP_POS_MSEC`` because it is unreliable on variable-frame-rate
    sources (notably iPhone HEVC).

    Releases the underlying capture when the generator is exhausted or
    garbage-collected.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendError(
            "frame iteration requires opencv-python; install with pip install '.[vision]'"
        ) from exc

    timestamps_s, fallback_start_s = _probe_frame_clock(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise BackendError(f"cv2.VideoCapture could not open {path}")

    try:
        frame_idx = 0
        previous_s: float | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = _resolve_frame_timestamp(
                frame_idx,
                timestamps_s=timestamps_s,
                fallback_start_s=fallback_start_s,
                fps=fps,
                previous_s=previous_s,
            )
            yield t, frame
            previous_s = t
            frame_idx += 1
    finally:
        cap.release()


__all__ = ["demux", "DEFAULT_SAMPLE_RATE"]
