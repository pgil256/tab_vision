"""Guitar detection — Phase 3 deliverable.

Public entrypoint: :func:`detect_guitar`. Walks a sequence of frames,
applies a :class:`tabvision.types.GuitarBackend` to each, and smooths
the resulting per-frame detections into a :class:`GuitarTrack`.

For pipeline use, ``tabvision.pipeline.run_pipeline`` does its own
single-pass loop and does not call this orchestrator (it would force a
re-decode of the video). This function is for callers who want
guitar-only output on a stand-alone clip.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tabvision.types import GuitarBackend, GuitarTrack
from tabvision.video.guitar.tracker import smooth_track


def detect_guitar(
    frames: Iterable[tuple[float, np.ndarray]],
    backend: GuitarBackend,
    *,
    fps: float,
) -> GuitarTrack:
    """Detect + smooth a per-frame guitar track from ``frames``.

    ``frames`` yields ``(timestamp_s, frame_bgr)`` tuples, matching what
    :func:`tabvision.demux.demux` produces. Per-bbox timestamps are
    recoverable by frame index since :class:`GuitarTrack` carries
    ``fps`` and the boxes list is in frame order.
    """
    detections = [backend.detect(frame) for _, frame in frames]
    return smooth_track(detections, fps=fps)


__all__ = ["detect_guitar"]
