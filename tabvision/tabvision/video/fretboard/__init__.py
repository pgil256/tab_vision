"""Fretboard rectification — Phase 3 deliverable.

Public entrypoint: :func:`track_fretboard`. Walks frames + a
:class:`GuitarTrack` in lockstep, calling a
:class:`tabvision.types.FretboardBackend` per frame to recover an image
→ canonical homography. Frames where the guitar bbox carries
zero confidence skip the backend and emit a degenerate identity
homography flagged ``method="skipped"``.

Strategy: geometric (Hough + RANSAC) primary, keypoint (YOLO-pose)
fallback — selection happens at backend construction time.

For pipeline use, ``tabvision.pipeline.run_pipeline`` runs its own
single-pass loop; this function is for stand-alone fretboard-only output.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tabvision.types import FretboardBackend, GuitarTrack, Homography


def track_fretboard(
    frames: Iterable[tuple[float, np.ndarray]],
    guitar_track: GuitarTrack,
    backend: FretboardBackend,
) -> list[Homography]:
    """Per-frame homographies aligned with ``guitar_track.boxes``.

    Length of the returned list matches the number of consumed frames.
    """
    homographies: list[Homography] = []
    for (_t, frame), bbox in zip(frames, guitar_track.boxes, strict=False):
        if bbox.confidence <= 0.0:
            homographies.append(Homography(H=np.eye(3), confidence=0.0, method="skipped"))
            continue
        homographies.append(backend.detect(frame, bbox))
    return homographies


__all__ = ["track_fretboard"]
