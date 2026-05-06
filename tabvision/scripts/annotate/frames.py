"""Frame-extraction helpers for the labeling tool.

Pure-Python (no Flask) so the unit tests can exercise frame sampling
and JPEG encoding without spinning up a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClipMeta:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def probe_clip(path: Path) -> ClipMeta:
    """Read the clip's metadata. Cheap — no decoding past the header."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {path}")
    try:
        return ClipMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Decode and return the BGR frame at ``frame_idx``.

    Raises ``IndexError`` if ``frame_idx`` is past the end of the clip.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IndexError(f"frame {frame_idx} not available in {path}")
        return frame
    finally:
        cap.release()


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """JPEG-encode a BGR frame for HTTP transport."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def evenly_spaced_frame_indices(n_frames: int, n_samples: int) -> list[int]:
    """Pick ``n_samples`` integer frame indices spread across [0, n_frames-1].

    Used by the fingering labeler to sample N frames per clip.
    """
    if n_samples <= 0:
        return []
    if n_frames <= 0:
        return []
    if n_samples >= n_frames:
        return list(range(n_frames))
    return [int(round(i)) for i in np.linspace(0, n_frames - 1, n_samples)]


def representative_frame_idx(meta: ClipMeta, target_s: float = 1.5) -> int:
    """Pick a representative frame for fretboard labeling.

    Default = 1.5 s in. The first ~1 s of clips is often a count-in or
    settling motion; 1.5 s is usually well inside steady playing.
    """
    target = int(target_s * meta.fps)
    return min(max(0, target), max(0, meta.n_frames - 1))


__all__ = [
    "ClipMeta",
    "encode_jpeg",
    "evenly_spaced_frame_indices",
    "probe_clip",
    "read_frame",
    "representative_frame_idx",
]
