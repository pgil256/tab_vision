"""Guitar bbox tracker — Phase 3 (SPEC §3.3, §7 Phase 3).

Lightweight EMA smoother over (x, y, w, h, rotation_deg). For typical
iPhone-on-lap shots the guitar barely moves; the tracker mostly hides
detector jitter and bridges single-frame missed detections.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from tabvision.types import GuitarBBox, GuitarTrack


def smooth_track(
    detections: Iterable[GuitarBBox | None],
    *,
    fps: float,
    alpha: float = 0.3,
    confidence_decay: float = 0.95,
) -> GuitarTrack:
    """Apply an exponential moving average across per-frame detections.

    None detections (frames where the model didn't fire) reuse the most
    recent smoothed box with confidence multiplied by ``confidence_decay``.
    If no detection has been seen yet, a None frame stays None and is
    represented as a zero-confidence stub box at origin.

    ``alpha`` controls smoothing weight: higher = faster reaction to
    detector changes, lower = smoother / lazier track. Default 0.3.

    Returns a ``GuitarTrack`` with one box per input frame and the
    ``stability_px`` field set to the std of the smoothed center.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    smoothed: list[GuitarBBox] = []
    last: GuitarBBox | None = None

    for det in detections:
        if det is None:
            if last is None:
                smoothed.append(GuitarBBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            else:
                smoothed.append(replace(last, confidence=last.confidence * confidence_decay))
            continue

        if last is None:
            new = det
        else:
            new = GuitarBBox(
                x=alpha * det.x + (1 - alpha) * last.x,
                y=alpha * det.y + (1 - alpha) * last.y,
                w=alpha * det.w + (1 - alpha) * last.w,
                h=alpha * det.h + (1 - alpha) * last.h,
                confidence=max(det.confidence, last.confidence * confidence_decay),
                rotation_deg=alpha * det.rotation_deg + (1 - alpha) * last.rotation_deg,
            )
        smoothed.append(new)
        last = new

    stability_px = _center_std(smoothed)
    return GuitarTrack(boxes=smoothed, fps=fps, stability_px=stability_px)


def _center_std(boxes: list[GuitarBBox]) -> float:
    """Standard deviation of (cx, cy) center pixel coordinates."""
    if not boxes:
        return 0.0
    centers_x = [b.x + b.w / 2.0 for b in boxes]
    centers_y = [b.y + b.h / 2.0 for b in boxes]
    n = len(boxes)
    mx = sum(centers_x) / n
    my = sum(centers_y) / n
    var = sum((cx - mx) ** 2 + (cy - my) ** 2 for cx, cy in zip(centers_x, centers_y)) / n
    return float(var**0.5)


__all__ = ["smooth_track"]
