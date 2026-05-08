"""Fretboard homography tracker — Phase 3 (SPEC §3.3).

Smooths the per-frame homography across frames; extrapolates through
single-frame detection failures. Implementation: per-element EMA on the
flattened 3×3 matrix. Crude but matches the v0 "tracking" function which
also blends consecutive detections without a true Kalman filter.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from tabvision.types import Homography


def smooth_homography_track(
    detections: Iterable[Homography],
    *,
    alpha: float = 0.3,
    confidence_decay: float = 0.9,
    min_confidence_for_update: float = 0.2,
) -> list[Homography]:
    """EMA smoothing over a stream of per-frame ``Homography`` outputs.

    Frames whose detection confidence is below ``min_confidence_for_update``
    are treated as missing and reuse the most recent smoothed homography
    (with confidence multiplied by ``confidence_decay``, method tagged
    ``"tracker_extrapolated"``).

    ``alpha`` weight on the new detection; ``1-alpha`` on the smoothed
    history. Higher alpha = faster reaction. Default 0.3 mirrors v0.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    out: list[Homography] = []
    last: Homography | None = None

    for det in detections:
        if det.confidence < min_confidence_for_update:
            if last is None:
                out.append(det)
                continue
            out.append(
                Homography(
                    H=last.H.copy(),
                    confidence=last.confidence * confidence_decay,
                    method="tracker_extrapolated",
                )
            )
            continue

        if last is None:
            new_H = det.H.copy()
        else:
            new_H = alpha * det.H + (1 - alpha) * last.H
        smoothed = Homography(
            H=new_H,
            confidence=max(det.confidence, (last.confidence if last else 0.0) * confidence_decay),
            method=det.method,
        )
        out.append(smoothed)
        last = smoothed

    return out


__all__ = ["smooth_homography_track"]
