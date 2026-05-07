"""Hand & fingertip tracking — Phase 4 deliverable.

Public entrypoint: :func:`track_hand`. Walks frames + per-frame
homographies in lockstep, calling a
:class:`tabvision.types.HandBackend` to produce a
:class:`FrameFingering` per frame. Frames whose homography has zero
confidence (no fretboard detected) emit a zero-logit fingering at the
correct timestamp so the output list aligns with the input frame
sequence.

For pipeline use, ``tabvision.pipeline.run_pipeline`` does its own
single-pass loop and does not call this orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from tabvision.types import (
    FrameFingering,
    GuitarConfig,
    HandBackend,
    Homography,
)
from tabvision.video.hand.neck_anchor import HandNeckAnchor, NeckAnchorConfig, compute_neck_anchor


def track_hand(
    frames: Iterable[tuple[float, np.ndarray]],
    homographies: list[Homography],
    backend: HandBackend,
    cfg: GuitarConfig,
) -> list[FrameFingering]:
    """Per-frame :class:`FrameFingering`s aligned with ``homographies``.

    Frames with zero-confidence homography are filled with a zero-logit
    fingering — fusion's :func:`playability.find_fingering_at` treats
    those as no-evidence and skips them.
    """
    out: list[FrameFingering] = []
    n_fingers = 4  # fretting fingers; matches Phase 4 convention
    for (t, frame), H in zip(frames, homographies, strict=False):  # noqa: N806 — H is the math-convention name for the homography
        if H.confidence <= 0.0:
            empty = np.zeros((n_fingers, cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
            out.append(FrameFingering(t=t, finger_pos_logits=empty, homography_confidence=0.0))
            continue
        ff = backend.detect(frame, H, cfg)
        # Backends produce a degenerate t=0.0; stamp the real timestamp here.
        out.append(replace(ff, t=t))
    return out


__all__ = [
    "HandNeckAnchor",
    "NeckAnchorConfig",
    "compute_neck_anchor",
    "track_hand",
]
