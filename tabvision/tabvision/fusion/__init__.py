"""Fusion — see SPEC.md §3.3, §8.

Public entrypoint: ``fuse(events, fingerings, cfg, session) -> list[TabEvent]``.

Combines audio events and per-frame hand posteriors into a decoded
(string, fret) sequence respecting playability constraints.
"""

from tabvision.fusion.neck_prior import (
    TimedNeckAnchor,
    anchor_position_prior,
    apply_neck_anchor_priors,
)
from tabvision.fusion.viterbi import fuse

__all__ = [
    "TimedNeckAnchor",
    "anchor_position_prior",
    "apply_neck_anchor_priors",
    "fuse",
]
