"""Fusion — see SPEC.md §3.3, §8.

Public entrypoint: ``fuse(events, fingerings, cfg, session) -> list[TabEvent]``.

Combines audio events and per-frame hand posteriors into a decoded
(string, fret) sequence respecting playability constraints.
"""

from tabvision.fusion.viterbi import fuse

__all__ = ["fuse"]
