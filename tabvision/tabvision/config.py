"""Configuration loading — see SPEC.md §5.5.

Two layers of configuration:

- ``GuitarConfig`` — physical instrument properties (tuning, capo, max fret).
- ``SessionConfig`` — per-recording context: instrument type, tone, playing
  style. Affects audio backend variant and fusion mode.

Implementation deferred to Phase 1+ — this module is a placeholder during
Phase 0 scaffolding.
"""

from tabvision.types import GuitarConfig, SessionConfig

__all__ = ["GuitarConfig", "SessionConfig"]
