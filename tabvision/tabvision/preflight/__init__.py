"""Preflight tool — see SPEC.md §3.3, §7 Phase 3.

Public entrypoint: ``check(video_path) -> PreflightReport``.

Validates camera framing on a clip and emits actionable feedback.
"""

from tabvision.preflight.check import check
from tabvision.preflight.feedback import render

__all__ = ["check", "render"]
