"""Video module — see SPEC.md §3.3.

Subpackages:
- ``guitar`` — bbox detection + tracking (Phase 3).
- ``fretboard`` — homography per frame (Phase 3).
- ``hand`` — fingertip → string/fret posteriors (Phase 4).

Strict layering: this module imports types only, never logic from audio
or fusion (SPEC.md §3.3).
"""
