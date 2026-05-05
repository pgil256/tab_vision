"""Fretboard rectification — Phase 3 deliverable.

Public entrypoint: ``track_fretboard(frames, guitar_track, backend) -> list[Homography]``.

Strategy: geometric (Hough + RANSAC) primary, keypoint (YOLOv8-pose) fallback.
"""
