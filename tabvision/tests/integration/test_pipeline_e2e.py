"""End-to-end integration test for ``tabvision.pipeline.run_pipeline``.

Skipped automatically when basic-pitch / ffmpeg / cv2 aren't installed.
The test uses ``data/fixtures/test_a440.mp4`` with the video stack
disabled — A440 is a pure tone with no guitar in frame, so ``video_enabled=True``
would just exercise the no-guitar-detected fallback (covered by unit
tests already). Audio-only here verifies the demux → audio backend →
fuse → TabEvent chain works on a real file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# basic-pitch is the audio backend exercised here; the highres backend
# would require ~1 GB of model downloads which we don't want in CI.
pytest.importorskip(
    "basic_pitch",
    reason="basic-pitch not installed — install with pip install '.[audio-baseline]'",
)
pytest.importorskip("soundfile")
pytest.importorskip("cv2", reason="opencv-python needed for demux frame iterator")

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not on PATH", allow_module_level=True)

FIXTURE = Path(__file__).parent.parent.parent / "data" / "fixtures" / "test_a440.mp4"

if not FIXTURE.exists():
    pytest.skip(f"fixture missing: {FIXTURE}", allow_module_level=True)


@pytest.mark.integration
def test_run_pipeline_audio_only_emits_a440_at_fret_5_high_e():
    """A440 (MIDI 69) audio-only → fret 5 on high E (string_idx=5).

    Exercises the pipeline.run_pipeline path with video disabled; the
    audio half goes through basic-pitch + the cluster Viterbi we shipped
    in Phase 5.
    """
    from tabvision.pipeline import run_pipeline

    events = run_pipeline(
        FIXTURE,
        audio_backend_name="basicpitch",
        video_enabled=False,
        lambda_vision=0.0,
    )

    assert events, "expected at least one TabEvent for A440"
    a440_events = [e for e in events if e.pitch_midi == 69]
    assert a440_events, f"no MIDI 69 (A4) detected; got: {[e.pitch_midi for e in events]}"
    sample = a440_events[0]
    assert sample.string_idx == 5, f"expected high E (idx 5), got string_idx={sample.string_idx}"
    assert sample.fret == 5, f"expected fret 5, got fret={sample.fret}"


@pytest.mark.integration
def test_run_pipeline_video_enabled_falls_back_gracefully_on_a440():
    """A440 fixture has no guitar in frame, so the video stack will either
    fail to import (no mediapipe / no YOLO checkpoint) or detect no guitar
    on every frame. Either way, the pipeline must complete and the result
    must match the audio-only output above.
    """
    from tabvision.pipeline import run_pipeline

    events = run_pipeline(
        FIXTURE,
        audio_backend_name="basicpitch",
        video_enabled=True,  # but no guitar present → pipeline copes
        lambda_vision=1.0,
    )

    assert events, "pipeline should produce events even with no guitar in frame"
    assert any(e.pitch_midi == 69 for e in events)
