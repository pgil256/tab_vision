"""Regression tests for demux metadata probing (ffprobe parsing).

Browser ``MediaRecorder`` WebM is a live/streaming container with no duration
in its header, so ffprobe reports ``N/A``. The old parser treated any line
containing ``/`` as the frame-rate fraction and crashed on ``float("A")`` when
splitting ``"N/A"``. These tests stub ffprobe, so they need neither ffmpeg nor
opencv on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tabvision import demux as demux_mod
from tabvision.demux import _probe_metadata

DUMMY = Path("clip.webm")


def _stub_ffprobe(monkeypatch, stdout: str) -> None:
    def fake_run(cmd, capture_output=True, text=True, check=False):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(demux_mod.subprocess, "run", fake_run)


def test_probe_na_only_audio_recording(monkeypatch):
    """Audio-only MediaRecorder WebM: ffprobe emits a single ``N/A``."""
    _stub_ffprobe(monkeypatch, "N/A\n")
    duration_s, fps = _probe_metadata(DUMMY)
    assert duration_s == 0.0
    assert fps == 0.0  # no video stream → no usable frame rate


def test_probe_normal_video(monkeypatch):
    """Well-formed source: r_frame_rate fraction + numeric durations."""
    _stub_ffprobe(monkeypatch, "30/1\n8.0\n8.0\n")
    duration_s, fps = _probe_metadata(DUMMY)
    assert fps == 30.0
    assert duration_s == 8.0


def test_probe_zero_frame_rate_no_div_by_zero(monkeypatch):
    """A ``0/0`` r_frame_rate must not raise ZeroDivisionError."""
    _stub_ffprobe(monkeypatch, "0/0\nN/A\n")
    duration_s, fps = _probe_metadata(DUMMY)
    assert fps == 0.0
    assert duration_s == 0.0


def test_probe_video_webm_na_duration(monkeypatch):
    """Video WebM from the browser: valid fps but ``N/A`` duration line."""
    _stub_ffprobe(monkeypatch, "25/1\nN/A\n")
    duration_s, fps = _probe_metadata(DUMMY)
    assert fps == 25.0
    assert duration_s == 0.0  # recovered from audio length in demux()
