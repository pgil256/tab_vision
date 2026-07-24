"""Focused tests for joining decoded video frames to the audio time axis."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tabvision import demux as demux_mod
from tabvision.demux import _extract_audio, _probe_frame_clock, _resolve_frame_timestamp

DUMMY = Path("offset-vfr.mp4")


def _stub_clock_probe(
    monkeypatch,
    *,
    audio_start_s: str | None,
    video_start_s: str | None,
    frame_pts: list[str | None],
) -> None:
    streams: list[dict[str, str]] = []
    if video_start_s is not None:
        streams.append({"codec_type": "video", "start_time": video_start_s})
    if audio_start_s is not None:
        streams.append({"codec_type": "audio", "start_time": audio_start_s})
    frames = [{} if pts is None else {"best_effort_timestamp_time": pts} for pts in frame_pts]

    def fake_run(cmd, capture_output=True, text=True, check=False):
        del capture_output, text, check
        show_entries = cmd[cmd.index("-show_entries") + 1]
        if show_entries == "stream=codec_type,start_time":
            stdout = json.dumps({"streams": streams})
        elif show_entries == "frame=best_effort_timestamp_time":
            stdout = json.dumps({"frames": frames})
        else:  # pragma: no cover - protects the test stub from silent drift
            raise AssertionError(f"unexpected ffprobe query: {show_entries}")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(demux_mod.subprocess, "run", fake_run)


def _resolve_all(
    timestamps_s: tuple[float | None, ...],
    fallback_start_s: float,
    *,
    fps: float,
) -> list[float]:
    resolved: list[float] = []
    previous_s: float | None = None
    for frame_idx in range(len(timestamps_s)):
        previous_s = _resolve_frame_timestamp(
            frame_idx,
            timestamps_s=timestamps_s,
            fallback_start_s=fallback_start_s,
            fps=fps,
            previous_s=previous_s,
        )
        resolved.append(previous_s)
    return resolved


def test_vfr_pts_are_normalized_to_nonzero_audio_start(monkeypatch):
    """Irregular video PTS survive; container time is removed using audio start."""
    _stub_clock_probe(
        monkeypatch,
        audio_start_s="100.250",
        video_start_s="100.500",
        frame_pts=["100.500", "100.540", "100.610"],
    )

    timestamps_s, fallback_start_s = _probe_frame_clock(DUMMY)

    assert timestamps_s == pytest.approx((0.250, 0.290, 0.360))
    assert fallback_start_s == pytest.approx(0.250)
    assert _resolve_all(timestamps_s, fallback_start_s, fps=25.0) == pytest.approx(
        (0.250, 0.290, 0.360)
    )


def test_video_before_audio_keeps_negative_media_timestamp(monkeypatch):
    """A frame before decoded audio sample zero must remain before zero."""
    _stub_clock_probe(
        monkeypatch,
        audio_start_s="12.000",
        video_start_s="11.800",
        frame_pts=["11.800", "11.840"],
    )

    timestamps_s, fallback_start_s = _probe_frame_clock(DUMMY)

    assert timestamps_s == pytest.approx((-0.200, -0.160))
    assert fallback_start_s == pytest.approx(-0.200)


def test_missing_pts_interpolates_without_shifting_later_frames(monkeypatch):
    """One absent PTS is interpolated while the next frame keeps its own PTS."""
    _stub_clock_probe(
        monkeypatch,
        audio_start_s="5.000",
        video_start_s="5.200",
        frame_pts=["5.200", None, "5.310"],
    )

    timestamps_s, fallback_start_s = _probe_frame_clock(DUMMY)

    assert timestamps_s[0] == pytest.approx(0.200)
    assert timestamps_s[1] is None
    assert timestamps_s[2] == pytest.approx(0.310)
    assert _resolve_all(timestamps_s, fallback_start_s, fps=25.0) == pytest.approx(
        (0.200, 0.255, 0.310)
    )


def test_unavailable_timeline_has_deterministic_historical_cfr_fallback(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, check=False):
        del capture_output, text, check
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unsupported")

    monkeypatch.setattr(demux_mod.subprocess, "run", fake_run)

    timestamps_s, fallback_start_s = _probe_frame_clock(DUMMY)

    assert timestamps_s == ()
    assert fallback_start_s == 0.0
    assert [
        _resolve_frame_timestamp(
            frame_idx,
            timestamps_s=timestamps_s,
            fallback_start_s=fallback_start_s,
            fps=25.0,
            previous_s=None if frame_idx == 0 else (frame_idx - 1) / 25.0,
        )
        for frame_idx in range(3)
    ] == pytest.approx((0.0, 0.04, 0.08))


def test_missing_pts_interpolates_without_outrunning_later_vfr_pts() -> None:
    timestamps_s = (0.0, 0.047, None, 0.106, 0.169)

    resolved = _resolve_all(timestamps_s, fallback_start_s=0.0, fps=16.58)

    assert resolved[0] == pytest.approx(0.0)
    assert resolved[1] == pytest.approx(0.047)
    assert 0.047 < resolved[2] < 0.106
    assert resolved[3:] == pytest.approx((0.106, 0.169))


def test_leading_missing_pts_are_derived_backwards_from_first_valid_pts() -> None:
    timestamps_s = (None, None, 0.100, 0.160)

    resolved = _resolve_all(timestamps_s, fallback_start_s=0.100, fps=25.0)

    assert resolved == pytest.approx((0.020, 0.060, 0.100, 0.160))


def test_audio_extraction_and_clock_use_the_same_first_audio_stream(monkeypatch) -> None:
    captured: list[str] = []

    def fake_run(cmd, capture_output=True, check=False):
        del capture_output, check
        captured.extend(cmd)
        samples = np.asarray([0.0], dtype=np.float32).tobytes()
        return subprocess.CompletedProcess(cmd, 0, stdout=samples, stderr=b"")

    monkeypatch.setattr(demux_mod.subprocess, "run", fake_run)

    audio = _extract_audio(DUMMY, 22_050)

    map_index = captured.index("-map")
    assert captured[map_index + 1] == "0:a:0"
    assert audio.tolist() == [0.0]
