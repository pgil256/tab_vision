"""Tests for advisory beat-grid detection (tabvision.audio.beats)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tabvision.audio.beats import (
    MAX_BPM,
    MIN_BEATS,
    MIN_BPM,
    MIN_CLIP_SECONDS,
    detect_beat_grid,
)
from tabvision.types import BeatGrid

SR = 22050


def _click_track(bpm: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Synthetic click track: short 1 kHz bursts on every beat."""
    wav = np.zeros(int(sr * duration_s), dtype=np.float32)
    burst_len = 220
    burst = (np.hanning(burst_len) * np.sin(2 * np.pi * 1000 * np.arange(burst_len) / sr)).astype(
        np.float32
    )
    beat_interval = 60.0 / bpm
    t = 0.0
    while t < duration_s:
        idx = int(t * sr)
        end = min(idx + burst_len, wav.size)
        wav[idx:end] += burst[: end - idx]
        t += beat_interval
    return wav


def test_click_track_detects_tempo_within_tolerance():
    pytest.importorskip("librosa")
    grid = detect_beat_grid(_click_track(120.0, 20.0), SR)
    assert grid is not None
    assert isinstance(grid, BeatGrid)
    assert abs(grid.tempo_bpm - 120.0) <= 5.0
    assert len(grid.beat_times) >= MIN_BEATS
    assert grid.beats_per_bar == 4
    assert grid.source == "librosa-beat-track"
    # Tracked beats must be sorted, in-range times.
    times = list(grid.beat_times)
    assert times == sorted(times)
    assert 0.0 <= times[0] and times[-1] <= 20.0


def test_short_clip_returns_none_before_importing_librosa():
    wav = np.zeros(int(SR * (MIN_CLIP_SECONDS - 1.0)), dtype=np.float32)
    assert detect_beat_grid(wav, SR) is None


def test_silence_returns_none():
    pytest.importorskip("librosa")
    assert detect_beat_grid(np.zeros(int(SR * 10.0), dtype=np.float32), SR) is None


def test_missing_librosa_fails_open(monkeypatch):
    # A None entry in sys.modules makes ``import librosa`` raise ImportError.
    monkeypatch.setitem(sys.modules, "librosa", None)
    assert detect_beat_grid(_click_track(120.0, 20.0), SR) is None


def test_bpm_gates_are_sane_constants():
    assert MIN_BPM < 120.0 < MAX_BPM
