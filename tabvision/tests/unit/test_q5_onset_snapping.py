"""Tests for the Q5 onset-snapping prototype.

The baseline variant must be a bit-exact identity — otherwise the paired
deltas measure the harness rather than snapping — and the strum variant must
group on the *original* onsets, since regrouping on snapped times would let
snapping silently redraw the 80 ms cluster boundaries it is supposed to
respect.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval.q5_onset_snapping import (
    VARIANTS,
    apply_snapping,
    onset_envelope,
    snap_onset,
)
from tabvision.types import AudioEvent

BY_NAME = {variant.name: variant for variant in VARIANTS}


def _event(onset: float, pitch: int = 64) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.3,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=0.9,
    )


def _ramp_envelope() -> tuple[np.ndarray, np.ndarray]:
    """Flux with a single sharp peak at t = 1.000 s."""
    times = np.arange(0.0, 2.0, 0.005)
    strength = np.zeros_like(times)
    strength[np.argmin(np.abs(times - 1.0))] = 10.0
    return strength, times


def test_baseline_is_a_bit_exact_identity() -> None:
    events = [_event(0.5), _event(1.02)]
    strength, times = _ramp_envelope()
    moved, shifts = apply_snapping(events, strength, times, BY_NAME["baseline"])
    assert moved == events
    assert shifts == [0.0, 0.0]


def test_snap_pulls_an_onset_onto_the_peak() -> None:
    strength, times = _ramp_envelope()
    assert snap_onset(1.02, strength, times, 0.030) == 1.0
    # Outside the window the peak is invisible and the onset must not move.
    assert snap_onset(1.20, strength, times, 0.030) == 1.20


def test_snap_preserves_duration() -> None:
    strength, times = _ramp_envelope()
    moved, _shifts = apply_snapping([_event(1.02)], strength, times, BY_NAME["snap-30ms"])
    assert moved[0].onset_s == 1.0
    assert moved[0].offset_s - moved[0].onset_s == pytest.approx(0.3)


def test_flat_envelope_leaves_onsets_alone() -> None:
    times = np.arange(0.0, 2.0, 0.005)
    strength = np.zeros_like(times)
    assert snap_onset(1.02, strength, times, 0.050) == 1.02


def test_strum_variant_collapses_a_cluster_to_one_onset() -> None:
    # Three notes inside the 80 ms cluster window, one isolated note later.
    events = [_event(1.00, 52), _event(1.03, 59), _event(1.06, 64), _event(5.00, 67)]
    times = np.arange(0.0, 6.0, 0.005)
    strength = np.zeros_like(times)
    for peak in (0.99, 1.04, 1.07):
        strength[np.argmin(np.abs(times - peak))] = 10.0
    moved, _shifts = apply_snapping(events, strength, times, BY_NAME["strum-30ms"])
    cluster = sorted({round(event.onset_s, 6) for event in moved if event.onset_s < 3.0})
    assert len(cluster) == 1  # one shared strum onset, not three
    assert any(abs(event.onset_s - 5.00) < 1e-9 for event in moved)  # isolated note untouched


def test_plain_snap_does_not_collapse_a_cluster() -> None:
    events = [_event(1.00, 52), _event(1.03, 59), _event(1.06, 64)]
    times = np.arange(0.0, 3.0, 0.005)
    strength = np.zeros_like(times)
    for peak in (0.99, 1.04, 1.07):
        strength[np.argmin(np.abs(times - peak))] = 10.0
    moved, _shifts = apply_snapping(events, strength, times, BY_NAME["snap-30ms"])
    # Notes may share a peak when one dominates their overlapping windows;
    # what matters is that plain snapping never *forces* a single shared
    # onset the way the strum variant deliberately does.
    assert len({round(event.onset_s, 6) for event in moved}) > 1


def test_onset_envelope_peaks_at_a_transient() -> None:
    sr = 22050
    wav = np.zeros(sr, dtype=np.float32)
    wav[sr // 2 :] = np.sin(
        2 * np.pi * 440 * np.arange(sr - sr // 2, dtype=np.float32) / sr
    ).astype(np.float32)
    strength, times = onset_envelope(wav, sr)
    assert abs(times[int(np.argmax(strength))] - 0.5) < 0.05
