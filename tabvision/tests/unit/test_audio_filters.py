"""Unit tests for tabvision.audio.filters."""

from tabvision.audio.filters import (
    AudioFilterConfig,
    apply_default_filters,
    filter_harmonics,
    filter_low_quality,
    filter_sustain_redetections,
    merge_consecutive,
)
from tabvision.types import AudioEvent


def _ev(
    midi: int,
    onset: float,
    *,
    offset: float | None = None,
    velocity: float = 0.5,
    confidence: float = 0.7,
) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=offset if offset is not None else onset + 0.25,
        pitch_midi=midi,
        velocity=velocity,
        confidence=confidence,
    )


# ---------- filter_low_quality ----------


def test_filter_low_quality_drops_below_confidence():
    cfg = AudioFilterConfig(min_confidence=0.5)
    events = [_ev(60, 0.0, confidence=0.4), _ev(60, 1.0, confidence=0.6)]
    out = filter_low_quality(events, cfg)
    assert len(out) == 1
    assert out[0].onset_s == 1.0


def test_filter_low_quality_drops_short_notes():
    cfg = AudioFilterConfig(min_duration_s=0.05)
    events = [_ev(60, 0.0, offset=0.02), _ev(60, 1.0, offset=1.10)]
    out = filter_low_quality(events, cfg)
    assert len(out) == 1
    assert out[0].onset_s == 1.0


def test_filter_low_quality_drops_quiet_notes():
    cfg = AudioFilterConfig(min_amplitude=0.2)
    events = [_ev(60, 0.0, velocity=0.1), _ev(60, 1.0, velocity=0.5)]
    out = filter_low_quality(events, cfg)
    assert len(out) == 1
    assert out[0].onset_s == 1.0


# ---------- merge_consecutive ----------


def test_merge_consecutive_same_pitch_overlap():
    cfg = AudioFilterConfig(merge_gap_s=0.02)
    events = [
        _ev(60, 0.0, offset=0.30),
        _ev(60, 0.28, offset=0.50),  # overlaps -> merge
    ]
    out = merge_consecutive(events, cfg)
    assert len(out) == 1
    assert out[0].onset_s == 0.0
    assert out[0].offset_s == 0.50


def test_merge_consecutive_keeps_distinct_pitches():
    cfg = AudioFilterConfig()
    events = [_ev(60, 0.0), _ev(64, 0.05)]
    out = merge_consecutive(events, cfg)
    assert len(out) == 2


def test_merge_consecutive_keeps_far_same_pitch():
    cfg = AudioFilterConfig(merge_gap_s=0.02)
    events = [
        _ev(60, 0.0, offset=0.30),
        _ev(60, 1.00, offset=1.20),  # gap 0.7s -> keep both
    ]
    out = merge_consecutive(events, cfg)
    assert len(out) == 2


# ---------- filter_sustain_redetections ----------


def test_filter_sustain_drops_quieter_redetection():
    cfg = AudioFilterConfig(sustain_amplitude_ratio=0.95)
    # Note 1: loud, long-sustaining. Note 2: same pitch, quieter, while
    # note 1 still rings -> drop.
    events = [
        _ev(60, 0.0, offset=1.0, velocity=0.8),
        _ev(60, 0.5, offset=1.0, velocity=0.3),  # sustain redetection
    ]
    out = filter_sustain_redetections(events, cfg)
    assert len(out) == 1
    assert out[0].velocity == 0.8


def test_filter_sustain_keeps_loud_replay():
    cfg = AudioFilterConfig(sustain_amplitude_ratio=0.95)
    events = [
        _ev(60, 0.0, offset=1.0, velocity=0.8),
        _ev(60, 0.5, offset=1.5, velocity=0.85),  # louder -> not a redetection
    ]
    out = filter_sustain_redetections(events, cfg)
    assert len(out) == 2


def test_filter_sustain_ignores_distant_events():
    cfg = AudioFilterConfig(sustain_window_s=0.6)
    events = [
        _ev(60, 0.0, offset=0.20, velocity=0.8),
        _ev(60, 1.00, offset=1.20, velocity=0.3),  # outside window
    ]
    out = filter_sustain_redetections(events, cfg)
    assert len(out) == 2


# ---------- filter_harmonics ----------


def test_filter_harmonics_drops_quiet_octave_above():
    cfg = AudioFilterConfig()
    events = [
        _ev(60, 0.0, velocity=0.6),
        _ev(72, 0.0, velocity=0.15),  # octave above, quieter than 0.35× -> drop
    ]
    out = filter_harmonics(events, cfg)
    assert len(out) == 1
    assert out[0].pitch_midi == 60


def test_filter_harmonics_keeps_loud_octave_above():
    cfg = AudioFilterConfig()
    events = [
        _ev(60, 0.0, velocity=0.40),
        _ev(72, 0.0, velocity=0.39),  # ratio ~0.97 > 0.35 -> keep
    ]
    out = filter_harmonics(events, cfg)
    assert len(out) == 2


def test_filter_harmonics_protects_loud_notes():
    cfg = AudioFilterConfig(harmonic_protect_amplitude=0.45)
    events = [
        _ev(60, 0.0, velocity=0.9),
        _ev(72, 0.0, velocity=0.5),  # would be flagged but protected
    ]
    out = filter_harmonics(events, cfg)
    assert len(out) == 2


def test_filter_harmonics_drops_subharmonic():
    cfg = AudioFilterConfig()
    events = [
        _ev(72, 0.0, velocity=0.6),
        _ev(60, 0.0, velocity=0.10),  # octave below + quiet -> sub-harmonic, drop
    ]
    out = filter_harmonics(events, cfg)
    assert len(out) == 1
    assert out[0].pitch_midi == 72


def test_filter_harmonics_ignores_non_harmonic_intervals():
    cfg = AudioFilterConfig()
    events = [
        _ev(60, 0.0, velocity=0.6),
        _ev(63, 0.0, velocity=0.1),  # 3 semitones — not a harmonic
    ]
    out = filter_harmonics(events, cfg)
    assert len(out) == 2


# ---------- apply_default_filters end-to-end ----------


def test_default_pipeline_noop_on_clean_input():
    events = [_ev(60, 0.0, velocity=0.5), _ev(64, 1.0, velocity=0.5)]
    out = apply_default_filters(events)
    assert len(out) == 2


def test_default_pipeline_runs_all_stages():
    cfg = AudioFilterConfig()
    events = [
        _ev(60, 0.0, offset=1.0, velocity=0.6),  # fundamental
        _ev(72, 0.0, velocity=0.10),              # harmonic, drops
        _ev(60, 0.5, velocity=0.05),              # sustain redetection, drops
        _ev(60, 0.001, offset=0.003, velocity=0.5),  # too short, drops in low-quality
    ]
    out = apply_default_filters(events, cfg)
    assert len(out) == 1
    assert out[0].pitch_midi == 60
    assert out[0].onset_s == 0.0
