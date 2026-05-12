"""Tests for the melodic-segment position prior."""
from __future__ import annotations

from tabvision.fusion.melodic_prior import apply_melodic_segment_prior, find_melodic_segments
from tabvision.types import AudioEvent, GuitarConfig


def _ev(onset: float, pitch: int) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.2,
        pitch_midi=pitch,
        velocity=0.7,
        confidence=0.8,
    )


def test_find_melodic_segments_detects_stepwise_run():
    events = [_ev(0.0, 57), _ev(0.2, 59), _ev(0.4, 60), _ev(0.6, 62)]
    assert find_melodic_segments(events) == [[0, 1, 2, 3]]


def test_apply_melodic_segment_prior_attaches_candidate_prior():
    events = [_ev(0.0, 57), _ev(0.2, 59), _ev(0.4, 60), _ev(0.6, 62)]
    out = apply_melodic_segment_prior(events, GuitarConfig())

    assert all(event.fret_prior is not None for event in out)
    for event in out:
        assert event.fret_prior.shape == (6, 25)
        assert abs(float(event.fret_prior.sum()) - 1.0) < 1e-9


def test_long_scale_run_prior_prefers_compact_mid_neck_window():
    # E minor 3-note-per-string run from training-14. Low-fret equivalents
    # exist for most notes, but the compact playable shape is around frets 7-10.
    pitches = [47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 67, 69, 71, 72, 74]
    events = [_ev(index * 0.2, pitch) for index, pitch in enumerate(pitches)]

    out = apply_melodic_segment_prior(events, GuitarConfig())

    first_prior = out[0].fret_prior
    assert first_prior is not None
    string_idx, fret = divmod(int(first_prior.argmax()), first_prior.shape[1])
    assert (string_idx, fret) == (0, 7)

    tenth_prior = out[9].fret_prior
    assert tenth_prior is not None
    string_idx, fret = divmod(int(tenth_prior.argmax()), tenth_prior.shape[1])
    assert fret in {7, 9, 10}


def test_apply_melodic_segment_prior_leaves_short_fragments_unchanged():
    events = [_ev(0.0, 57), _ev(0.2, 59), _ev(0.4, 60)]
    out = apply_melodic_segment_prior(events, GuitarConfig())

    assert [event.fret_prior for event in out] == [None, None, None]
