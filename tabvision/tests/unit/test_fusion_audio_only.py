"""Unit tests for the audio-only fusion path."""

from tabvision.fusion import fuse
from tabvision.types import AudioEvent, GuitarConfig


def _ev(midi: int, t: float) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=0.8,
    )


def test_empty_input_yields_empty_output():
    assert fuse([], [], GuitarConfig()) == []


def test_single_event_picks_lowest_fret():
    """A4 (MIDI 69) without prior context should land on fret 5 / high E."""
    out = fuse([_ev(69, 0.0)], [], GuitarConfig())
    assert len(out) == 1
    assert out[0].fret == 5
    assert out[0].string_idx == 5  # high E
    assert out[0].pitch_midi == 69


def test_continuity_bias_keeps_us_on_string():
    """If we just played A4 on high E, then play B4 (MIDI 71), continuity
    bonus should keep us on high E (fret 7) rather than moving to B (fret 12).
    """
    events = [_ev(69, 0.0), _ev(71, 0.5)]
    out = fuse(events, [], GuitarConfig())
    assert out[0].string_idx == 5 and out[0].fret == 5
    assert out[1].string_idx == 5 and out[1].fret == 7


def test_out_of_range_event_is_skipped():
    """A pitch with no candidates is dropped, not fabricated."""
    events = [_ev(20, 0.0), _ev(69, 0.5)]
    out = fuse(events, [], GuitarConfig())
    assert len(out) == 1
    assert out[0].pitch_midi == 69


def test_capo_shifts_picks():
    cfg = GuitarConfig(capo=2)
    out = fuse([_ev(69, 0.0)], [], cfg)
    assert len(out) == 1
    assert out[0].fret >= 2
