"""Unit tests for ``tabvision.fusion.viterbi.fuse``.

Covers both the audio-only path (no / uniform fingerings) and the
video-aware Viterbi behaviour (vision evidence pulls picks; lookahead
changes earlier picks when later events benefit from a different anchor).
"""

import numpy as np

from tabvision.fusion import fuse
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig


def _ev(midi: int, t: float) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=0.8,
    )


def _peaked_fingering(t: float, string_idx: int, fret: int) -> FrameFingering:
    """Marginal sharply peaked at ``(string_idx, fret)``."""
    logits = np.zeros((4, 6, 25), dtype=np.float64)
    logits[0, string_idx, fret] = 10.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


def _uniform_fingering(t: float) -> FrameFingering:
    """Marginal ≈ uniform across (string, fret) cells."""
    logits = np.ones((4, 6, 25), dtype=np.float64)
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


# ---------- audio-only regression ----------


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


# ---------- video-aware Viterbi ----------


def test_uniform_vision_matches_no_vision():
    """A uniform fingering must not change the audio-only picks."""
    events = [_ev(69, 0.0), _ev(71, 0.5)]
    cfg = GuitarConfig()
    fings = [_uniform_fingering(0.0), _uniform_fingering(0.5)]
    out_with = fuse(events, fings, cfg)
    out_without = fuse(events, [], cfg)
    assert [(e.string_idx, e.fret) for e in out_with] == [
        (e.string_idx, e.fret) for e in out_without
    ]


def test_decisive_vision_moves_single_pick():
    """A vision peak at a non-default candidate should override the lowest-fret bias.

    A4's audio-only pick is (5, 5). With the fingering peaked at the G-string
    A4 position (3, 14), Viterbi should land there instead.
    """
    cfg = GuitarConfig()
    events = [_ev(69, 0.0)]
    fings = [_peaked_fingering(0.0, string_idx=3, fret=14)]
    out = fuse(events, fings, cfg, lambda_vision=1.0)
    assert len(out) == 1
    assert out[0].string_idx == 3
    assert out[0].fret == 14


def test_lambda_zero_disables_vision():
    """Setting ``lambda_vision=0`` should reproduce the audio-only pick even
    when a peaked fingering is present."""
    cfg = GuitarConfig()
    events = [_ev(69, 0.0)]
    fings = [_peaked_fingering(0.0, string_idx=3, fret=14)]
    out = fuse(events, fings, cfg, lambda_vision=0.0)
    assert len(out) == 1
    assert out[0].string_idx == 5  # back to high E
    assert out[0].fret == 5


def test_viterbi_lookahead_changes_earlier_pick():
    """A future event's vision evidence should pull the earlier pick onto
    the same string when staying lowest-fret would force a giant hand jump.

    Sequence: A4 (MIDI 69) → B4 (MIDI 71). The B4 fingering is peaked at
    (string=3, fret=16) — the G-string B4 position. A greedy decoder picks
    (5, 5) for A4 (lowest fret) and would then have to leap from fret 5 →
    fret 16 across two strings; the hand-span barrier makes that path
    expensive. Viterbi instead picks (3, 14) for A4 — same string, two
    frets below the upcoming B4 — so the entire path is cheap.
    """
    cfg = GuitarConfig()
    events = [_ev(69, 0.0), _ev(71, 0.5)]
    fings = [_peaked_fingering(0.5, string_idx=3, fret=16)]
    out = fuse(events, fings, cfg, lambda_vision=1.0)
    assert len(out) == 2
    # Vision-decisive on the second event:
    assert (out[1].string_idx, out[1].fret) == (3, 16)
    # Lookahead-driven on the first event: must NOT be the audio-only (5, 5);
    # specifically should land on the G-string A4 anchor.
    assert (out[0].string_idx, out[0].fret) == (3, 14)
