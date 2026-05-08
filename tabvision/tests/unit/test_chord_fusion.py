"""Unit tests for chord-aware fusion (``tabvision.fusion.chord`` plus
the cluster-level Viterbi in :mod:`tabvision.fusion.viterbi`).

Covers:
- ``cluster_events``: clustering by onset gap.
- ``enumerate_chord_states``: per-string monophony + hand-span pruning.
- ``chord_anchor``: lowest-fret pressed note as anchor.
- End-to-end ``fuse``: simultaneous events emit distinct strings, picks
  fall within the hand-span constraint, and vision evidence on one
  chord member pulls the whole shape onto a vision-supported voicing.
"""

from __future__ import annotations

import numpy as np

from tabvision.fusion import fuse
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.chord import (
    CHORD_MAX_GAP_S,
    chord_anchor,
    cluster_events,
    enumerate_chord_states,
)
from tabvision.fusion.playability import MAX_HAND_SPAN
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig


def _ev(midi: int, t: float, confidence: float = 0.8) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=confidence,
    )


def _peaked_fingering(t: float, string_idx: int, fret: int) -> FrameFingering:
    logits = np.zeros((4, 6, 25), dtype=np.float64)
    logits[0, string_idx, fret] = 10.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


# ---------- cluster_events ----------


def test_cluster_events_single_event_yields_one_cluster():
    clusters = cluster_events([_ev(60, 0.0)])
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_cluster_events_close_events_join_one_cluster():
    """Two events 50 ms apart should be one chord cluster."""
    events = [_ev(60, 0.0), _ev(64, 0.05)]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_events_far_events_split():
    """Two events 200 ms apart should be two clusters."""
    events = [_ev(60, 0.0), _ev(64, 0.20)]
    clusters = cluster_events(events)
    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_cluster_events_chain_through_threshold():
    """Three events at 0, 80, 160 ms (each adjacent gap == threshold)
    should form one cluster (chain semantics)."""
    events = [
        _ev(60, 0.0),
        _ev(64, CHORD_MAX_GAP_S),
        _ev(67, 2 * CHORD_MAX_GAP_S),
    ]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_events_unsorted_input_is_sorted():
    """Out-of-order input should still produce a chronologically grouped
    output."""
    events = [_ev(67, 0.05), _ev(60, 0.0)]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert clusters[0][0].pitch_midi == 60  # low-onset first


# ---------- enumerate_chord_states ----------


def test_enumerate_chord_states_enforces_monophony():
    """C major triad (C4 + E4 + G4) — no enumerated state may put two
    notes on the same string."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0), _ev(67, 0.0)]
    states = enumerate_chord_states(events, cfg)
    assert states  # non-empty
    for state in states:
        strings = [c.string_idx for c in state]
        assert len(strings) == len(set(strings)), f"per-string monophony violated: {state}"


def test_enumerate_chord_states_enforces_hand_span():
    """Every enumerated state must respect MAX_HAND_SPAN over pressed frets."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0), _ev(67, 0.0)]
    states = enumerate_chord_states(events, cfg)
    for state in states:
        pressed = [c.fret for c in state if c.fret > 0]
        if pressed:
            assert max(pressed) - min(pressed) <= MAX_HAND_SPAN


def test_enumerate_chord_states_empty_when_event_unfretable():
    """If any event has no candidates, no chord state survives."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(20, 0.0)]  # 20 = far below low E
    assert enumerate_chord_states(events, cfg) == []


# ---------- chord_anchor ----------


def test_chord_anchor_picks_lowest_pressed_fret():
    state = (
        Candidate(string_idx=4, fret=5),
        Candidate(string_idx=5, fret=0),  # open
        Candidate(string_idx=3, fret=3),
    )
    assert chord_anchor(state) == Candidate(string_idx=3, fret=3)


def test_chord_anchor_falls_back_to_first_when_all_open():
    state = (
        Candidate(string_idx=5, fret=0),
        Candidate(string_idx=4, fret=0),
    )
    assert chord_anchor(state) == state[0]


# ---------- end-to-end fuse() through chord clusters ----------


def test_fuse_simultaneous_events_emit_distinct_strings():
    """C4 + E4 fired together — picks must use different strings."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0)]
    out = fuse(events, [], cfg)
    assert len(out) == 2
    assert out[0].string_idx != out[1].string_idx


def test_fuse_three_note_chord_within_hand_span():
    """C major triad (C4 + E4 + G4) — picks form a hand-span-feasible voicing."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0), _ev(67, 0.0)]
    out = fuse(events, [], cfg)
    assert len(out) == 3
    strings = [t.string_idx for t in out]
    assert len(set(strings)) == 3  # all distinct
    pressed = [t.fret for t in out if t.fret > 0]
    if pressed:
        assert max(pressed) - min(pressed) <= MAX_HAND_SPAN


def test_fuse_chord_prefers_open_string_voicing_with_uniform_vision():
    """C major triad — the open-E voicing should win on emission cost
    when no vision evidence pushes elsewhere.

    E4 has an open-string candidate (5, 0). The open-string bonus +
    low-fret bias should make at least one note an open string in the
    chosen voicing."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0), _ev(67, 0.0)]
    out = fuse(events, [], cfg)
    assert any(t.fret == 0 for t in out)


def test_fuse_chord_vision_pulls_voicing():
    """If the fingering is peaked at a non-default position for one of
    the chord notes, the chosen state should include that exact pick."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(64, 0.0), _ev(67, 0.0)]
    # Push C4 onto string 3 fret 5 (G-string). The default voicing
    # would have C4 on string 4 fret 1. With this peak, C4 should move.
    fings = [_peaked_fingering(0.0, string_idx=3, fret=5)]
    out = fuse(events, fings, cfg, lambda_vision=2.0)

    c4 = next(t for t in out if t.pitch_midi == 60)
    assert (c4.string_idx, c4.fret) == (3, 5)
    # Other notes still produce a valid voicing.
    strings = [t.string_idx for t in out]
    assert len(set(strings)) == 3
