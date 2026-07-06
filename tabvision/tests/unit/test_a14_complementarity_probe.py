"""Tests for the A14 complementarity probe's fuse-mirroring decode.

The probe's analysis is only valid if ``decode_with_margins`` reproduces
``fuse(events, [], cfg)`` exactly and its local margins behave like a
best-vs-next-best trellis quantity (non-negative at the global optimum).
"""

from __future__ import annotations

import math

import pytest

from scripts.eval.a14_video_complementarity_probe import (
    NoteJoin,
    decode_with_margins,
    routing_accuracy,
)
from tabvision.fusion.viterbi import fuse
from tabvision.types import AudioEvent, GuitarConfig


def _ev(onset: float, pitch: int) -> AudioEvent:
    return AudioEvent(
        onset_s=onset, offset_s=onset + 0.2, pitch_midi=pitch, velocity=1.0, confidence=1.0
    )


def _sequence_with_chord() -> list[AudioEvent]:
    # Single line, then an E-minor-ish cluster (simultaneous onsets), then more line.
    return [
        _ev(0.0, 64),
        _ev(0.5, 67),
        _ev(1.0, 52),
        _ev(1.0, 59),
        _ev(1.0, 64),
        _ev(1.6, 62),
        _ev(2.2, 57),
    ]


def test_decode_matches_fuse() -> None:
    cfg = GuitarConfig()
    events = _sequence_with_chord()
    decoded = decode_with_margins(events, cfg)

    ours = sorted(
        (events[i].onset_s, events[i].pitch_midi, c.string_idx, c.fret)
        for i, (c, _m, _sz) in decoded.items()
    )
    ref = sorted((t.onset_s, t.pitch_midi, t.string_idx, t.fret) for t in fuse(events, [], cfg))
    assert ours == ref
    assert len(decoded) == len(events)


def test_margins_non_negative_at_optimum() -> None:
    """Global Viterbi optimality implies every local string-flip costs >= 0."""
    cfg = GuitarConfig()
    events = _sequence_with_chord()
    decoded = decode_with_margins(events, cfg)

    for _cand, margin, _size in decoded.values():
        assert margin >= -1e-9


def test_cluster_sizes_reflect_chord_membership() -> None:
    cfg = GuitarConfig()
    events = _sequence_with_chord()
    decoded = decode_with_margins(events, cfg)

    sizes = [decoded[i][2] for i in range(len(events))]
    assert sizes[0] == 1  # isolated opener
    assert sizes[2] == sizes[3] == sizes[4] == 3  # the simultaneous cluster
    assert sizes[6] == 1


def test_empty_events_decode_empty() -> None:
    assert decode_with_margins([], GuitarConfig()) == {}


def test_routing_accuracy_routes_below_threshold() -> None:
    notes = [
        # audio wrong / video right, low margin -> routing to video fixes it
        NoteJoin(
            clip="x",
            gold_string=2,
            audio_string=3,
            audio_margin=0.1,
            video_string=2,
            cluster_size=1,
        ),
        # audio right, high margin -> stays with audio
        NoteJoin(
            clip="x",
            gold_string=1,
            audio_string=1,
            audio_margin=5.0,
            video_string=4,
            cluster_size=1,
        ),
    ]
    assert routing_accuracy(notes, 0.0) == pytest.approx(0.5)  # nothing routed
    assert routing_accuracy(notes, 1.0) == pytest.approx(1.0)  # low-margin note routed
    assert routing_accuracy(notes, math.inf) == pytest.approx(0.5)  # all routed -> video everywhere
