"""Unit tests for the D1-b technique-baseline proxy derivation.

Pure-logic tests (no GuitarSet data, no models): they pin the bend/slide proxy
definitions so the reported baseline stays reproducible and the vibrato/attack
rejection does not silently regress into over-counting again.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.eval.d1b_technique_baseline import (
    BEND_SEMITONES_PRIMARY,
    _count_slides,
    _hz_to_midi,
    _is_bend,
    _Note,
    _StringContour,
    _wilson_halfwidth,
    census_track,
)

HOP = 0.005


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def _contour(pairs: list[tuple[float, float]]) -> _StringContour:
    """Build a contour from (time, midi) samples."""
    c = _StringContour()
    for t, m in pairs:
        c.times.append(t)
        c.midis.append(m)
    return c


def _ramp_samples(onset: float, dur: float, midi_fn) -> list[tuple[float, float]]:
    out = []
    t = onset
    while t <= onset + dur + 1e-9:
        out.append((round(t, 6), midi_fn(t - onset)))
        t += HOP
    return out


def test_hz_to_midi_reference_pitches():
    assert _hz_to_midi(440.0) == 69.0
    assert round(_hz_to_midi(880.0), 6) == 81.0
    assert round(_hz_to_midi(220.0), 6) == 57.0


def test_sustained_shift_is_a_bend():
    # midi holds at 60 for the first half, then holds at 61.6 — a clear held bend.
    note = _Note(string_idx=0, onset_s=0.0, duration_s=0.5, pitch_midi=60)
    contour = _contour(_ramp_samples(0.0, 0.5, lambda dt: 60.0 if dt < 0.25 else 61.6))
    assert _is_bend(note, contour, BEND_SEMITONES_PRIMARY) is True


def test_vibrato_is_not_a_bend():
    # Oscillates +/- 0.6 st around 60 with zero net shift — must be rejected.
    note = _Note(string_idx=0, onset_s=0.0, duration_s=0.6, pitch_midi=60)
    contour = _contour(
        _ramp_samples(0.0, 0.6, lambda dt: 60.0 + 0.6 * math.sin(2 * math.pi * 7.0 * dt))
    )
    assert _is_bend(note, contour, BEND_SEMITONES_PRIMARY) is False


def test_flat_and_too_short_notes_are_not_bends():
    flat = _Note(0, 0.0, 0.5, 60)
    flat_c = _contour(_ramp_samples(0.0, 0.5, lambda dt: 60.0))
    assert _is_bend(flat, flat_c, BEND_SEMITONES_PRIMARY) is False

    short = _Note(0, 0.0, 0.05, 60)  # below MIN_BEND_NOTE_DUR_S
    short_c = _contour(_ramp_samples(0.0, 0.05, lambda dt: 60.0 + 40 * dt))
    assert _is_bend(short, short_c, BEND_SEMITONES_PRIMARY) is False


def test_legato_glide_is_a_slide_but_gapped_notes_are_not():
    a = _Note(0, 0.0, 0.2, 60)
    b = _Note(0, 0.22, 0.2, 63)  # +3 st, 0.02 s gap == legato
    bridge = _contour(_ramp_samples(0.0, 0.42, lambda dt: 60.0 + 3.0 * (dt / 0.42)))
    assert _count_slides([a, b], bridge) == 1

    far = _Note(0, 1.0, 0.2, 63)  # 0.8 s gap == not legato
    assert _count_slides([a, far], bridge) == 0


def test_wilson_halfwidth_shrinks_with_support():
    assert _wilson_halfwidth(0.70, 0) == 1.0
    wide = _wilson_halfwidth(0.70, 50)
    tight = _wilson_halfwidth(0.70, 5000)
    assert 0.0 < tight < wide < 1.0


def test_census_track_end_to_end(tmp_path: Path):
    # A synthetic JAMS: one held-bend note + one legato slide pair on string 0.
    bend = _ramp_samples(0.0, 0.5, lambda dt: 60.0 if dt < 0.25 else 61.6)
    slide_a = _ramp_samples(1.0, 0.2, lambda dt: 60.0)
    slide_b = _ramp_samples(1.22, 0.2, lambda dt: 63.0)
    contour_samples = bend + slide_a + slide_b

    jams = {
        "annotations": [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "0"},
                "data": [
                    {"time": 0.0, "duration": 0.5, "value": 60.0},  # bend
                    {"time": 1.0, "duration": 0.2, "value": 60.0},  # slide from
                    {"time": 1.22, "duration": 0.2, "value": 63.0},  # slide to
                ],
            },
            {
                "namespace": "pitch_contour",
                "annotation_metadata": {"data_source": "0"},
                "data": {
                    "time": [t for t, _ in contour_samples],
                    "duration": [0.0] * len(contour_samples),
                    "value": [
                        {"voiced": True, "index": 0, "frequency": _midi_to_hz(m)}
                        for _, m in contour_samples
                    ],
                    "confidence": [None] * len(contour_samples),
                },
            },
        ]
    }
    path = tmp_path / "synthetic.jams"
    path.write_text(json.dumps(jams), encoding="utf-8")

    census = census_track(path, BEND_SEMITONES_PRIMARY)
    assert census.n_notes == 3
    assert census.n_bend == 1  # only the held-bend note
    assert census.n_slide == 1  # the legato 60 -> 63 pair
