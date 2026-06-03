"""Tests for the Guitar-TECHS MIDI parser (Phase 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from tabvision.eval.parsers import get_parser  # noqa: E402
from tabvision.eval.parsers.guitar_techs_midi import (  # noqa: E402
    DEFAULT_TRACK_TO_STRING,
    parse,
)
from tabvision.types import GuitarConfig  # noqa: E402


def _make_midi(tmp_path: Path, *tracks_of_notes: list[tuple[int, float, float]]) -> Path:
    """Build a multi-track MIDI fixture.

    Each positional arg is a list of ``(pitch, start, end)`` tuples for
    one track. Pass an empty list to create an empty track.
    """
    midi = pretty_midi.PrettyMIDI()
    for notes in tracks_of_notes:
        instrument = pretty_midi.Instrument(program=24)  # acoustic guitar
        for pitch, start, end in notes:
            instrument.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            )
        midi.instruments.append(instrument)
    midi_path = tmp_path / "clip.mid"
    midi.write(str(midi_path))
    return midi_path


def test_track_zero_maps_to_low_e_string(tmp_path: Path) -> None:
    """Track 0 should carry low-E notes (string_idx 0, MIDI 40 → fret 0)."""
    midi_path = _make_midi(
        tmp_path,
        [(40, 0.0, 0.5)],
        [],
        [],
        [],
        [],
        [],
    )

    events = parse(midi_path)

    assert len(events) == 1
    assert events[0].string_idx == 0
    assert events[0].fret == 0
    assert events[0].pitch_midi == 40


def test_per_string_pitch_to_fret_derivation(tmp_path: Path) -> None:
    """Pitch minus open-string MIDI gives the fret for each string."""
    # Standard tuning MIDI: (40, 45, 50, 55, 59, 64) — low E .. high E.
    midi_path = _make_midi(
        tmp_path,
        [(40, 0.00, 0.10)],  # track 0 (E2)  → fret 0
        [(50, 0.10, 0.20)],  # track 1 (A2 + 5 semitones) → fret 5
        [(55, 0.20, 0.30)],  # track 2 (D3 + 5 semitones) → fret 5
        [(62, 0.30, 0.40)],  # track 3 (G3 + 7 semitones) → fret 7
        [(64, 0.40, 0.50)],  # track 4 (B3 + 5 semitones) → fret 5
        [(76, 0.50, 0.60)],  # track 5 (high E + 12) → fret 12
    )

    events = parse(midi_path)

    by_string = {ev.string_idx: ev.fret for ev in events}
    assert by_string == {0: 0, 1: 5, 2: 5, 3: 7, 4: 5, 5: 12}


def test_drops_notes_outside_fret_range(tmp_path: Path) -> None:
    """Notes that imply fret < 0 or > max_fret are skipped silently."""
    # MIDI 35 < open low-E (40) → fret -5, drop.
    # MIDI 90 > 40+24 → fret 50, drop.
    midi_path = _make_midi(
        tmp_path,
        [(35, 0.0, 0.1), (90, 0.5, 0.6)],
        [],
        [],
        [],
        [],
        [],
    )

    assert parse(midi_path) == []


def test_events_sorted_by_onset(tmp_path: Path) -> None:
    """Output is sorted by ``(onset_s, string_idx, fret)`` regardless of input order."""
    midi_path = _make_midi(
        tmp_path,
        [(40, 2.00, 2.10), (40, 0.00, 0.10)],
        [],
        [],
        [],
        [],
        [],
    )

    events = parse(midi_path)
    assert [ev.onset_s for ev in events] == [0.0, 2.0]


def test_capo_filters_below_capo_fret(tmp_path: Path) -> None:
    """``cfg.capo`` raises the lower-bound for accepted frets."""
    midi_path = _make_midi(
        tmp_path,
        [(40, 0.0, 0.1), (42, 0.1, 0.2)],
        [],
        [],
        [],
        [],
        [],
    )

    cfg = GuitarConfig(capo=3)
    events = parse(midi_path, cfg)
    # MIDI 40 → fret 0 < capo 3, dropped. MIDI 42 → fret 2 < 3, dropped.
    assert events == []


def test_extra_tracks_beyond_six_are_ignored(tmp_path: Path) -> None:
    """If a MIDI has > 6 tracks, only the first 6 are read."""
    midi_path = _make_midi(
        tmp_path,
        [(40, 0.0, 0.1)],
        [],
        [],
        [],
        [],
        [],
        [(40, 0.0, 0.1)],  # 7th track — outside the mapping
    )

    events = parse(midi_path)
    assert len(events) == 1
    assert events[0].string_idx == 0


def test_custom_track_to_string_mapping(tmp_path: Path) -> None:
    """A reversed mapping should put track 0's notes on high E."""
    midi_path = _make_midi(
        tmp_path,
        [(64, 0.0, 0.1)],
        [],
        [],
        [],
        [],
        [],
    )

    reversed_map: tuple[int, ...] = (5, 4, 3, 2, 1, 0)
    events = parse(midi_path, track_to_string=reversed_map)

    assert len(events) == 1
    assert events[0].string_idx == 5
    assert events[0].fret == 0


def test_default_mapping_is_identity() -> None:
    assert DEFAULT_TRACK_TO_STRING == (0, 1, 2, 3, 4, 5)


def test_dispatch_via_registry(tmp_path: Path) -> None:
    """End-to-end: parser is reachable via the composite-eval dispatch path."""
    midi_path = _make_midi(
        tmp_path,
        [(40, 0.0, 0.1)],
        [],
        [],
        [],
        [],
        [],
    )
    parser = get_parser("guitar_techs_midi")
    assert parser is parse

    events = parser(midi_path, None)
    assert len(events) == 1
