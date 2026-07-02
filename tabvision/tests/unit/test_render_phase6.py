from __future__ import annotations

from io import BytesIO

import pytest

from tabvision.types import GuitarConfig, TabEvent


def _fixture_events() -> list[TabEvent]:
    return [
        TabEvent(
            onset_s=0.0,
            duration_s=0.25,
            string_idx=5,
            fret=0,
            pitch_midi=64,
            confidence=0.95,
        ),
        TabEvent(
            onset_s=0.5,
            duration_s=0.5,
            string_idx=0,
            fret=3,
            pitch_midi=43,
            confidence=0.32,
        ),
    ]


@pytest.mark.render
def test_ascii_snapshot_marks_low_confidence_notes() -> None:
    from tabvision.render.ascii import render

    assert render(_fixture_events(), GuitarConfig()) == (
        "TabVision ASCII tab\n"
        "Tuning: E A D G B E   Capo: none   Notes: 2\n"
        "Low-confidence notes marked with '?'.\n"
        "\n"
        "e|0-----|\n"
        "B|------|\n"
        "G|------|\n"
        "D|------|\n"
        "A|------|\n"
        "E|---3?-|"
    )


@pytest.mark.render
def test_ascii_rejects_out_of_range_string_idx() -> None:
    """gp5/midi/musicxml all reject an out-of-range ``string_idx`` with a
    ``ValueError`` (see each renderer's ``_validate_event``/``_validate_string``).
    ASCII is the default format and previously had no such guard — an
    out-of-range note silently failed to match any row in
    ``_columns_to_lines`` and vanished from the tab instead of erroring."""
    from tabvision.render.ascii import render

    bad_event = TabEvent(
        onset_s=0.0, duration_s=0.25, string_idx=6, fret=0, pitch_midi=64, confidence=0.9
    )
    with pytest.raises(ValueError, match="string_idx out of range"):
        render([bad_event], GuitarConfig())


@pytest.mark.render
def test_ascii_rejects_negative_string_idx() -> None:
    from tabvision.render.ascii import render

    bad_event = TabEvent(
        onset_s=0.0, duration_s=0.25, string_idx=-1, fret=0, pitch_midi=64, confidence=0.9
    )
    with pytest.raises(ValueError, match="string_idx out of range"):
        render([bad_event], GuitarConfig())


@pytest.mark.render
def test_ascii_rejects_out_of_range_fret() -> None:
    from tabvision.render.ascii import render

    bad_event = TabEvent(
        onset_s=0.0, duration_s=0.25, string_idx=0, fret=25, pitch_midi=64, confidence=0.9
    )
    with pytest.raises(ValueError, match="fret out of range"):
        render([bad_event], GuitarConfig())


@pytest.mark.render
def test_public_render_entrypoint_returns_bytes_for_ascii() -> None:
    from tabvision.render import render

    payload = render(_fixture_events(), "ascii", GuitarConfig())
    assert isinstance(payload, bytes)
    assert b"3?" in payload


@pytest.mark.render
def test_midi_round_trip_preserves_string_channels() -> None:
    mido = pytest.importorskip("mido")
    from tabvision.render.midi import render

    midi = mido.MidiFile(file=BytesIO(render(_fixture_events(), GuitarConfig())))
    note_ons = [
        msg for track in midi.tracks for msg in track if msg.type == "note_on" and msg.velocity > 0
    ]
    assert [(msg.note, msg.channel) for msg in note_ons] == [(64, 5), (43, 0)]


@pytest.mark.render
def test_musicxml_round_trip_preserves_notes() -> None:
    music21 = pytest.importorskip("music21")
    from tabvision.render.musicxml import render

    score = music21.converter.parseData(render(_fixture_events(), GuitarConfig()).decode())
    notes = list(score.recurse().notes)
    assert [note.pitch.midi for note in notes] == [64, 43]


@pytest.mark.render
def test_gp5_round_trip_preserves_notes_when_pyguitarpro_is_available(tmp_path) -> None:
    guitarpro = pytest.importorskip("guitarpro")
    from tabvision.render.gp5 import render

    path = tmp_path / "fixture.gp5"
    path.write_bytes(render(_fixture_events(), GuitarConfig()))

    song = guitarpro.parse(str(path))
    notes = [
        note
        for track in song.tracks
        for measure in track.measures
        for voice in measure.voices
        for beat in voice.beats
        for note in beat.notes
    ]
    assert [(GuitarConfig().n_strings - note.string, note.value) for note in notes] == [
        (5, 0),
        (0, 3),
    ]
