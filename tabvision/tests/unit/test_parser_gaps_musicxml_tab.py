"""Tests for the GAPS MusicXML-tab parser.

Builds synthetic ``musicxml/ + midi/ + syncpoints/`` sibling fixtures and
checks the gold-derivation contract: staff-tuning awareness (scordatura),
the ``string_idx = 6 - musicxml_string`` mapping, exact onset-snapping to the
aligned MIDI, and dropping of score notes the performer never played.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

import xml.etree.ElementTree as ET  # noqa: E402

from tabvision.eval.parsers import get_parser  # noqa: E402
from tabvision.eval.parsers.gaps_musicxml_tab import _unfold_measures, parse  # noqa: E402
from tabvision.types import GuitarConfig  # noqa: E402


def _measure(
    *,
    forward: bool = False,
    backward: int | str | None = None,
    ending_start: str | None = None,
    ending_stop: bool = False,
) -> ET.Element:
    """Build a bare <measure> carrying only the repeat/volta barlines under test."""
    bars = ""
    if forward:
        bars += '<barline location="left"><repeat direction="forward"/></barline>'
    if ending_start is not None:
        bars += f'<barline location="left"><ending number="{ending_start}" type="start"/></barline>'
    if ending_stop:
        bars += '<barline location="right"><ending number="1" type="stop"/></barline>'
    if backward is not None:
        times = f' times="{backward}"' if backward is not None else ""
        bars += f'<barline location="right"><repeat direction="backward"{times}/></barline>'
    return ET.fromstring(f"<measure>{bars}</measure>")


# line -> (step, octave) for the six tab lines (bottom=1 .. top=6).
_STANDARD_TUNING = {1: ("E", 2), 2: ("A", 2), 3: ("D", 3), 4: ("G", 3), 5: ("B", 3), 6: ("E", 4)}
_DROP_D_TUNING = {**_STANDARD_TUNING, 1: ("D", 2)}  # string 6 dropped E2 -> D2


def _note_xml(step: str, octave: int, duration: int, string: int, fret: int, alter: int = 0) -> str:
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    return (
        "<note>"
        f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
        f"<duration>{duration}</duration>"
        f"<notations><technical><string>{string}</string><fret>{fret}</fret></technical></notations>"
        "</note>"
    )


def _build_dataset(
    tmp_path: Path,
    *,
    measure_notes: list[str],
    midi_notes: list[tuple[int, float, float]],
    syncpoints: list[list[float]],
    divisions: int = 4,
    tuning: dict[int, tuple[str, int]] | None = None,
    stem: str = "clip",
    write_midi: bool = True,
    write_sync: bool = True,
) -> Path:
    """Lay out a GAPS-style root and return the MusicXML annotation path."""
    tuning = tuning or _STANDARD_TUNING
    staff = "".join(
        f'<staff-tuning line="{line}"><tuning-step>{s}</tuning-step>'
        f"<tuning-octave>{o}</tuning-octave></staff-tuning>"
        for line, (s, o) in sorted(tuning.items())
    )
    xml = (
        '<?xml version="1.0"?>'
        '<score-partwise version="3.1">'
        "<part-list>"
        '<score-part id="P1"><part-name>Guitar</part-name></score-part>'
        '<score-part id="P2"><part-name>TAB</part-name></score-part>'
        "</part-list>"
        # P1: notation only, no <string> markings -> not the tab part.
        '<part id="P1"><measure number="1">'
        f"<attributes><divisions>{divisions}</divisions></attributes>"
        "<note><pitch><step>E</step><octave>2</octave></pitch>"
        f"<duration>{divisions * 4}</duration></note>"
        "</measure></part>"
        # P2: tablature part with staff-tuning + per-note string/fret.
        '<part id="P2"><measure number="1">'
        f"<attributes><divisions>{divisions}</divisions>"
        f"<staff-details>{staff}</staff-details></attributes>"
        f"{''.join(measure_notes)}"
        "</measure></part>"
        "</score-partwise>"
    )
    (tmp_path / "musicxml").mkdir(exist_ok=True)
    (tmp_path / "midi").mkdir(exist_ok=True)
    (tmp_path / "syncpoints").mkdir(exist_ok=True)
    xml_path = tmp_path / "musicxml" / f"{stem}.xml"
    xml_path.write_text(xml, encoding="utf-8")

    if write_midi:
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=24)
        for pitch, start, end in midi_notes:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
        midi.instruments.append(inst)
        midi.write(str(tmp_path / "midi" / f"{stem}.mid"))
    if write_sync:
        (tmp_path / "syncpoints" / f"{stem}.json").write_text(
            json.dumps(syncpoints), encoding="utf-8"
        )
    return xml_path


def test_standard_tuning_string_mapping_and_midi_snap(tmp_path: Path) -> None:
    """string_idx = 6 - musicxml_string; onsets come from the MIDI, not the warp."""
    # Three quarter notes (divisions=4) on strings 6/5/1 at score divisions 0/4/8.
    notes = [
        _note_xml("E", 2, 4, string=6, fret=0),  # E2 -> string_idx 0
        _note_xml("D", 3, 4, string=5, fret=5),  # A2+5 -> string_idx 1, D3
        _note_xml("G", 4, 4, string=1, fret=3),  # E4+3 -> string_idx 5, G4
    ]
    # Performance onsets deliberately offset from the linear warp (0,1,2 s).
    midi = [(40, 0.10, 0.40), (50, 1.23, 1.50), (67, 2.40, 2.70)]
    sync = [[0, 0.0], [1, 4.0]]  # measure 0 -> 0 s, measure 1 -> 4 s (16 div span)
    xml = _build_dataset(tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync)

    events = parse(xml, GuitarConfig())

    assert [(e.string_idx, e.fret, e.pitch_midi) for e in events] == [
        (0, 0, 40),
        (1, 5, 50),
        (5, 3, 67),
    ]
    # Onsets are the exact MIDI onsets (snapped), not the warp estimate.
    assert [round(e.onset_s, 2) for e in events] == [0.10, 1.23, 2.40]
    assert [round(e.duration_s, 2) for e in events] == [0.30, 0.27, 0.30]


def test_scordatura_drop_d_via_staff_tuning(tmp_path: Path) -> None:
    """A drop-D low string (D2) is read from <staff-tuning>; pitch stays D2."""
    notes = [_note_xml("D", 2, 4, string=6, fret=0)]  # open 6th string = D2 in drop-D
    midi = [(38, 0.05, 0.40)]  # D2 = MIDI 38
    sync = [[0, 0.0], [1, 4.0]]
    xml = _build_dataset(
        tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync, tuning=_DROP_D_TUNING
    )

    events = parse(xml, GuitarConfig())

    assert len(events) == 1
    assert (events[0].string_idx, events[0].fret, events[0].pitch_midi) == (0, 0, 38)
    assert round(events[0].onset_s, 2) == 0.05


def test_score_note_without_midi_counterpart_is_dropped(tmp_path: Path) -> None:
    """A written note the performer never played (no same-pitch MIDI) is excluded."""
    notes = [
        _note_xml("E", 2, 4, string=6, fret=0),  # E2 — played
        _note_xml("C", 4, 4, string=2, fret=1),  # C4 — NOT in the MIDI
    ]
    midi = [(40, 0.10, 0.40)]  # only E2
    sync = [[0, 0.0], [1, 4.0]]
    xml = _build_dataset(tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync)

    events = parse(xml, GuitarConfig())

    assert [e.pitch_midi for e in events] == [40]


def test_repeated_pitch_matches_first_traversal_in_order(tmp_path: Path) -> None:
    """Two written A2s align to the first two of three performed A2s (monotonic)."""
    notes = [
        _note_xml("A", 2, 8, string=5, fret=0),  # A2 at div 0
        _note_xml("A", 2, 8, string=5, fret=0),  # A2 at div 8
    ]
    # Performer played the bar, then repeated the first A2 (3 onsets total).
    midi = [(45, 0.0, 0.4), (45, 1.0, 1.4), (45, 2.0, 2.4)]
    sync = [[0, 0.0], [1, 2.0]]
    xml = _build_dataset(tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync)

    events = parse(xml, GuitarConfig())

    assert len(events) == 2
    assert [round(e.onset_s, 1) for e in events] == [0.0, 1.0]


def test_fret_out_of_range_dropped(tmp_path: Path) -> None:
    """A fret above max_fret is skipped."""
    notes = [_note_xml("E", 2, 4, string=6, fret=0)]
    midi = [(40, 0.1, 0.4)]
    sync = [[0, 0.0], [1, 4.0]]
    xml = _build_dataset(tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync)

    assert parse(xml, GuitarConfig(max_fret=-1)) == []


def test_dispatch_via_registry(tmp_path: Path) -> None:
    notes = [_note_xml("E", 2, 4, string=6, fret=0)]
    midi = [(40, 0.1, 0.4)]
    sync = [[0, 0.0], [1, 4.0]]
    xml = _build_dataset(tmp_path, measure_notes=notes, midi_notes=midi, syncpoints=sync)

    parser = get_parser("gaps_musicxml_tab")
    assert parser is parse
    assert len(parser(xml, None)) == 1


def test_missing_midi_sibling_raises(tmp_path: Path) -> None:
    notes = [_note_xml("E", 2, 4, string=6, fret=0)]
    sync = [[0, 0.0], [1, 4.0]]
    xml = _build_dataset(
        tmp_path, measure_notes=notes, midi_notes=[], syncpoints=sync, write_midi=False
    )
    with pytest.raises(FileNotFoundError, match="MIDI sibling"):
        parse(xml, None)


def test_missing_syncpoints_sibling_raises(tmp_path: Path) -> None:
    notes = [_note_xml("E", 2, 4, string=6, fret=0)]
    xml = _build_dataset(
        tmp_path, measure_notes=notes, midi_notes=[(40, 0.1, 0.4)], syncpoints=[], write_sync=False
    )
    with pytest.raises(FileNotFoundError, match="syncpoints sibling"):
        parse(xml, None)


# --- A6: repeat/volta unfolding -------------------------------------------------


def test_unfold_no_repeats_is_identity() -> None:
    measures = [_measure(), _measure(), _measure()]
    assert _unfold_measures(measures) == [0, 1, 2]


def test_unfold_simple_repeat_doubles_section() -> None:
    # m0, |: m1 m2 :|, m3  ->  0 1 2 1 2 3
    measures = [
        _measure(),
        _measure(forward=True),
        _measure(backward=2),
        _measure(),
    ]
    assert _unfold_measures(measures) == [0, 1, 2, 1, 2, 3]


def test_unfold_backward_without_forward_loops_from_start() -> None:
    # m0 m1 :|  -> 0 1 0 1
    measures = [_measure(), _measure(backward=2)]
    assert _unfold_measures(measures) == [0, 1, 0, 1]


def test_unfold_times_three() -> None:
    measures = [_measure(forward=True), _measure(backward=3)]
    assert _unfold_measures(measures) == [0, 1, 0, 1, 0, 1]


def test_unfold_first_and_second_endings() -> None:
    # |: A [1. B :] [2. C]  ->  A B(pass1) A C(pass2)
    measures = [
        _measure(forward=True),  # 0: A
        _measure(ending_start="1", ending_stop=True, backward=2),  # 1: volta 1 + repeat
        _measure(ending_start="2", ending_stop=True),  # 2: volta 2
    ]
    assert _unfold_measures(measures) == [0, 1, 0, 2]


def _build_multi(
    tmp_path: Path,
    *,
    p2_measures: list[str],
    midi_notes: list[tuple[int, float, float]],
    syncpoints: list[list[float]],
    divisions: int = 4,
    stem: str = "clip",
) -> Path:
    """GAPS root with a multi-measure TAB part (P2) — for repeat-unfold tests."""
    staff = "".join(
        f'<staff-tuning line="{line}"><tuning-step>{s}</tuning-step>'
        f"<tuning-octave>{o}</tuning-octave></staff-tuning>"
        for line, (s, o) in sorted(_STANDARD_TUNING.items())
    )
    # First P2 measure carries the attributes/staff-tuning.
    p2_first = (
        f'<measure number="1"><attributes><divisions>{divisions}</divisions>'
        f"<staff-details>{staff}</staff-details></attributes>{p2_measures[0]}</measure>"
    )
    p2_rest = "".join(
        f'<measure number="{i + 2}">{m}</measure>' for i, m in enumerate(p2_measures[1:])
    )
    xml = (
        '<?xml version="1.0"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Guitar</part-name></score-part>'
        '<score-part id="P2"><part-name>TAB</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        f"<attributes><divisions>{divisions}</divisions></attributes>"
        f"<note><pitch><step>E</step><octave>2</octave></pitch><duration>{divisions}</duration>"
        "</note></measure></part>"
        f'<part id="P2">{p2_first}{p2_rest}</part></score-partwise>'
    )
    (tmp_path / "musicxml").mkdir(exist_ok=True)
    (tmp_path / "midi").mkdir(exist_ok=True)
    (tmp_path / "syncpoints").mkdir(exist_ok=True)
    xml_path = tmp_path / "musicxml" / f"{stem}.xml"
    xml_path.write_text(xml, encoding="utf-8")
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    for pitch, start, end in midi_notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)
    midi.write(str(tmp_path / "midi" / f"{stem}.mid"))
    (tmp_path / "syncpoints" / f"{stem}.json").write_text(json.dumps(syncpoints), encoding="utf-8")
    return xml_path


def test_repeat_unfold_recovers_second_traversal_gold(tmp_path: Path) -> None:
    """A repeated section: gold covers both traversals once unfolded.

    Score: |: A2 (m1) | D3 (m2) :|  performed A2 D3 A2 D3. Without unfolding the
    gold is only the first A2/D3 (the artifact); unfolding restores all four.
    """
    whole = 16  # one whole measure at divisions=4
    m1 = _note_xml("A", 2, whole, string=5, fret=0)  # A2 (MIDI 45)
    m2 = _note_xml("D", 3, whole, string=5, fret=5)  # A2+5 = D3 (MIDI 50)
    p2 = [
        f'<barline location="left"><repeat direction="forward"/></barline>{m1}',
        f'{m2}<barline location="right"><repeat direction="backward"/></barline>',
    ]
    # Performance plays the section twice.
    midi = [(45, 0.0, 0.4), (50, 1.0, 1.4), (45, 2.0, 2.4), (50, 3.0, 3.4)]
    # Unfolded = 4 measures; syncpoints span 4 (indices 0..3) → unfold trusted.
    sync = [[0, 0.0], [1, 1.0], [2, 2.0], [3, 3.0]]
    xml = _build_multi(tmp_path, p2_measures=p2, midi_notes=midi, syncpoints=sync)

    events = parse(xml, GuitarConfig())

    assert [e.pitch_midi for e in events] == [45, 50, 45, 50]
    assert [round(e.onset_s, 1) for e in events] == [0.0, 1.0, 2.0, 3.0]


def test_repeat_unfold_falls_back_when_span_mismatches(tmp_path: Path) -> None:
    """When the syncpoint span is far from the unfold length (a nonstandard
    volta encoding), keep the written order — gold is the first traversal only.
    This is the safe fallback that protects the odd clips from corrupted gold."""
    whole = 16
    m1 = _note_xml("A", 2, whole, string=5, fret=0)
    m2 = _note_xml("D", 3, whole, string=5, fret=5)
    p2 = [
        f'<barline location="left"><repeat direction="forward"/></barline>{m1}',
        f'{m2}<barline location="right"><repeat direction="backward"/></barline>',
    ]
    midi = [(45, 0.0, 0.4), (50, 1.0, 1.4), (45, 2.0, 2.4), (50, 3.0, 3.4)]
    # Unfold length is 4, but the syncpoint span is 10 → |4-10| = 6 > tol(3) →
    # fallback to the written 2 measures → gold is the first traversal only.
    sync = [[k, float(k)] for k in range(10)]
    xml = _build_multi(tmp_path, p2_measures=p2, midi_notes=midi, syncpoints=sync)

    events = parse(xml, GuitarConfig())

    assert [e.pitch_midi for e in events] == [45, 50]  # first traversal only
