"""Unit tests for the PDMX TAB-staff scan's pure core (A15 acquisition).

Exercises the CSV row filter, path normalization, MXL container resolution,
and the TAB-walk validation on a synthetic in-memory MXL — no archive or
network needed.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from scripts.acquire.pdmx_tab_scan import (
    is_guitar_row,
    normalize_member,
    score_xml_bytes,
    validate_tab_walk,
)

# E2 / A2 / C3 on strings 6 / 5 / 5 — all pitch-consistent under standard tuning.
_TAB_XML = b"""<score-partwise>
  <part id="P1"><measure number="1">
    <note><pitch><step>E</step><octave>2</octave></pitch><duration>4</duration></note>
  </measure></part>
  <part id="P2"><measure number="1">
    <note><pitch><step>E</step><octave>2</octave></pitch><duration>4</duration>
      <notations><technical><string>6</string><fret>0</fret></technical></notations></note>
    <note><pitch><step>A</step><octave>2</octave></pitch><duration>4</duration>
      <notations><technical><string>5</string><fret>0</fret></technical></notations></note>
    <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration>
      <notations><technical><string>5</string><fret>3</fret></technical></notations></note>
  </measure></part>
</score-partwise>"""

_CONTAINER = b'<container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>'


def _mxl(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ---------- is_guitar_row ----------


def _row(**overrides) -> dict[str, str]:
    row = {
        "subset:no_license_conflict": "True",
        "mxl": "./mxl/1/11/Qm.mxl",
        "tracks": "0-25",
    }
    row.update(overrides)
    return row


def test_guitar_row_accepted():
    assert is_guitar_row(_row()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"subset:no_license_conflict": "False"},
        {"mxl": "NA"},
        {"mxl": ""},
        {"tracks": "0-52"},  # no guitar program
        {"tracks": "NA"},
        {"tracks": ""},
    ],
)
def test_non_guitar_rows_rejected(overrides):
    assert is_guitar_row(_row(**overrides)) is False


def test_all_guitar_program_family_accepted():
    for program in range(24, 32):
        assert is_guitar_row(_row(tracks=str(program))) is True


# ---------- normalize_member ----------


def test_normalize_member_strips_dot_prefix_and_backslashes():
    assert normalize_member("./mxl/1/11/x.mxl") == "mxl/1/11/x.mxl"
    assert normalize_member("mxl\\1\\11\\x.mxl") == "mxl/1/11/x.mxl"


# ---------- score_xml_bytes ----------


def test_score_xml_resolved_via_container():
    mxl = _mxl({"META-INF/container.xml": _CONTAINER, "score.xml": _TAB_XML})
    assert score_xml_bytes(mxl) == _TAB_XML


def test_score_xml_falls_back_to_first_xml_without_container():
    mxl = _mxl({"piece.xml": _TAB_XML})
    assert score_xml_bytes(mxl) == _TAB_XML


def test_score_xml_none_when_no_xml_entries():
    assert score_xml_bytes(_mxl({"readme.txt": b"nope"})) is None


# ---------- validate_tab_walk ----------


def test_tab_walk_counts_and_pitch_consistency():
    result = validate_tab_walk(_TAB_XML)
    assert result["n_tab_notes"] == 3
    assert result["n_pitch_consistent"] == 3
    assert result["consistency"] == 1.0
    assert result["declares_staff_tuning"] is False
    assert result["tuning_is_standard"] is True


def test_tab_walk_distinguishes_declared_standard_from_nonstandard():
    # MuseScore always writes <staff-tuning>; declaring standard EADGBE must
    # not read as a nonstandard-tuning score. Line 1 = bottom tab line =
    # MusicXML string 6 (low E).
    def attrs(low_e_step: str, low_e_octave: int) -> bytes:
        lines = [
            (1, low_e_step, low_e_octave),
            (2, "A", 2),
            (3, "D", 3),
            (4, "G", 3),
            (5, "B", 3),
            (6, "E", 4),
        ]
        tunings = b"".join(
            b'<staff-tuning line="%d"><tuning-step>%s</tuning-step>'
            b"<tuning-octave>%d</tuning-octave></staff-tuning>" % (line, step.encode(), octave)
            for line, step, octave in lines
        )
        return b"<attributes><staff-details>" + tunings + b"</staff-details></attributes>"

    standard = _TAB_XML.replace(
        b'<part id="P2"><measure number="1">', b'<part id="P2"><measure number="1">' + attrs("E", 2)
    )
    result = validate_tab_walk(standard)
    assert result["declares_staff_tuning"] is True
    assert result["tuning_is_standard"] is True

    drop_d = _TAB_XML.replace(
        b'<part id="P2"><measure number="1">', b'<part id="P2"><measure number="1">' + attrs("D", 2)
    )
    result = validate_tab_walk(drop_d)
    assert result["tuning_is_standard"] is False


def test_tab_walk_flags_inconsistent_fret():
    # A2 marked as string 5 fret 5 would sound C#3 (50), not A2 (45).
    bad = _TAB_XML.replace(b"<string>5</string><fret>3</fret>", b"<string>5</string><fret>5</fret>")
    result = validate_tab_walk(bad)
    assert result["n_tab_notes"] == 3
    assert result["n_pitch_consistent"] == 2
