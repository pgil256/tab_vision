"""Tests for ``manifest_builder.scan_gaps`` (GAPS dataset discovery + filtering)."""

from __future__ import annotations

import json
from pathlib import Path

from tabvision.eval.manifest_builder import scan_gaps

_STANDARD = {1: ("E", 2), 2: ("A", 2), 3: ("D", 3), 4: ("G", 3), 5: ("B", 3), 6: ("E", 4)}
_DROP_D = {**_STANDARD, 1: ("D", 2)}


def _xml_with_tuning(tuning: dict[int, tuple[str, int]]) -> str:
    staff = "".join(
        f'<staff-tuning line="{line}"><tuning-step>{s}</tuning-step>'
        f"<tuning-octave>{o}</tuning-octave></staff-tuning>"
        for line, (s, o) in sorted(tuning.items())
    )
    return (
        '<?xml version="1.0"?><score-partwise version="3.1">'
        '<part-list><score-part id="P2"><part-name>TAB</part-name></score-part></part-list>'
        '<part id="P2"><measure number="1">'
        f"<attributes><divisions>4</divisions><staff-details>{staff}</staff-details></attributes>"
        "<note><pitch><step>E</step><octave>2</octave></pitch><duration>4</duration>"
        "<notations><technical><string>6</string><fret>0</fret></technical></notations></note>"
        "</measure></part></score-partwise>"
    )


def _make_clip(
    root: Path,
    stem: str,
    *,
    tuning: dict[int, tuple[str, int]] = _STANDARD,
    with_audio: bool = True,
    with_midi: bool = True,
    with_sync: bool = True,
) -> None:
    for sub in ("musicxml", "midi", "syncpoints", "audio"):
        (root / sub).mkdir(exist_ok=True)
    (root / "musicxml" / f"{stem}.xml").write_text(_xml_with_tuning(tuning), encoding="utf-8")
    if with_midi:
        (root / "midi" / f"{stem}.mid").write_bytes(b"MThd")  # existence only
    if with_sync:
        (root / "syncpoints" / f"{stem}.json").write_text(json.dumps([[0, 0.0]]), encoding="utf-8")
    if with_audio:
        (root / "audio" / f"{stem}.wav").write_bytes(b"RIFF")  # existence only


def _write_csv(root: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["id,split"] + [f"{stem},{split}" for stem, split in rows]
    (root / "gaps_metadata_with_splits.csv").write_text("\n".join(lines), encoding="utf-8")


def test_includes_standard_tuning_clip_with_split(tmp_path: Path) -> None:
    _make_clip(tmp_path, "001_aa")
    _write_csv(tmp_path, [("001_aa", "test")])

    entries = scan_gaps(tmp_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == "gaps/001_aa"
    assert entry.tier == "clean_acoustic_single_line"
    assert entry.source == "GAPS"
    assert entry.split == "test"
    assert entry.annotation_format == "gaps_musicxml_tab"
    assert entry.annotation_path.endswith("001_aa.xml")
    assert entry.media_path.endswith("001_aa.wav")


def test_scordatura_dropped_by_default_kept_when_disabled(tmp_path: Path) -> None:
    _make_clip(tmp_path, "001_std")
    _make_clip(tmp_path, "002_dropd", tuning=_DROP_D)
    _write_csv(tmp_path, [("001_std", "test"), ("002_dropd", "test")])

    default = scan_gaps(tmp_path)
    assert {e.id for e in default} == {"gaps/001_std"}

    all_tunings = scan_gaps(tmp_path, standard_tuning_only=False)
    assert {e.id for e in all_tunings} == {"gaps/001_std", "gaps/002_dropd"}


def test_unlabeled_split_excluded(tmp_path: Path) -> None:
    _make_clip(tmp_path, "001_aa")
    _make_clip(tmp_path, "002_bb")
    _write_csv(tmp_path, [("001_aa", "train"), ("002_bb", "")])  # 002 unlabeled

    entries = scan_gaps(tmp_path)
    assert {e.id for e in entries} == {"gaps/001_aa"}


def test_missing_sibling_excludes_clip(tmp_path: Path) -> None:
    _make_clip(tmp_path, "001_aa", with_audio=False)  # no .wav
    _write_csv(tmp_path, [("001_aa", "test")])

    assert scan_gaps(tmp_path) == []


def test_missing_root_returns_empty(tmp_path: Path) -> None:
    assert scan_gaps(tmp_path / "does_not_exist") == []
