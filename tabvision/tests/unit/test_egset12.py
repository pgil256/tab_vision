"""Focused tests for EGSet12 parsing and discovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tabvision.eval.egset12 as egset12_module
from tabvision.eval.egset12 import (
    PUBLISHED_FILE_BY_NAME,
    TRACK_IDS,
    PublishedFile,
    default_egset12_root,
    parse_egset12_jams,
    scan_egset12,
)
from tabvision.types import GuitarConfig


def _write_jams(path: Path, annotations: list[dict]) -> None:
    path.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")


def _sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def test_parse_observed_note_midi_string_slices(tmp_path: Path) -> None:
    path = tmp_path / "01.jams"
    _write_jams(
        path,
        [
            {"namespace": "tempo", "data": [{"time": 0, "value": 120}]},
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "5"},
                "data": [
                    {"time": 1.0, "duration": 0.25, "value": 67, "confidence": 0.8},
                ],
            },
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": 0},
                "data": [
                    {"time": 0.5, "duration": 0.4, "value": 43},
                    {"time": 2.0, "duration": 0.1, "value": 39},  # below open string
                    {"time": "bad", "duration": 0.1, "value": 45},
                ],
            },
        ],
    )

    events = parse_egset12_jams(path)

    assert [
        (event.onset_s, event.string_idx, event.fret, event.pitch_midi) for event in events
    ] == [
        (0.5, 0, 3, 43),
        (1.0, 5, 3, 67),
    ]
    assert events[0].confidence == 1.0
    assert events[1].confidence == 0.8


def test_note_midi_wins_over_note_tab_to_avoid_duplicate_gold(tmp_path: Path) -> None:
    path = tmp_path / "02.jams"
    _write_jams(
        path,
        [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": 0},
                "data": [{"time": 0.1, "duration": 0.2, "value": 40}],
            },
            {
                "namespace": "note_tab",
                "sandbox": {"string_index": 6, "open_tuning": 40},
                "data": [{"time": 0.1, "duration": 0.2, "value": {"fret": 0}}],
            },
        ],
    )

    events = parse_egset12_jams(path)

    assert len(events) == 1
    assert (events[0].string_idx, events[0].fret, events[0].pitch_midi) == (0, 0, 40)


def test_parse_note_tab_fallback_uses_explicit_standard_tuning(tmp_path: Path) -> None:
    path = tmp_path / "03.jams"
    _write_jams(
        path,
        [
            {
                "namespace": "note_tab",
                "sandbox": {"string_index": 1, "open_tuning": 64},
                "data": [{"time": 0.7, "duration": 0.3, "value": {"fret": 2, "pitch": 66}}],
            },
            {
                "namespace": "note_tab",
                "sandbox": {"string_index": 6, "open_tuning": 40},
                "data": [{"time": 0.2, "duration": 0.5, "value": {"fret": 5}}],
            },
        ],
    )

    events = parse_egset12_jams(path)

    assert [(event.string_idx, event.fret, event.pitch_midi) for event in events] == [
        (0, 5, 45),
        (5, 2, 66),
    ]


def test_parser_rejects_nonstandard_tuning(tmp_path: Path) -> None:
    path = tmp_path / "04.jams"
    _write_jams(
        path,
        [
            {
                "namespace": "note_tab",
                "sandbox": {"string_index": 6, "open_tuning": 38},
                "data": [{"time": 0.0, "duration": 0.2, "value": {"fret": 0}}],
            }
        ],
    )

    with pytest.raises(ValueError, match="non-standard open pitch"):
        parse_egset12_jams(path)
    with pytest.raises(ValueError, match="standard tuning only"):
        parse_egset12_jams(path, GuitarConfig(tuning_midi=(38, 45, 50, 55, 59, 64)))


def test_default_root_honors_tabvision_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABVISION_DATA_ROOT", str(tmp_path))
    assert default_egset12_root() == tmp_path / "egset12"


def test_scan_is_all_or_nothing_and_emits_twelve_clean_electric_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        egset12_module,
        "_md5_file",
        lambda path: PUBLISHED_FILE_BY_NAME[path.name].md5,
    )
    assert scan_egset12(tmp_path) == []

    for track_id in TRACK_IDS:
        for suffix in (".jams", ".wav"):
            published = PUBLISHED_FILE_BY_NAME[f"{track_id}{suffix}"]
            _sparse_file(tmp_path / published.name, published.size_bytes)

    entries = scan_egset12(tmp_path)

    assert len(entries) == 12
    assert [entry.id for entry in entries] == [f"egset12/{track_id}" for track_id in TRACK_IDS]
    assert {entry.tier for entry in entries} == {"clean_electric"}
    assert {entry.source for entry in entries} == {"EGSet12"}
    assert {entry.split for entry in entries} == {"test"}
    assert {entry.annotation_format for entry in entries} == {"egset12_jams"}

    # A published-size mismatch invalidates the entire corpus rather than
    # silently emitting a partial transfer set.
    (tmp_path / "12.jams").write_bytes(b"partial")
    assert scan_egset12(tmp_path) == []


def test_scan_supports_authors_audios_and_jams_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        egset12_module,
        "_md5_file",
        lambda path: PUBLISHED_FILE_BY_NAME[path.name].md5,
    )
    for track_id in TRACK_IDS:
        wav = PUBLISHED_FILE_BY_NAME[f"{track_id}.wav"]
        jams = PUBLISHED_FILE_BY_NAME[f"{track_id}.jams"]
        _sparse_file(tmp_path / "audios" / wav.name, wav.size_bytes)
        _sparse_file(tmp_path / "jams_corrected" / jams.name, jams.size_bytes)

    entries = scan_egset12(tmp_path)

    assert len(entries) == 12
    assert all("/audios/" in entry.media_path.replace("\\", "/") for entry in entries)
    assert all("/jams_corrected/" in entry.annotation_path.replace("\\", "/") for entry in entries)


def test_scan_rejects_same_size_wrong_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, PublishedFile] = {}
    payloads: dict[str, bytes] = {}
    for track_id in TRACK_IDS:
        for suffix in (".jams", ".wav"):
            name = f"{track_id}{suffix}"
            payload = f"official-egset12:{name}".encode()
            payloads[name] = payload
            specs[name] = PublishedFile(
                name=name,
                size_bytes=len(payload),
                md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            )
    monkeypatch.setattr(egset12_module, "PUBLISHED_FILE_BY_NAME", specs)

    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    assert len(scan_egset12(tmp_path)) == 12

    path = tmp_path / "11.wav"
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert path.stat().st_size == specs[path.name].size_bytes
    assert scan_egset12(tmp_path) == []
