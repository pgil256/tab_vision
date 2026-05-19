"""Tests for the annotation-parser registry (Phase 0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabvision.eval.parsers import (
    clear_parsers,
    get_parser,
    list_parsers,
    register_parser,
)
from tabvision.eval.parsers.registry import _PARSERS as _GLOBAL_PARSERS


@pytest.fixture
def isolated_registry():
    """Save + restore the registry around tests that mutate it."""
    saved = dict(_GLOBAL_PARSERS)
    yield
    clear_parsers()
    _GLOBAL_PARSERS.update(saved)


def test_builtin_parsers_registered_on_import():
    """The package import should auto-register at least GuitarSet JAMS."""
    parsers = list_parsers()
    assert "guitarset_jams" in parsers


def test_get_parser_returns_callable():
    parser = get_parser("guitarset_jams")
    assert callable(parser)


def test_get_parser_raises_keyerror_with_known_formats_listed():
    with pytest.raises(KeyError) as excinfo:
        get_parser("nonexistent_format")
    assert "guitarset_jams" in str(excinfo.value)


def test_register_parser_rejects_duplicate(isolated_registry):
    def fake_parser(path, cfg=None):
        return []

    with pytest.raises(ValueError, match="already registered"):
        register_parser("guitarset_jams", fake_parser)


def test_register_then_get_roundtrip(isolated_registry):
    def fake_parser(path, cfg=None):
        return []

    register_parser("fake_format", fake_parser)
    assert get_parser("fake_format") is fake_parser
    assert "fake_format" in list_parsers()


def test_dispatch_via_registry_parses_jams(tmp_path: Path):
    """End-to-end: composite-eval dispatch path runs through the registry."""
    payload = {
        "annotations": [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "0"},
                "data": [
                    {"time": 0.10, "duration": 0.25, "value": 42},
                ],
            }
        ]
    }
    jams_path = tmp_path / "clip.jams"
    jams_path.write_text(json.dumps(payload), encoding="utf-8")

    parser = get_parser("guitarset_jams")
    events = parser(jams_path, None)

    assert len(events) == 1
    assert events[0].string_idx == 0
    assert events[0].pitch_midi == 42
    # Low E = MIDI 40, so MIDI 42 on string 0 → fret 2.
    assert events[0].fret == 2
