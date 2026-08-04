from __future__ import annotations

import json

import pytest

from tabvision.errors import InvalidInputError
from tabvision.personal.bank import bank_corrected_document


def _document(**overrides):
    value = {
        "id": "take-1",
        "capoFret": 0,
        "tuning": ["E", "A", "D", "G", "B", "E"],
        "tuningMidi": [40, 45, 50, 55, 59, 64],
        "notes": [
            {"timestamp": 0.2, "string": 6, "fret": 3},
            {"timestamp": 0.8, "string": 1, "fret": 0},
        ],
    }
    value.update(overrides)
    return value


def test_audio_document_banks_prior_without_video_frames(tmp_path):
    source = tmp_path / "take.wav"
    source.write_bytes(b"fake")
    document = tmp_path / "editor.json"
    document.write_text(json.dumps(_document()), encoding="utf-8")

    summary = bank_corrected_document(source, document, root=tmp_path / "personal")

    assert summary["notes"] == 2
    assert summary["frames_written"] == 0
    assert summary["session_dir"] is None
    assert summary["prior_labels"] == 2
    assert (tmp_path / "personal" / "labels.jsonl").is_file()


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"capoFret": 2}, "capo 0"),
        ({"tuningMidi": [38, 45, 50, 55, 59, 64]}, "standard tuning"),
    ],
)
def test_banking_enforces_store_domain_guards(tmp_path, patch, message):
    source = tmp_path / "take.wav"
    source.write_bytes(b"fake")
    document = tmp_path / "editor.json"
    document.write_text(json.dumps(_document(**patch)), encoding="utf-8")

    with pytest.raises(InvalidInputError, match=message):
        bank_corrected_document(source, document, root=tmp_path / "personal")
