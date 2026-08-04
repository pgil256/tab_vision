from __future__ import annotations

from types import SimpleNamespace

from tabvision.assist.document import build_editor_document
from tabvision.eval.string_assignment import RankedCandidate
from tabvision.types import GuitarConfig, TabEvent


def test_editor_document_uses_client_strings_and_python_rankings(monkeypatch, tmp_path) -> None:
    event = TabEvent(1.25, 0.4, 5, 0, 64, 0.42)
    policy = SimpleNamespace(
        resolved_sequence_prior="sequence-v1",
        requested_position_prior="auto",
        resolved_position_prior="position-v1",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        resolved_string_evidence="physics-v1",
        artifacts=(),
    )
    result = SimpleNamespace(
        tab_events=(event,),
        audio_events=(),
        policy=policy,
        resolved_audio_backend="highres-ensemble",
        resolved_video_backend="none",
        position_observation_count=0,
        notes_affected_by_video=0,
        beat_grid=None,
    )
    observed: dict[str, object] = {}

    def fake_candidates(tab_events, audio_events, *, cfg, sequence_prior):
        observed["sequence_prior"] = sequence_prior
        return [(RankedCandidate(5, 0, 0.0), RankedCandidate(4, 5, 1.0))]

    monkeypatch.setattr("tabvision.assist.document.compute_note_candidates", fake_candidates)

    document = build_editor_document(
        result,
        cfg=GuitarConfig(),
        source_path=tmp_path / "take.mp4",
        video_enabled=False,
    )

    note = document["notes"][0]
    assert note["string"] == 1
    assert note["detectedMidiNote"] == 64
    assert note["confidenceLevel"] == "low"
    assert note["candidates"] == [{"string": 1, "fret": 0}, {"string": 2, "fret": 5}]
    assert document["tuning"] == ["E", "A", "D", "G", "B", "E"]
    assert document["tuningMidi"] == [40, 45, 50, 55, 59, 64]
    assert observed["sequence_prior"] == "sequence-v1"
