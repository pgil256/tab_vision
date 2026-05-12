"""Unit tests for the v1 GuitarSet audio-only eval helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tabvision.errors import BackendError
from tabvision.eval.guitarset_audio import (
    EventF1Result,
    TrackEvalResult,
    _score_event_f1,
    build_guitarset_position_prior,
    list_guitarset_track_ids,
    parse_guitarset_jams,
    score_audio_only,
    summarize_results,
)
from tabvision.eval.metrics import TabF1Result
from tabvision.types import AudioEvent, TabEvent


def _write_jams(path: Path) -> None:
    payload = {
        "annotations": [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "0"},
                "data": [
                    {"time": 0.10, "duration": 0.25, "value": 44.1},
                    {"time": 0.60, "duration": 0.30, "value": 38.9},
                ],
            },
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "5"},
                "data": [
                    {"time": 1.00, "duration": 0.40, "value": 72.0},
                ],
            },
            {
                "namespace": "pitch_contour",
                "annotation_metadata": {"data_source": "5"},
                "data": {"time": [], "duration": [], "value": [], "confidence": []},
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_guitarset_jams_retains_string_fret_and_pitch(tmp_path: Path):
    jams_path = tmp_path / "clip.jams"
    _write_jams(jams_path)

    notes = parse_guitarset_jams(jams_path)

    assert [(n.onset_s, n.duration_s, n.string_idx, n.fret, n.pitch_midi) for n in notes] == [
        (0.10, 0.25, 0, 4, 44),
        (1.00, 0.40, 5, 8, 72),
    ]


def test_validation_track_listing_uses_held_out_player(tmp_path: Path):
    ann = tmp_path / "annotation"
    audio = tmp_path / "audio_mono-mic"
    ann.mkdir()
    audio.mkdir()
    for track_id in ["00_alpha", "05_beta", "05_gamma"]:
        (ann / f"{track_id}.jams").write_text("{}", encoding="utf-8")
        (audio / f"{track_id}_mic.wav").write_bytes(b"RIFF")

    assert list_guitarset_track_ids(tmp_path, split="validation") == ["05_beta", "05_gamma"]


def test_build_guitarset_position_prior_uses_train_split_only(tmp_path: Path):
    ann = tmp_path / "annotation"
    audio = tmp_path / "audio_mono-mic"
    ann.mkdir()
    audio.mkdir()

    train_jams = tmp_path / "annotation" / "00_train.jams"
    train_jams.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "namespace": "note_midi",
                        "annotation_metadata": {"data_source": "3"},
                        "data": [{"time": 0.0, "duration": 0.2, "value": 69.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    validation_jams = tmp_path / "annotation" / "05_validation.jams"
    validation_jams.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "namespace": "note_midi",
                        "annotation_metadata": {"data_source": "5"},
                        "data": [{"time": 0.0, "duration": 0.2, "value": 69.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (audio / "00_train_mic.wav").write_bytes(b"RIFF")
    (audio / "05_validation_mic.wav").write_bytes(b"RIFF")

    prior = build_guitarset_position_prior(tmp_path)
    matrix = prior.matrix_for_pitch(69)

    assert matrix is not None
    assert matrix[3, 14] > matrix[5, 5]


def test_event_f1_can_score_onsets_separately_from_pitch():
    pred = [
        TabEvent(1.00, 0.2, string_idx=0, fret=5, pitch_midi=45, confidence=1.0),
        TabEvent(2.00, 0.2, string_idx=5, fret=3, pitch_midi=67, confidence=1.0),
    ]
    gold = [
        TabEvent(1.02, 0.2, string_idx=0, fret=4, pitch_midi=44, confidence=1.0),
        TabEvent(2.02, 0.2, string_idx=5, fret=3, pitch_midi=67, confidence=1.0),
    ]

    onset = _score_event_f1(pred, gold, match_pitch=False)
    pitch = _score_event_f1(pred, gold, match_pitch=True)

    assert onset.f1 == 1.0
    assert pitch.true_positives == 1
    assert pitch.false_positives == 1
    assert pitch.false_negatives == 1


def test_score_audio_only_separates_pitch_from_tab_candidate_selection():
    gold = [TabEvent(0.0, 0.2, string_idx=3, fret=14, pitch_midi=69, confidence=1.0)]
    predicted_audio = [
        AudioEvent(0.01, 0.21, pitch_midi=69, velocity=0.8, confidence=0.9),
    ]

    scored = score_audio_only(predicted_audio, gold)

    assert scored.onset.f1 == 1.0
    assert scored.pitch.f1 == 1.0
    assert scored.tab.f1 == 0.0
    assert scored.decoded[0].string_idx == 5
    assert scored.decoded[0].fret == 5


def test_score_audio_only_can_apply_melodic_segment_prior():
    pitches = [47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 67, 69, 71, 72, 74]
    predicted_audio = [
        AudioEvent(
            onset_s=index * 0.2,
            offset_s=index * 0.2 + 0.1,
            pitch_midi=pitch,
            velocity=0.8,
            confidence=0.9,
        )
        for index, pitch in enumerate(pitches)
    ]
    gold = [
        TabEvent(
            onset_s=index * 0.2,
            duration_s=0.1,
            string_idx=string_idx,
            fret=fret,
            pitch_midi=pitch,
            confidence=1.0,
        )
        for index, (pitch, string_idx, fret) in enumerate(
            [
                (47, 0, 7),
                (48, 0, 8),
                (50, 0, 10),
                (52, 1, 7),
                (53, 1, 8),
                (55, 1, 10),
                (57, 2, 7),
                (59, 2, 9),
                (60, 2, 10),
                (62, 3, 7),
                (64, 3, 9),
                (67, 4, 8),
                (69, 4, 10),
                (71, 5, 7),
                (72, 5, 8),
                (74, 5, 10),
            ]
        )
    ]

    disabled = score_audio_only(predicted_audio, gold, melodic_prior_enabled=False)
    enabled = score_audio_only(predicted_audio, gold, melodic_prior_enabled=True)

    assert enabled.tab.f1 > disabled.tab.f1
    assert enabled.decoded[0].string_idx == 0
    assert enabled.decoded[0].fret == 7


def test_summarize_results_uses_all_micro_counts():
    result = TrackEvalResult(
        track_id="clip",
        backend="highres",
        gold_notes=3,
        audio_events=4,
        decoded_events=4,
        onset=EventF1Result(0.50, 0.67, 0.57, 2, 2, 1),
        pitch=EventF1Result(0.25, 0.33, 0.29, 1, 3, 2),
        tab=TabF1Result(0.25, 0.33, 0.29, 1, 3, 2),
    )

    summary = summarize_results([result], backend="highres", split="validation")

    assert summary.micro_onset.true_positives == 2
    assert summary.melodic_prior is False
    assert summary.micro_onset.false_positives == 2
    assert summary.micro_onset.false_negatives == 1
    assert summary.micro_onset.f1 == pytest.approx(4 / 7)
    assert summary.micro_tab.true_positives == 1
    assert summary.micro_tab.false_positives == 3
    assert summary.micro_tab.false_negatives == 2
    assert summary.micro_tab.f1 == pytest.approx(2 / 7)


def test_main_reports_backend_setup_blocker(monkeypatch: pytest.MonkeyPatch, capsys):
    import tabvision.eval.guitarset_audio as guitarset_audio

    def _raise_blocker(**_kwargs):
        raise BackendError("basic-pitch is not installed")

    monkeypatch.setattr(guitarset_audio, "run_eval", _raise_blocker)

    code = guitarset_audio.main(["--backend", "basicpitch", "--limit", "1"])

    assert code == 2
    assert "setup_blocker=basic-pitch is not installed" in capsys.readouterr().err


def test_run_eval_reuses_backend_across_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import tabvision.audio.backend as backend_registry
    import tabvision.eval.guitarset_audio as guitarset_audio

    ann = tmp_path / "annotation"
    audio = tmp_path / "audio_mono-mic"
    ann.mkdir()
    audio.mkdir()
    for track_id in ["05_one", "05_two"]:
        _write_jams(ann / f"{track_id}.jams")
        (audio / f"{track_id}_mic.wav").write_bytes(b"RIFF")

    class FakeBackend:
        name = "fake"

        def __init__(self) -> None:
            self.transcribe_calls = 0

        def transcribe(self, _wav, _sr, _session):
            self.transcribe_calls += 1
            return [
                AudioEvent(
                    onset_s=0.10,
                    offset_s=0.35,
                    pitch_midi=44,
                    velocity=1.0,
                    confidence=1.0,
                )
            ]

    fake = FakeBackend()
    make_calls = 0

    seen_kwargs = []

    def fake_make(name: str, **kwargs):
        nonlocal make_calls
        assert name == "fake"
        make_calls += 1
        seen_kwargs.append(kwargs)
        return fake

    monkeypatch.setattr(backend_registry, "make", fake_make)
    monkeypatch.setattr(
        guitarset_audio,
        "load_mono_audio",
        lambda _path: (np.zeros(8, dtype=np.float32), 8_000),
    )

    results, _summary = guitarset_audio.run_eval(
        backend_name="fake",
        data_home=tmp_path,
        split="validation",
        backend_kwargs={"device": "cuda"},
    )

    assert len(results) == 2
    assert make_calls == 1
    assert seen_kwargs == [{"device": "cuda"}]
    assert fake.transcribe_calls == 2
