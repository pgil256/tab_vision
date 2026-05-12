"""Tests for the v1 production transcription adapter."""
from __future__ import annotations

import json

import pytest

from app.models import Job
from app.v1_adapter import (
    V1PipelineConfig,
    run_v1_transcription,
    tab_events_to_tab_document,
)


class _TabEvent:
    def __init__(
        self,
        *,
        onset_s,
        duration_s,
        string_idx,
        fret,
        pitch_midi,
        confidence,
        techniques=(),
    ):
        self.onset_s = onset_s
        self.duration_s = duration_s
        self.string_idx = string_idx
        self.fret = fret
        self.pitch_midi = pitch_midi
        self.confidence = confidence
        self.techniques = techniques


def test_tab_events_convert_to_frontend_tab_document_with_string_mapping():
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=2)
    events = [
        _TabEvent(
            onset_s=1.0,
            duration_s=0.5,
            string_idx=0,
            fret=5,
            pitch_midi=45,
            confidence=0.92,
        ),
        _TabEvent(
            onset_s=2.0,
            duration_s=0.25,
            string_idx=5,
            fret=7,
            pitch_midi=71,
            confidence=0.64,
            techniques=("hammer_on",),
        ),
    ]
    config = V1PipelineConfig(
        audio_backend="highres",
        position_prior="guitarset-v1",
        video_enabled=False,
        melodic_prior_enabled=False,
        accuracy_mode="accurate",
    )

    document = tab_events_to_tab_document(
        job,
        events,
        config,
        diagnostics={"fallbackUsed": False},
    )

    assert document["capoFret"] == 2
    assert document["notes"][0]["string"] == 6  # v1 low E idx 0 -> frontend string 6
    assert document["notes"][1]["string"] == 1  # v1 high E idx 5 -> frontend string 1
    assert document["notes"][1]["endTime"] == 2.25
    assert document["notes"][1]["technique"] == "hammer_on"
    assert document["metadata"]["pipelineVersion"] == "v1"
    assert document["metadata"]["audioBackend"] == "highres"
    assert document["metadata"]["positionPrior"] == "guitarset-v1"
    assert document["metadata"]["videoEnabled"] is False
    assert document["metadata"]["diagnostics"]["fallbackUsed"] is False


def test_v1_transcription_saves_frontend_document_and_uses_context(tmp_path):
    job = Job.create(
        video_path="/tmp/example.mp4",
        capo_fret=1,
        instrument="classical",
        tone="clean",
        style="fingerstyle",
        accuracy_mode="accurate",
    )
    config = V1PipelineConfig(
        audio_backend="highres",
        position_prior="guitarset-v1",
        video_enabled=False,
        accuracy_mode="accurate",
    )
    captured = {}

    def fake_runner(video_path, **kwargs):
        captured.update(kwargs)
        return [
            _TabEvent(
                onset_s=0.0,
                duration_s=0.4,
                string_idx=4,
                fret=3,
                pitch_midi=62,
                confidence=0.8,
            )
        ]

    result_path = run_v1_transcription(
        job,
        str(tmp_path),
        config=config,
        pipeline_runner=fake_runner,
    )

    with open(result_path) as f:
        document = json.load(f)

    assert document["notes"][0]["string"] == 2
    assert document["metadata"]["totalNotes"] == 1
    assert captured["audio_backend_name"] == "highres"
    assert captured["position_prior"] == "guitarset-v1"
    assert captured["video_enabled"] is False
    assert captured["melodic_prior_enabled"] is False
    assert captured["cfg"].capo == 1
    assert captured["session"].instrument == "classical"
    assert captured["session"].style == "fingerstyle"


def test_v1_config_defaults_do_not_enable_basicpitch_fallback(monkeypatch):
    monkeypatch.delenv("TABVISION_FALLBACK_AUDIO_BACKEND", raising=False)

    assert V1PipelineConfig().fallback_audio_backend is None
    assert V1PipelineConfig.from_env().fallback_audio_backend is None


def test_v1_transcription_does_not_fallback_without_opt_in(tmp_path):
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=0)
    config = V1PipelineConfig(
        audio_backend="highres",
        position_prior="guitarset-v1",
        video_enabled=False,
        accuracy_mode="accurate",
    )
    calls = []

    def fake_runner(video_path, **kwargs):
        calls.append(kwargs["audio_backend_name"])
        raise RuntimeError(f"{kwargs['audio_backend_name']} unavailable")

    with pytest.raises(RuntimeError, match="highres unavailable"):
        run_v1_transcription(
            job,
            str(tmp_path),
            config=config,
            pipeline_runner=fake_runner,
        )

    assert calls == ["highres"]


def test_v1_transcription_falls_back_to_basicpitch_when_highres_fails(tmp_path):
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=0)
    config = V1PipelineConfig(
        audio_backend="highres",
        fallback_audio_backend="basicpitch",
        position_prior="guitarset-v1",
        video_enabled=False,
        accuracy_mode="accurate",
    )
    calls = []

    def fake_runner(video_path, **kwargs):
        calls.append(kwargs["audio_backend_name"])
        if kwargs["audio_backend_name"] == "highres":
            raise RuntimeError("highres unavailable")
        return [
            _TabEvent(
                onset_s=0.0,
                duration_s=0.2,
                string_idx=5,
                fret=0,
                pitch_midi=64,
                confidence=0.7,
            )
        ]

    result_path = run_v1_transcription(
        job,
        str(tmp_path),
        config=config,
        pipeline_runner=fake_runner,
    )

    with open(result_path) as f:
        document = json.load(f)

    assert calls == ["highres", "basicpitch"]
    assert document["metadata"]["audioBackend"] == "basicpitch"
    assert document["metadata"]["diagnostics"]["fallbackUsed"] is True
    assert "highres unavailable" in document["metadata"]["diagnostics"]["fallbackReason"]
