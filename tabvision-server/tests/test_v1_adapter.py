"""Tests for the v1 production transcription adapter."""
from __future__ import annotations

import json

import pytest

from app.models import Job
from app.v1_adapter import (
    V1PipelineConfig,
    humanize_pipeline_error,
    process_v1_job,
    run_v1_transcription,
    tab_events_to_tab_document,
)


class _RecordingStorage:
    """Capture every persisted (status, stage, progress) snapshot."""

    def __init__(self):
        self.history: list[tuple[str, str, float]] = []

    def save(self, job: Job) -> None:
        self.history.append((job.status, job.current_stage, job.progress))


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


def test_process_v1_job_persists_real_pipeline_stages(tmp_path):
    """The two hardcoded save_stage calls are replaced by run_pipeline's
    progress callback: each reported stage lands in storage with a
    monotonically rising progress, and video_enabled is stamped on the job."""
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=0)
    storage = _RecordingStorage()
    config = V1PipelineConfig(
        audio_backend="highres",
        position_prior="guitarset-v1",
        video_enabled=False,
        accuracy_mode="accurate",
    )

    def fake_runner(video_path, **kwargs):
        callback = kwargs["progress_callback"]
        for stage in ("demux", "model_load", "audio_inference", "decode"):
            callback(stage)
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

    process_v1_job(job, storage, str(tmp_path), config=config, pipeline_runner=fake_runner)

    assert job.status == "completed"
    assert job.video_enabled is False
    stages = [(stage, progress) for _status, stage, progress in storage.history]
    assert stages == [
        ("extracting_audio", 0.05),
        ("extracting_audio", 0.10),
        ("analyzing_audio", 0.20),
        ("analyzing_audio", 0.35),
        ("fusing", 0.80),
        ("saving", 0.9),
        ("complete", 1.0),
    ]
    progresses = [progress for _stage, progress in stages]
    assert progresses == sorted(progresses), "progress must never move backwards"


def test_process_v1_job_ignores_unknown_pipeline_stages(tmp_path):
    """Future/unknown stage names from run_pipeline must not corrupt the
    poll-visible state."""
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=0)
    storage = _RecordingStorage()
    config = V1PipelineConfig(video_enabled=False)

    def fake_runner(video_path, **kwargs):
        kwargs["progress_callback"]("some_future_stage")
        return []

    process_v1_job(job, storage, str(tmp_path), config=config, pipeline_runner=fake_runner)

    assert job.status == "completed"
    assert "some_future_stage" not in [stage for _s, stage, _p in storage.history]


def test_process_v1_job_maps_failure_to_short_message(tmp_path):
    """The client sees a one-line humane message; the traceback stays out of
    the job record (server logs only)."""
    job = Job.create(video_path="/tmp/example.mp4", capo_fret=0)
    storage = _RecordingStorage()
    config = V1PipelineConfig(video_enabled=False)

    def fake_runner(video_path, **kwargs):
        raise RuntimeError("ffmpeg not on PATH; required by tabvision.demux")

    process_v1_job(job, storage, str(tmp_path), config=config, pipeline_runner=fake_runner)

    assert job.status == "failed"
    assert "ffmpeg" in job.error_message
    assert "Traceback" not in job.error_message
    assert "tabvision.demux" not in job.error_message
    assert len(job.error_message) < 300


@pytest.mark.parametrize(
    ("exc", "needle"),
    [
        (RuntimeError("ffmpeg not on PATH; required by tabvision.demux"), "audio toolkit"),
        (RuntimeError("ffprobe not on PATH; required by tabvision.demux"), "audio toolkit"),
        (RuntimeError("ffmpeg returned empty audio stream"), "No audio"),
        (
            RuntimeError("ffmpeg audio decode failed: Output file #0 does not contain any stream"),
            "No audio",
        ),
        (RuntimeError("ffmpeg audio decode failed: Invalid data found"), "format or codec"),
        (RuntimeError("ffprobe failed: moov atom not found"), "format or codec"),
        (RuntimeError("video file not found: /tmp/gone.mp4"), "upload it again"),
        (FileNotFoundError("/tmp/gone.mp4"), "upload it again"),
        (
            ConnectionError("HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded"),
            "could not be downloaded",
        ),
        (RuntimeError("getaddrinfo failed"), "could not be downloaded"),
        (ValueError("v1 string_idx must be 0..5, got 9"), "Transcription failed"),
    ],
)
def test_humanize_pipeline_error_maps_known_failures(exc, needle):
    message = humanize_pipeline_error(exc)
    assert needle in message
    assert "\n" not in message, "client messages must be single-line"
    assert len(message) < 300


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
