"""Adapter from the v1 ``tabvision`` package to the Flask API contract."""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from app.models import Job

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class V1PipelineConfig:
    """Runtime-selectable v1 pipeline settings."""

    audio_backend: str = "highres"
    fallback_audio_backend: str | None = None
    position_prior: str | None = "auto"
    sequence_prior: str | None = "auto"
    string_evidence: str | None = "auto"
    video_enabled: bool = False
    melodic_prior_enabled: bool = False
    accuracy_mode: str = "accurate"

    @classmethod
    def from_env(cls) -> "V1PipelineConfig":
        return cls(
            audio_backend=os.getenv("TABVISION_AUDIO_BACKEND", "highres").strip().lower(),
            fallback_audio_backend=_optional_env(
                "TABVISION_FALLBACK_AUDIO_BACKEND", None
            ),
            position_prior=_policy_env("TABVISION_POSITION_PRIOR", "auto"),
            sequence_prior=_policy_env("TABVISION_SEQUENCE_PRIOR", "auto"),
            string_evidence=_policy_env("TABVISION_STRING_EVIDENCE", "auto"),
            video_enabled=_truthy(os.getenv("TABVISION_VIDEO_ENABLED", "false")),
            melodic_prior_enabled=_truthy(os.getenv("TABVISION_MELODIC_PRIOR_ENABLED", "false")),
            accuracy_mode=os.getenv("TABVISION_ACCURACY_MODE", "accurate").strip().lower(),
        )


def _optional_env(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() == "none":
        return None
    return value


def _policy_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() or default


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_v1_on_path() -> None:
    """Make the sibling v1 package importable in local development."""
    repo_root = Path(__file__).resolve().parents[2]
    local_package_root = repo_root / "tabvision"
    if local_package_root.exists():
        sys.path.insert(0, str(local_package_root))


def _load_v1_runner() -> Callable[..., Any]:
    _ensure_v1_on_path()
    from tabvision.pipeline import run_pipeline_with_artifacts

    return run_pipeline_with_artifacts


def _load_v1_types():
    _ensure_v1_on_path()
    from tabvision.types import GuitarConfig, SessionConfig

    return GuitarConfig, SessionConfig


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _video_diagnostics(video_enabled: bool) -> dict[str, Any]:
    return {
        "fretboardDetectionConfidence": None,
        "handDetectionRate": 0.0,
        "videoObservationCount": 0,
        "notesAffectedByVideo": 0,
        "videoIgnoredByQualityGate": not video_enabled,
    }


def _fallback_policy_metadata(config: V1PipelineConfig) -> dict[str, Any]:
    """Metadata for legacy/injected runners that still return a bare list."""

    requested_position = config.position_prior if config.position_prior is not None else "none"
    requested_sequence = config.sequence_prior if config.sequence_prior is not None else "none"
    requested_evidence = config.string_evidence if config.string_evidence is not None else "none"
    return {
        "requestedPositionPrior": requested_position,
        "resolvedPositionPrior": requested_position,
        "requestedSequencePrior": requested_sequence,
        "resolvedSequencePrior": requested_sequence,
        "requestedStringEvidence": requested_evidence,
        "resolvedStringEvidence": requested_evidence,
        "artifactVersions": {},
        "artifactSha256": {},
    }


def _unpack_pipeline_result(
    result: Any,
    config: V1PipelineConfig,
) -> tuple[Iterable[Any], dict[str, Any]]:
    """Accept both the additive detailed result and the legacy bare event list."""

    if not hasattr(result, "tab_events") or not hasattr(result, "policy"):
        return result, _fallback_policy_metadata(config)
    policy = result.policy
    artifacts = tuple(getattr(policy, "artifacts", ()))
    metadata = {
        "requestedPositionPrior": policy.requested_position_prior,
        "resolvedPositionPrior": policy.resolved_position_prior,
        "requestedSequencePrior": policy.requested_sequence_prior,
        "resolvedSequencePrior": policy.resolved_sequence_prior,
        "requestedStringEvidence": policy.requested_string_evidence,
        "resolvedStringEvidence": policy.resolved_string_evidence,
        "artifactVersions": {item.name: item.version for item in artifacts},
        "artifactSha256": {item.name: item.sha256 for item in artifacts},
    }
    return result.tab_events, metadata


def tab_events_to_tab_document(
    job: Job,
    events: Iterable[Any],
    config: V1PipelineConfig,
    *,
    diagnostics: dict[str, Any] | None = None,
    inference_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert v1 TabEvents to the existing frontend TabDocument JSON."""
    event_list = sorted(list(events), key=lambda event: event.onset_s)
    notes_data: list[dict[str, Any]] = []

    for index, event in enumerate(event_list):
        string_idx = int(event.string_idx)
        if string_idx < 0 or string_idx > 5:
            raise ValueError(f"v1 string_idx must be 0..5, got {string_idx}")

        confidence = float(event.confidence)
        duration = max(0.0, float(event.duration_s))
        note = {
            "id": f"{job.id}-v1-{index}-{uuid.uuid4().hex[:8]}",
            "timestamp": float(event.onset_s),
            "endTime": float(event.onset_s) + duration,
            "string": 6 - string_idx,
            "fret": int(event.fret),
            "confidence": confidence,
            "confidenceLevel": _confidence_level(confidence),
            "isEdited": False,
        }
        techniques = tuple(getattr(event, "techniques", ()) or ())
        if techniques:
            note["technique"] = techniques[0]
        notes_data.append(note)

    total_notes = len(notes_data)
    high_conf = sum(1 for note in notes_data if note["confidenceLevel"] == "high")
    med_conf = sum(1 for note in notes_data if note["confidenceLevel"] == "medium")
    low_conf = sum(1 for note in notes_data if note["confidenceLevel"] == "low")
    max_time = max((note.get("endTime", note["timestamp"]) for note in notes_data), default=0.0)

    merged_diagnostics = _video_diagnostics(config.video_enabled)
    if diagnostics:
        merged_diagnostics.update(diagnostics)

    policy = inference_policy or _fallback_policy_metadata(config)
    return {
        "id": job.id,
        "createdAt": job.created_at.isoformat(),
        "duration": max_time + 1,
        "capoFret": job.capo_fret,
        "tuning": ["E", "B", "G", "D", "A", "E"],
        "notes": notes_data,
        "metadata": {
            "totalNotes": total_notes,
            "highConfidenceNotes": high_conf,
            "mediumConfidenceNotes": med_conf,
            "lowConfidenceNotes": low_conf,
            "videoConfirmedNotes": merged_diagnostics["notesAffectedByVideo"],
            "averageConfidence": (
                sum(note["confidence"] for note in notes_data) / total_notes
                if total_notes > 0 else 0
            ),
            "pipelineVersion": "v1",
            "audioBackend": config.audio_backend,
            "positionPrior": policy["resolvedPositionPrior"],
            **policy,
            "videoEnabled": config.video_enabled,
            "accuracyMode": config.accuracy_mode,
            "noteCountRatio": None,
            "diagnostics": merged_diagnostics,
        },
    }


def run_v1_transcription(
    job: Job,
    output_dir: str,
    *,
    config: V1PipelineConfig | None = None,
    pipeline_runner: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Run v1 transcription and write a frontend-compatible result JSON."""
    config = replace(config or V1PipelineConfig.from_env(), accuracy_mode=job.accuracy_mode)
    runner = pipeline_runner or _load_v1_runner()
    GuitarConfig, SessionConfig = _load_v1_types()

    diagnostics: dict[str, Any] = {
        "fallbackUsed": False,
        "requestedAudioBackend": config.audio_backend,
        "positionSweep": None,
    }

    common_kwargs = {
        "position_prior": config.position_prior if config.position_prior is not None else "none",
        "sequence_prior": config.sequence_prior if config.sequence_prior is not None else "none",
        "string_evidence": config.string_evidence if config.string_evidence is not None else "none",
        "video_enabled": config.video_enabled,
        "melodic_prior_enabled": config.melodic_prior_enabled,
        "lambda_vision": 1.0 if config.video_enabled else 0.0,
        "cfg": GuitarConfig(capo=job.capo_fret),
        "session": SessionConfig(
            instrument=job.instrument,
            tone=job.tone,
            style=job.style,
        ),
    }
    if progress_callback is not None:
        common_kwargs["progress_callback"] = progress_callback

    effective_config = config
    try:
        pipeline_result = runner(
            job.video_path,
            audio_backend_name=config.audio_backend,
            **common_kwargs,
        )
    except Exception as exc:
        fallback = config.fallback_audio_backend
        if not fallback or fallback == config.audio_backend:
            raise

        diagnostics["fallbackUsed"] = True
        diagnostics["fallbackReason"] = str(exc)
        pipeline_result = runner(
            job.video_path,
            audio_backend_name=fallback,
            **common_kwargs,
        )
        effective_config = replace(config, audio_backend=fallback)

    events, inference_policy = _unpack_pipeline_result(pipeline_result, effective_config)

    tab_document = tab_events_to_tab_document(
        job,
        events,
        effective_config,
        diagnostics=diagnostics,
        inference_policy=inference_policy,
    )

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{job.id}_result.json")
    with open(result_path, "w") as f:
        json.dump(tab_document, f, indent=2)

    return result_path


# run_pipeline stage name → (client stage key, progress). Stage keys match the
# web client's PIPELINE_STAGES checklist. model_load and audio_inference both
# read as "analyzing audio" to the user; the rising progress distinguishes them.
_PIPELINE_STAGE_MAP: dict[str, tuple[str, float]] = {
    "demux": ("extracting_audio", 0.10),
    "model_load": ("analyzing_audio", 0.20),
    "audio_inference": ("analyzing_audio", 0.35),
    "video_analysis": ("analyzing_video", 0.60),
    "decode": ("fusing", 0.80),
}


def humanize_pipeline_error(exc: BaseException) -> str:
    """Map a pipeline failure to a short, actionable message for the client.

    The full exception + traceback stay in the server logs; the client only
    ever sees these one-liners.
    """
    low = str(exc).lower()
    if "ffmpeg not on path" in low or "ffprobe not on path" in low:
        return (
            "The server is missing its audio toolkit (ffmpeg), so no uploads can "
            "be processed right now. This needs an operator fix — not a different file."
        )
    if "empty audio stream" in low or "does not contain any stream" in low:
        return (
            "No audio could be read from the file. Make sure the recording has an "
            "audio track (was the microphone enabled?) and try again."
        )
    if "audio decode failed" in low or "ffprobe failed" in low or "invalid data found" in low:
        return (
            "The file could not be decoded — its format or codec isn't supported. "
            "Try MP4, MOV, WEBM, WAV, MP3, or M4A."
        )
    if "file not found" in low or isinstance(exc, FileNotFoundError):
        return "The uploaded file went missing on the server. Please upload it again."
    if _looks_like_model_download_failure(low):
        return (
            "The transcription model could not be downloaded. Check the server's "
            "internet connection and try again in a few minutes."
        )
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return f"Transcription failed: {first_line[:200]}"


def _looks_like_model_download_failure(low: str) -> bool:
    # Nothing else in a transcription job touches the network, so network-ish
    # failures mean the checkpoint download (Hugging Face Hub) broke.
    markers = (
        "huggingface",
        "hf.co",
        "connection error",
        "connectionerror",
        "connection refused",
        "getaddrinfo",
        "name resolution",
        "timed out",
        "max retries exceeded",
        "download",
    )
    return any(marker in low for marker in markers)


def process_v1_job(
    job: Job,
    storage,
    output_dir: str,
    *,
    config: V1PipelineConfig | None = None,
    result_saved_hook=None,
    pipeline_runner: Callable[..., Any] | None = None,
) -> None:
    """Process a job with v1 while persisting each poll-visible state."""
    config = config or V1PipelineConfig.from_env()

    def save_stage(stage: str, progress: float, *, status: str = "processing") -> None:
        job.status = status
        job.current_stage = stage
        job.progress = progress
        job.updated_at = datetime.now(timezone.utc)
        storage.save(job)

    def on_pipeline_stage(stage: str) -> None:
        mapped = _PIPELINE_STAGE_MAP.get(stage)
        if mapped is not None:
            save_stage(*mapped)

    try:
        # Tell pollers which stages to expect before the pipeline starts.
        job.video_enabled = config.video_enabled
        save_stage("extracting_audio", 0.05)
        result_path = run_v1_transcription(
            job,
            output_dir,
            config=config,
            pipeline_runner=pipeline_runner,
            progress_callback=on_pipeline_stage,
        )

        save_stage("saving", 0.9)
        if result_saved_hook:
            result_saved_hook()
        job.result_path = result_path
        job.status = "completed"
        job.current_stage = "complete"
        job.progress = 1.0
        job.updated_at = datetime.now(timezone.utc)
        storage.save(job)
    except Exception as exc:
        # Full traceback to the server logs only; the client gets a short,
        # humane message instead of a wall of Python.
        logger.exception("v1 job %s failed", job.id)
        job.status = "failed"
        job.error_message = humanize_pipeline_error(exc)
        job.updated_at = datetime.now(timezone.utc)
        storage.save(job)
