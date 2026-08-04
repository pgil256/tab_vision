"""Frontend-neutral editor document emitted by the Python pipeline.

The desktop and web shells may edit this presentation model, but candidate
ranking remains a Python responsibility.  The shape intentionally mirrors the
stabilized web editor contract so both shells consume the same semantics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabvision.assist.candidates import compute_note_candidates
from tabvision.pipeline import PipelineArtifacts
from tabvision.types import GuitarConfig

_NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def build_editor_document(
    result: PipelineArtifacts,
    *,
    cfg: GuitarConfig,
    source_path: str | Path,
    video_enabled: bool,
) -> dict[str, Any]:
    """Return a JSON-serializable assisted-editing document."""

    events = sorted(result.tab_events, key=lambda event: event.onset_s)
    ranked = compute_note_candidates(
        events,
        result.audio_events,
        cfg=cfg,
        sequence_prior=result.policy.resolved_sequence_prior,
    )
    document_id = uuid.uuid4().hex
    notes: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        confidence = float(event.confidence)
        candidates = ranked[index] if index < len(ranked) else None
        note: dict[str, Any] = {
            "id": f"{document_id}-{index}",
            "timestamp": float(event.onset_s),
            "endTime": float(event.onset_s + max(0.0, event.duration_s)),
            "string": 6 - int(event.string_idx),
            "fret": int(event.fret),
            "confidence": confidence,
            "confidenceLevel": _confidence_level(confidence),
            "isEdited": False,
            "detectedMidiNote": int(event.pitch_midi),
        }
        if candidates:
            note["candidates"] = [
                {"string": 6 - int(candidate.string_idx), "fret": int(candidate.fret)}
                for candidate in candidates
            ]
        if event.techniques:
            note["technique"] = event.techniques[0]
        notes.append(note)

    artifacts = tuple(result.policy.artifacts)
    counts = {
        level: sum(note["confidenceLevel"] == level for note in notes)
        for level in ("high", "medium", "low")
    }
    beat_grid = getattr(result, "beat_grid", None)
    return {
        "id": document_id,
        "createdAt": datetime.now(UTC).isoformat(),
        "duration": max((note["endTime"] for note in notes), default=0.0) + 1.0,
        "capoFret": cfg.capo,
        "tuning": [_NOTE_NAMES[midi % 12] for midi in cfg.tuning_midi],
        "tuningMidi": list(cfg.tuning_midi),
        "sourcePath": str(Path(source_path).resolve()),
        "notes": notes,
        "metadata": {
            "totalNotes": len(notes),
            "highConfidenceNotes": counts["high"],
            "mediumConfidenceNotes": counts["medium"],
            "lowConfidenceNotes": counts["low"],
            "averageConfidence": (
                sum(note["confidence"] for note in notes) / len(notes) if notes else 0.0
            ),
            "pipelineVersion": "v1",
            "audioBackend": result.resolved_audio_backend,
            "requestedPositionPrior": result.policy.requested_position_prior,
            "resolvedPositionPrior": result.policy.resolved_position_prior,
            "requestedSequencePrior": result.policy.requested_sequence_prior,
            "resolvedSequencePrior": result.policy.resolved_sequence_prior,
            "requestedStringEvidence": result.policy.requested_string_evidence,
            "resolvedStringEvidence": result.policy.resolved_string_evidence,
            "artifactVersions": {item.name: item.version for item in artifacts},
            "artifactSha256": {item.name: item.sha256 for item in artifacts},
            "videoEnabled": video_enabled,
            "assistCandidateNotes": sum(bool(note.get("candidates")) for note in notes),
            **(
                {
                    "tempoBpm": float(beat_grid.tempo_bpm),
                    "beatTimes": [float(value) for value in beat_grid.beat_times],
                    "beatsPerBar": int(beat_grid.beats_per_bar),
                    "beatDetectionSource": beat_grid.source,
                }
                if beat_grid is not None
                else {}
            ),
            "diagnostics": {
                "resolvedVideoBackend": result.resolved_video_backend,
                "videoObservationCount": result.position_observation_count,
                "notesAffectedByVideo": result.notes_affected_by_video,
            },
        },
    }


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


__all__ = ["build_editor_document"]
