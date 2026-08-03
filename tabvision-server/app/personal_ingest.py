"""Local-only gold-session banking for the studio loop.

Enabled only when ``TABVISION_PERSONAL_ROOT`` is set — ``studio.ps1``
sets it for the local backend; the Modal production deployment never
does, so in prod the endpoint answers 404 and none of this imports.

The flow: the user corrected a transcription in the web editor, so the
corrected notes are ground truth for the take the job was created from.
They become (a) labelled JPEG frames in the local video-training corpus
and (b) rows in the personal-prior label store — the same two assets the
CLI gold-tab ingest builds, sourced from the app the user already runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

AUDIO_ONLY_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def personal_root() -> Path | None:
    value = os.environ.get("TABVISION_PERSONAL_ROOT", "").strip()
    return Path(value) if value else None


def bank_gold_session(
    job_id: str,
    video_path: str,
    notes: list[dict[str, Any]],
    *,
    root: Path,
    bank_prior: bool = True,
) -> dict[str, Any]:
    """Ingest one corrected transcription; returns a JSON-ready summary."""
    from app.v1_adapter import _ensure_v1_on_path

    _ensure_v1_on_path()
    from tabvision.personal.corrections import (
        CORRECTION_SOURCE,
        correction_notes_to_matches,
    )
    from tabvision.personal.video_corpus import (
        ingest_frames,
        matches_to_personal_labels,
    )

    matches = correction_notes_to_matches(notes)

    frames_written = 0
    session_dir = root / "video_corpus" / f"job-{job_id}"
    if Path(video_path).suffix.lower() in AUDIO_ONLY_EXTENSIONS:
        # An audio-only take has no frames to harvest; the labels still count.
        session_dir = None
    else:
        from tabvision.demux import demux

        demuxed = demux(video_path)
        try:
            summary = ingest_frames(
                demuxed.frame_iterator,
                matches,
                session_dir,
                source_media=video_path,
            )
        finally:
            close = getattr(demuxed.frame_iterator, "close", None)
            if callable(close):
                close()
        frames_written = summary.frames_written

    prior_labels = 0
    if bank_prior:
        from tabvision.fusion.personal_prior import append_personal_labels

        labels = matches_to_personal_labels(matches, source=CORRECTION_SOURCE)
        append_personal_labels(
            root / "labels.jsonl",
            labels,
            source_media=video_path,
        )
        prior_labels = len(labels)

    return {
        "notes": len(matches),
        "frames_written": frames_written,
        "session_dir": str(session_dir) if session_dir else None,
        "prior_labels": prior_labels,
        "prior_store": str(root / "labels.jsonl") if bank_prior else None,
    }
