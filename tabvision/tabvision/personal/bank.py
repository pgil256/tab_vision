"""Local corrected-session banking shared by desktop and browser shells."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tabvision.errors import InvalidInputError
from tabvision.fusion.personal_prior import append_personal_labels
from tabvision.personal.corrections import CORRECTION_SOURCE, correction_notes_to_matches
from tabvision.personal.video_corpus import ingest_frames, matches_to_personal_labels

AUDIO_ONLY_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}
STANDARD_TUNING_MIDI = [40, 45, 50, 55, 59, 64]
STANDARD_TUNING_NAMES = ["E", "A", "D", "G", "B", "E"]


def bank_corrected_document(
    source_path: str | Path,
    document_path: str | Path,
    *,
    root: str | Path,
    bank_prior: bool = True,
) -> dict[str, Any]:
    """Bank one editor document as local frame and position-prior truth."""
    source = Path(source_path).resolve()
    document_file = Path(document_path).resolve()
    target_root = Path(root).resolve()
    if not source.is_file():
        raise InvalidInputError(f"source recording not found: {source}")
    if not document_file.is_file():
        raise InvalidInputError(f"editor document not found: {document_file}")
    try:
        document = json.loads(document_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidInputError(f"could not read editor document: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("notes"), list):
        raise InvalidInputError("editor document must contain a notes list")
    if int(document.get("capoFret", 0)) != 0:
        raise InvalidInputError("gold sessions require capo 0 (stores are capo-0 indexed)")
    tuning_midi = document.get("tuningMidi")
    tuning_names = document.get("tuning")
    if tuning_midi not in (None, [], STANDARD_TUNING_MIDI) or (
        tuning_midi in (None, []) and tuning_names not in (None, [], STANDARD_TUNING_NAMES)
    ):
        raise InvalidInputError(
            "gold sessions require standard tuning (stores are standard-tuning indexed)"
        )
    try:
        matches = correction_notes_to_matches(document["notes"])
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    target_root.mkdir(parents=True, exist_ok=True)
    raw_id = str(document.get("id") or document_file.stem)
    session_id = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_id).strip("-.") or "session"
    frames_written = 0
    session_dir: Path | None = target_root / "video_corpus" / f"desktop-{session_id}"
    if source.suffix.lower() in AUDIO_ONLY_EXTENSIONS:
        session_dir = None
    else:
        from tabvision.demux import demux

        demuxed = demux(source)
        try:
            summary = ingest_frames(
                demuxed.frame_iterator,
                matches,
                session_dir,
                source_media=str(source),
            )
        finally:
            close = getattr(demuxed.frame_iterator, "close", None)
            if callable(close):
                close()
        frames_written = summary.frames_written

    prior_labels = 0
    if bank_prior:
        labels = matches_to_personal_labels(matches, source=CORRECTION_SOURCE)
        append_personal_labels(
            target_root / "labels.jsonl",
            labels,
            source_media=str(source),
        )
        prior_labels = len(labels)
    return {
        "notes": len(matches),
        "frames_written": frames_written,
        "session_dir": str(session_dir) if session_dir else None,
        "prior_labels": prior_labels,
        "prior_store": str(target_root / "labels.jsonl") if bank_prior else None,
    }


__all__ = ["AUDIO_ONLY_EXTENSIONS", "bank_corrected_document"]
