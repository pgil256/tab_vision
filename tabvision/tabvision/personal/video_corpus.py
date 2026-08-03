"""A growing, local-only corpus of labelled frames from gold-tab sessions.

Each ingested session contributes JPEG frames sampled just after every
aligned note's onset, each carrying the gold tab's ``(string, fret)`` as
ground truth. This is exactly the artifact the string-resolution question
was blocked on ("no signal *at 640×360*; a labelled corpus at the user's
real resolution is the blocker") — built incidentally from practice takes
the user records anyway, in their own lighting, on their own rig.

Frames are sampled at small offsets *after* the onset because that is when
the fretting hand is guaranteed to be planted on the note; the offsets stay
well inside typical note durations. Rows are only written for frames that
actually existed within tolerance, so a corpus row never points at
interpolated or missing pixels.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabvision.fusion.personal_prior import PersonalLabel
from tabvision.personal.alignment import AlignedNote

CORPUS_SCHEMA_VERSION = 1

# Sampling constants. Offsets sit after the onset (hand planted) and inside
# typical note durations; the tolerance is 1.5 frame intervals at 30 fps so
# a wanted instant is only dropped when decoding genuinely skipped there.
DEFAULT_FRAME_OFFSETS_S = (0.04, 0.12, 0.20)
DEFAULT_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class CorpusIngestSummary:
    session_dir: Path
    notes: int
    frames_written: int
    rows_path: Path


def ingest_frames(
    frames: Iterable[tuple[float, np.ndarray]],
    matches: Iterable[AlignedNote],
    session_dir: str | Path,
    *,
    source_media: str,
    frame_offsets_s: tuple[float, ...] = DEFAULT_FRAME_OFFSETS_S,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> CorpusIngestSummary:
    """Single-pass extraction of labelled frames into ``session_dir``.

    The frame iterator is consumed once in timestamp order (the demuxer's
    contract). Each wanted instant — one per (aligned note, offset) — takes
    the nearest decoded frame within ``tolerance_s``; ties resolve to the
    earlier frame, and instants with no frame in range are dropped rather
    than approximated.
    """
    import cv2

    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    ordered_matches = sorted(matches, key=lambda match: match.onset_s)
    target = Path(session_dir)
    frames_dir = target / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    @dataclass
    class _Wanted:
        instant_s: float
        match_index: int
        offset_index: int
        best_gap: float = float("inf")
        best_timestamp: float = 0.0
        best_frame: np.ndarray | None = None

    wanted = sorted(
        (
            _Wanted(
                instant_s=match.onset_s + offset,
                match_index=match_index,
                offset_index=offset_index,
            )
            for match_index, match in enumerate(ordered_matches)
            for offset_index, offset in enumerate(frame_offsets_s)
        ),
        key=lambda item: item.instant_s,
    )

    rows_path = target / "rows.jsonl"
    frames_written = 0

    def _finalize(item: _Wanted, handle) -> int:
        if item.best_frame is None:
            return 0
        match = ordered_matches[item.match_index]
        name = f"note{item.match_index:05d}_f{item.offset_index}.jpg"
        if not cv2.imwrite(str(frames_dir / name), item.best_frame):
            raise OSError(f"failed to write frame {frames_dir / name}")
        handle.write(
            json.dumps(
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "frame": f"frames/{name}",
                    "string_idx": match.note.string_idx,
                    "fret": match.note.fret,
                    "pitch_midi": match.note.pitch_midi,
                    "onset_s": match.onset_s,
                    "frame_offset_s": frame_offsets_s[item.offset_index],
                    "frame_timestamp_s": item.best_timestamp,
                    "confidence": match.confidence,
                    "media": source_media,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 1

    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        pending_start = 0
        for timestamp_s, frame in frames:
            timestamp = float(timestamp_s)
            # Anything whose window closed before this frame can be flushed.
            while (
                pending_start < len(wanted)
                and wanted[pending_start].instant_s + tolerance_s < timestamp
            ):
                frames_written += _finalize(wanted[pending_start], handle)
                pending_start += 1
            for item in _iter_in_window(wanted, pending_start, timestamp, tolerance_s):
                gap = abs(timestamp - item.instant_s)
                if gap < item.best_gap:
                    item.best_gap = gap
                    item.best_timestamp = timestamp
                    item.best_frame = frame.copy()
        for item in wanted[pending_start:]:
            frames_written += _finalize(item, handle)

    (target / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "media": source_media,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": len(ordered_matches),
                "frames_written": frames_written,
                "frame_offsets_s": list(frame_offsets_s),
                "tolerance_s": tolerance_s,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CorpusIngestSummary(
        session_dir=target,
        notes=len(ordered_matches),
        frames_written=frames_written,
        rows_path=rows_path,
    )


def _iter_in_window(wanted: list, start: int, timestamp: float, tolerance_s: float) -> Iterator:
    for index in range(start, len(wanted)):
        if wanted[index].instant_s - tolerance_s > timestamp:
            break
        yield wanted[index]


def matches_to_personal_labels(matches: Iterable[AlignedNote]) -> list[PersonalLabel]:
    """Gold-tab matches as personal-prior labels (source ``gold-tab``).

    Strictly better than camera-window labels — every note, no camera in
    the loop — so they feed the same store and the same builder.
    """
    return [
        PersonalLabel(
            pitch_midi=match.note.pitch_midi,
            string_idx=match.note.string_idx,
            fret=match.note.fret,
            onset_s=match.onset_s,
            confidence=match.confidence,
            source="gold-tab",
        )
        for match in matches
    ]


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "DEFAULT_FRAME_OFFSETS_S",
    "DEFAULT_TOLERANCE_S",
    "CorpusIngestSummary",
    "ingest_frames",
    "matches_to_personal_labels",
]
