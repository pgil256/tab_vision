"""MusicXML renderer — Phase 6 deliverable.

Via music21.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from tabvision.types import GuitarConfig, TabEvent

QUARTERS_PER_SECOND = 2.0
MUSICXML_GRID_DIVISIONS = 64


def render(events: Sequence[TabEvent], cfg: GuitarConfig | None = None) -> bytes:
    """Render events to MusicXML bytes."""
    if cfg is None:
        cfg = GuitarConfig()

    try:
        from music21 import instrument, metadata, note, stream, tempo
        from music21.musicxml.m21ToXml import GeneralObjectExporter
    except ImportError as exc:
        raise RuntimeError(
            "MusicXML rendering requires music21. Install with: pip install '.[render]'"
        ) from exc

    score = stream.Score(id="tabvision")
    score.metadata = metadata.Metadata(title="TabVision Transcription")
    part = stream.Part(id="guitar")
    part.insert(0, instrument.Guitar())
    part.insert(0, tempo.MetronomeMark(number=120))

    for event in sorted(events, key=lambda e: e.onset_s):
        _validate_string(event, cfg)
        n = note.Note(event.pitch_midi)
        onset_quarters = Fraction(
            round(max(0.0, event.onset_s) * QUARTERS_PER_SECOND * MUSICXML_GRID_DIVISIONS),
            MUSICXML_GRID_DIVISIONS,
        )
        duration_quarters = Fraction(
            max(
                MUSICXML_GRID_DIVISIONS // 4,
                round(event.duration_s * QUARTERS_PER_SECOND * MUSICXML_GRID_DIVISIONS),
            ),
            MUSICXML_GRID_DIVISIONS,
        )
        n.quarterLength = duration_quarters
        n.volume.velocityScalar = max(0.0, min(1.0, event.confidence))
        n.addLyric(f"s{event.string_idx + 1} f{event.fret} c{event.confidence:.2f}")
        part.insert(onset_quarters, n)

    score.insert(0, part)
    payload = GeneralObjectExporter(score).parse()
    if isinstance(payload, bytes):
        return payload
    return str(payload).encode("utf-8")


def _validate_string(event: TabEvent, cfg: GuitarConfig) -> None:
    if not 0 <= event.string_idx < cfg.n_strings:
        raise ValueError(f"string_idx out of range: {event.string_idx}")


__all__ = ["render"]
