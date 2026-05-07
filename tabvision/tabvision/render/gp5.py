"""GuitarPro 5 renderer — Phase 6 deliverable.

Via PyGuitarPro. LGPL license — see LICENSES.md verification gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from tabvision.types import GuitarConfig, TabEvent


def render(events: Sequence[TabEvent], cfg: GuitarConfig | None = None) -> bytes:
    """Render events to Guitar Pro 5 bytes."""
    if cfg is None:
        cfg = GuitarConfig()

    try:
        import guitarpro
        from guitarpro import models
    except ImportError as exc:
        raise RuntimeError(
            "GP5 rendering requires PyGuitarPro. Install with: pip install '.[render]'"
        ) from exc

    song = models.Song(title="TabVision Transcription")
    song.tempo = 120
    track = song.tracks[0]
    track.name = "TabVision"
    track.fretCount = cfg.max_fret
    track.strings = [
        models.GuitarString(number=i + 1, value=cfg.tuning_midi[cfg.n_strings - 1 - i])
        for i in range(cfg.n_strings)
    ]

    voice = track.measures[0].voices[0]
    voice.beats.clear()
    for event in sorted(events, key=lambda e: e.onset_s):
        _validate_event(event, cfg)
        beat = models.Beat(voice=voice)
        beat.status = models.BeatStatus.normal
        beat.duration = _duration_for_event(event, models)
        beat.start = _gp_start(event.onset_s)
        note = models.Note(
            beat=beat,
            value=event.fret,
            velocity=_velocity(event.confidence),
            string=cfg.n_strings - event.string_idx,
            type=models.NoteType.normal,
        )
        beat.notes.append(note)
        if event.confidence < 0.5:
            beat.text = f"low confidence {event.confidence:.2f}"
        voice.beats.append(beat)

    out = BytesIO()
    guitarpro.write(song, out, version=(5, 1, 0))
    return out.getvalue()


def _duration_for_event(event: TabEvent, models):
    if event.duration_s >= 0.75:
        return models.Duration(value=2)
    if event.duration_s >= 0.35:
        return models.Duration(value=4)
    return models.Duration(value=8)


def _gp_start(onset_s: float) -> int:
    return int(round(960 + max(0.0, onset_s) * 960))


def _velocity(confidence: float) -> int:
    return max(1, min(127, int(round(confidence * 127))))


def _validate_event(event: TabEvent, cfg: GuitarConfig) -> None:
    if not 0 <= event.string_idx < cfg.n_strings:
        raise ValueError(f"string_idx out of range: {event.string_idx}")
    if not 0 <= event.fret <= cfg.max_fret:
        raise ValueError(f"fret out of range: {event.fret}")


__all__ = ["render"]
