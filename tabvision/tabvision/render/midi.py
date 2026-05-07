"""MIDI renderer — Phase 6 deliverable.

Via mido. String assignment is surfaced as one MIDI channel per string:
channel 0 = low E through channel 5 = high E.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from tabvision.types import GuitarConfig, TabEvent

TICKS_PER_BEAT = 480
BEATS_PER_SECOND = 2.0  # 120 BPM; enough for deterministic interchange.


def render(events: Sequence[TabEvent], cfg: GuitarConfig | None = None) -> bytes:
    """Render events to a standard MIDI file."""
    if cfg is None:
        cfg = GuitarConfig()

    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "MIDI rendering requires mido. Install with: pip install '.[render]'"
        ) from exc

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="TabVision", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    messages: list[tuple[int, int, object]] = []
    for order, event in enumerate(sorted(events, key=lambda e: e.onset_s)):
        _validate_string(event, cfg)
        start_tick = _seconds_to_ticks(event.onset_s)
        end_tick = _seconds_to_ticks(event.onset_s + max(event.duration_s, 0.05))
        channel = event.string_idx
        velocity = _velocity(event.confidence)
        messages.append(
            (
                start_tick,
                order * 2 + 1,
                mido.Message(
                    "note_on",
                    note=event.pitch_midi,
                    velocity=velocity,
                    channel=channel,
                    time=0,
                ),
            )
        )
        messages.append(
            (
                end_tick,
                order * 2,
                mido.Message(
                    "note_off",
                    note=event.pitch_midi,
                    velocity=0,
                    channel=channel,
                    time=0,
                ),
            )
        )

    last_tick = 0
    for absolute_tick, _order, message in sorted(messages, key=lambda item: (item[0], item[1])):
        message.time = max(0, absolute_tick - last_tick)
        track.append(message)
        last_tick = absolute_tick
    track.append(mido.MetaMessage("end_of_track", time=0))

    out = BytesIO()
    mid.save(file=out)
    return out.getvalue()


def _seconds_to_ticks(seconds: float) -> int:
    return int(round(max(0.0, seconds) * BEATS_PER_SECOND * TICKS_PER_BEAT))


def _velocity(confidence: float) -> int:
    return max(1, min(127, int(round(confidence * 127))))


def _validate_string(event: TabEvent, cfg: GuitarConfig) -> None:
    if not 0 <= event.string_idx < cfg.n_strings:
        raise ValueError(f"string_idx out of range: {event.string_idx}")


__all__ = ["render"]
