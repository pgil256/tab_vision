"""Guitar-TECHS 6-track MIDI annotation parser.

Per arXiv:2501.03720 §3, Guitar-TECHS distributes one MIDI file per
clip with six instrument tracks, each carrying the notes for one
guitar string. The default ordering is low E → high E, matching the
:class:`tabvision.types.GuitarConfig` ``tuning_midi`` convention
(low E = ``string_idx`` 0).

If a particular Guitar-TECHS release uses a different track ordering,
pass ``track_to_string`` to ``parse`` directly; manifest-level support
for parser arguments is deferred to a later phase.
"""

from __future__ import annotations

from pathlib import Path

from tabvision.eval.parsers.registry import register_parser
from tabvision.types import GuitarConfig, TabEvent

FORMAT_NAME = "guitar_techs_midi"

DEFAULT_TRACK_TO_STRING: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
"""Track-index → ``string_idx`` mapping; default = identity (low E first)."""


def parse(
    midi_path: str | Path,
    cfg: GuitarConfig | None = None,
    *,
    track_to_string: tuple[int, ...] = DEFAULT_TRACK_TO_STRING,
) -> list[TabEvent]:
    """Parse Guitar-TECHS MIDI into v1 :class:`TabEvent` gold notes.

    Pitch ``p`` on the track mapped to string ``s`` is assigned
    ``fret = p - cfg.tuning_midi[s]``. Notes that would imply a fret
    below ``cfg.capo`` or above ``cfg.max_fret`` are dropped.
    """
    try:
        import pretty_midi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - skip path
        raise ImportError(
            "guitar_techs_midi parser requires pretty_midi. Install with: "
            "pip install -e 'tabvision[audio-highres]'"
        ) from exc

    if cfg is None:
        cfg = GuitarConfig()

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    out: list[TabEvent] = []
    for track_index, instrument in enumerate(midi.instruments):
        if track_index >= len(track_to_string):
            break
        string_idx = track_to_string[track_index]
        if not 0 <= string_idx < cfg.n_strings:
            continue

        open_pitch = cfg.tuning_midi[string_idx]
        for note in instrument.notes:
            pitch_midi = int(note.pitch)
            fret = pitch_midi - open_pitch
            if fret < cfg.capo or fret > cfg.max_fret:
                continue
            out.append(
                TabEvent(
                    onset_s=float(note.start),
                    duration_s=float(max(0.0, note.end - note.start)),
                    string_idx=string_idx,
                    fret=fret,
                    pitch_midi=pitch_midi,
                    confidence=1.0,
                )
            )

    out.sort(key=lambda ev: (ev.onset_s, ev.string_idx, ev.fret))
    return out


register_parser(FORMAT_NAME, parse)


__all__ = ["DEFAULT_TRACK_TO_STRING", "FORMAT_NAME", "parse"]
