"""Spotify Basic Pitch backend — Phase 1 baseline.

Wraps Basic Pitch's ``predict`` behind the SPEC.md §8 ``AudioBackend``
protocol. Emits raw note events as ``AudioEvent`` instances; downstream
filtering / fusion is the responsibility of other modules.

Phase 1 deliverable. Apache-2.0 license; see LICENSES.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from tabvision.errors import BackendError, InvalidInputError
from tabvision.types import AudioEvent, SessionConfig

# Guitar-relevant pitch range (E2 = 40 .. E6 = 88 for 24 frets on high E).
GUITAR_MIDI_MIN = 40
GUITAR_MIDI_MAX = 88
GUITAR_MIN_HZ = 80.0
GUITAR_MAX_HZ = 1400.0

# Basic Pitch thresholds tuned for guitar in v0; reuse as Phase 1 default.
DEFAULT_ONSET_THRESHOLD = 0.4
DEFAULT_FRAME_THRESHOLD = 0.25
DEFAULT_MIN_NOTE_LENGTH_MS = 30


class BasicPitchBackend:
    """Audio backend wrapping Spotify Basic Pitch.

    Implements the ``AudioBackend`` protocol from ``tabvision.types``.
    """

    name = "basicpitch"

    def __init__(
        self,
        *,
        onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
        frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
        min_note_length_ms: int = DEFAULT_MIN_NOTE_LENGTH_MS,
        min_pitch_midi: int = GUITAR_MIDI_MIN,
        max_pitch_midi: int = GUITAR_MIDI_MAX,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_length_ms = min_note_length_ms
        self.min_pitch_midi = min_pitch_midi
        self.max_pitch_midi = max_pitch_midi

    def transcribe(
        self, wav: np.ndarray, sr: int, session: SessionConfig
    ) -> Sequence[AudioEvent]:
        """Run Basic Pitch on ``wav`` and return guitar-range note events.

        Basic Pitch reads audio from disk; we materialize to a temp WAV
        rather than feeding the array directly — the public ``predict``
        wrapper does its own format normalization that's hard to bypass.
        """
        if wav.ndim != 1:
            raise InvalidInputError(f"expected mono wav, got shape {wav.shape}")
        if sr <= 0:
            raise InvalidInputError(f"invalid sample rate: {sr}")

        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401  (presence check)
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise BackendError(
                "basic-pitch is not installed. Install with: pip install basic-pitch"
            ) from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            sf.write(str(wav_path), wav, sr, subtype="PCM_16")

            _, _, note_events = predict(
                str(wav_path),
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.min_note_length_ms,
                minimum_frequency=GUITAR_MIN_HZ,
                maximum_frequency=GUITAR_MAX_HZ,
            )

        events = self._note_events_to_audio_events(note_events)
        events.sort(key=lambda e: e.onset_s)
        return events

    def _note_events_to_audio_events(
        self, note_events: list
    ) -> list[AudioEvent]:
        """Convert Basic Pitch's ``(start, end, midi, amp, bends)`` tuples."""
        guitar_amps: list[float] = []
        for ev in note_events:
            midi = int(ev[2])
            if self.min_pitch_midi <= midi <= self.max_pitch_midi:
                guitar_amps.append(float(ev[3]))
        max_amp = max(guitar_amps) if guitar_amps else 1.0

        out: list[AudioEvent] = []
        for ev in note_events:
            start_s, end_s, midi_raw, amp_raw, bends = ev
            midi = int(midi_raw)
            if midi < self.min_pitch_midi or midi > self.max_pitch_midi:
                continue

            amp = float(amp_raw)
            # Basic Pitch amplitudes are roughly in [0.1, 0.8]. Map linearly
            # into [0.3, 1.0] so quiet-but-clear notes still get useful
            # confidence; matches v0's normalization.
            confidence = 0.3 + (amp / max_amp) * 0.7 if max_amp > 0 else 0.3

            tags: tuple[str, ...] = ()
            if bends is not None and len(bends) > 0 and max(abs(float(b)) for b in bends) > 0.5:
                tags = ("bend",)

            out.append(
                AudioEvent(
                    onset_s=float(start_s),
                    offset_s=float(end_s),
                    pitch_midi=midi,
                    velocity=amp,
                    confidence=float(confidence),
                    tags=tags,
                )
            )

        return out


def transcribe(
    wav: np.ndarray, sr: int, session: SessionConfig
) -> Sequence[AudioEvent]:
    """Convenience function: instantiate the default backend and run."""
    return BasicPitchBackend().transcribe(wav, sr, session)


__all__ = [
    "BasicPitchBackend",
    "transcribe",
    "GUITAR_MIDI_MIN",
    "GUITAR_MIDI_MAX",
]
