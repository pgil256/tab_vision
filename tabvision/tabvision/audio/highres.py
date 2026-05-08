"""High-resolution audio backend — Phase 2 deliverable.

Wraps `xavriley/hf_midi_transcription` (multi-instrument SOTA), which
ships pretrained guitar checkpoints from the Riley/Edwards Domain
Adaptation paper and the Cwitkowitz GAPS paper. MIT-licensed (see
LICENSES.md, verified 2026-05-05).

The package's Python API takes audio file → MIDI file. We materialize
both via temp files, then parse the MIDI back into `AudioEvent` instances.

Pretrained guitar variants exposed as ``checkpoint`` argument:

- ``"guitar"`` / ``"guitar_gaps"`` — Cwitkowitz GAPS-trained (default;
  classical-leaning).
- ``"guitar_fl"`` — Francois Leduc dataset (broader timbral coverage).

Phase 2 acceptance: ``highres`` beats ``basicpitch`` by ≥ 5 pp on Pitch
F1 over the eval set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from tabvision.errors import BackendError, InvalidInputError
from tabvision.types import AudioEvent, SessionConfig

# hf_midi_transcription's model is fixed at 16 kHz; demux gives us 22050.
HIGHRES_SAMPLE_RATE = 16_000

DEFAULT_HF_REPO = "xavriley/midi-transcription-models"

GUITAR_VARIANTS = ("guitar", "guitar_gaps", "guitar_fl")


class HighResBackend:
    """Audio backend wrapping `hf_midi_transcription` for guitar SOTA."""

    name = "highres"

    def __init__(
        self,
        *,
        checkpoint: str = "guitar",
        device: str | None = None,
        batch_size: int = 8,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        frame_threshold: float = 0.1,
        hf_repo: str = DEFAULT_HF_REPO,
    ) -> None:
        if checkpoint not in GUITAR_VARIANTS:
            raise InvalidInputError(
                f"unknown guitar checkpoint: {checkpoint!r}; "
                f"expected one of {GUITAR_VARIANTS}"
            )
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.hf_repo = hf_repo

        self._model = None  # type: ignore[assignment]  # lazy-loaded

    def _load_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model

        try:
            from hf_midi_transcription import MidiTranscriptionModel
        except ImportError as exc:
            raise BackendError(
                "hf-midi-transcription not installed. Install with: "
                "pip install '.[audio-highres]' "
                "(brings in torch, hf-midi-transcription, safetensors)."
            ) from exc

        # The package's HuggingFace `from_pretrained` is broken against
        # current huggingface_hub (missing `proxies` / `resume_download`
        # kwargs). The constructor itself calls `hf_hub_download` to fetch
        # the checkpoint when given an instrument name, so we use that.
        # ``self.hf_repo`` is unused for now; the constructor hard-codes
        # ``xavriley/midi-transcription-models`` as the default repo.
        self._model = MidiTranscriptionModel(
            instrument=self.checkpoint,
            device=self.device,
            batch_size=self.batch_size,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshold,
            frame_threshold=self.frame_threshold,
        )
        return self._model

    def transcribe(
        self, wav: np.ndarray, sr: int, session: SessionConfig
    ) -> Sequence[AudioEvent]:
        if wav.ndim != 1:
            raise InvalidInputError(f"expected mono wav, got shape {wav.shape}")

        wav_16k = _resample_if_needed(wav, sr, HIGHRES_SAMPLE_RATE)
        model = self._load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "in.wav"
            out_path = Path(tmpdir) / "out.mid"
            sf.write(str(in_path), wav_16k, HIGHRES_SAMPLE_RATE, subtype="PCM_16")

            model.transcribe(str(in_path), str(out_path))
            events = _parse_midi(out_path)

        return events


def _resample_if_needed(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear-phase resampling via scipy when sample rates differ."""
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise BackendError(
            "scipy is required to resample audio for the highres backend"
        ) from exc

    # Use rational up/down to avoid integer-rate artifacts.
    from math import gcd

    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return resample_poly(wav, up=up, down=down).astype(np.float32, copy=False)


def _parse_midi(midi_path: Path) -> list[AudioEvent]:
    """Parse a MIDI file produced by hf_midi_transcription into AudioEvents.

    The model writes one MIDI track with note-on / note-off events keyed
    by absolute seconds (via pretty_midi's note objects).
    """
    try:
        import pretty_midi
    except ImportError as exc:
        raise BackendError(
            "pretty_midi is required to parse highres-backend MIDI output"
        ) from exc

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out: list[AudioEvent] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            # MIDI velocity is 0–127; spec velocity is 0–1 float.
            velocity = float(note.velocity) / 127.0 if note.velocity else 0.0
            out.append(
                AudioEvent(
                    onset_s=float(note.start),
                    offset_s=float(note.end),
                    pitch_midi=int(note.pitch),
                    velocity=velocity,
                    # MIDI files don't carry per-note posterior; use velocity
                    # as a proxy for confidence (loud + sustained == confident).
                    # Phase 5 fusion can improve on this when it has access
                    # to the model's raw posteriors.
                    confidence=max(velocity, 0.5),
                )
            )

    out.sort(key=lambda e: e.onset_s)
    return out


__all__ = [
    "HighResBackend",
    "HIGHRES_SAMPLE_RATE",
    "DEFAULT_HF_REPO",
    "GUITAR_VARIANTS",
]
