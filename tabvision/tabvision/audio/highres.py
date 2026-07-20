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

import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from tabvision.audio.filters import AudioFilterConfig, apply_default_filters
from tabvision.errors import BackendError, InvalidInputError
from tabvision.types import AudioEvent, SessionConfig

# hf_midi_transcription's model is fixed at 16 kHz; demux gives us 22050.
HIGHRES_SAMPLE_RATE = 16_000

DEFAULT_HF_REPO = "xavriley/midi-transcription-models"

GUITAR_VARIANTS = ("guitar", "guitar_gaps", "guitar_fl", "guitar_electric")

# Env var holding the path (or HF repo/file) of the fine-tuned electric checkpoint.
# The electric backbone is a v2 deliverable (see the electric fine-tune design doc);
# until it's trained, selecting highres-electric raises a clear, actionable error.
HIGHRES_ELECTRIC_CKPT_ENV = "TABVISION_HIGHRES_ELECTRIC_CKPT"

# The pinned hf_midi_transcription only exposes instrument="guitar" (which maps
# to guitar-gaps.pth). The other guitar checkpoints live in the same HF repo and
# are loaded via checkpoint_path (the package downloads by filename if not local).
_CHECKPOINT_FILE: dict[str, str | None] = {
    "guitar": None,  # package default → guitar-gaps.pth
    "guitar_gaps": "guitar-gaps.pth",
    # Not a built-in default, so give the full HF "repo/file" path: the package
    # only auto-downloads its own defaults or a "<user>/<repo>/<file>" path.
    "guitar_fl": f"{DEFAULT_HF_REPO}/guitar-fl.pth",  # Francois Leduc; electric timbre
}

HIGHRES_FRAMES_PER_SECOND = 100
HIGHRES_BEGIN_NOTE = 21
HIGHRES_CLASSES = 88
PITCH_LOGIT_SIZE = 128
_POSTERIOR_KEYS = (
    "reg_onset_output",
    "reg_offset_output",
    "frame_output",
    "velocity_output",
)


@dataclass(frozen=True)
class HighResPosteriors:
    """Raw frame-level outputs from the pinned high-resolution model."""

    checkpoint: str
    frames_per_second: int
    begin_note: int
    reg_onset_output: np.ndarray
    reg_offset_output: np.ndarray
    frame_output: np.ndarray
    velocity_output: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.reg_onset_output,
            self.reg_offset_output,
            self.frame_output,
            self.velocity_output,
        )
        if self.frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        if self.begin_note < 0:
            raise ValueError("begin_note must be non-negative")
        expected_shape = arrays[0].shape
        if len(expected_shape) != 2 or expected_shape[1] != HIGHRES_CLASSES:
            raise ValueError(
                f"highres posterior arrays must have shape (frames, 88); got {expected_shape}"
            )
        if expected_shape[0] < 1:
            raise ValueError("highres posterior arrays cannot be empty")
        for array in arrays:
            if array.shape != expected_shape:
                raise ValueError("highres posterior arrays must share one shape")
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError("highres posterior arrays must be floating point")
            if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                raise ValueError("highres posterior values must be finite probabilities")

    @property
    def frame_count(self) -> int:
        return int(self.frame_output.shape[0])

    def pitch_logits_at(self, onset_s: float, *, radius_frames: int = 1) -> np.ndarray:
        """Return real onset posterior log-odds indexed directly by MIDI pitch."""

        if radius_frames < 0:
            raise ValueError("radius_frames must be non-negative")
        center = int(round(max(0.0, onset_s) * self.frames_per_second))
        start = max(0, center - radius_frames)
        stop = min(self.frame_count, center + radius_frames + 1)
        if start >= stop:
            start = self.frame_count - 1
            stop = self.frame_count
        probabilities = np.max(self.reg_onset_output[start:stop], axis=0)
        clipped = np.clip(probabilities.astype(np.float64), 1.0e-6, 1.0 - 1.0e-6)
        values = np.full(PITCH_LOGIT_SIZE, -13.815509, dtype=np.float32)
        end_note = self.begin_note + len(clipped)
        if end_note > PITCH_LOGIT_SIZE:
            raise ValueError("posterior MIDI range exceeds pitch_logits size")
        values[self.begin_note : end_note] = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        return values

    def event_scores(self, onset_s: float, pitch_midi: int) -> tuple[float, float]:
        """Return chosen-pitch onset and short-frame posterior maxima."""

        class_index = int(pitch_midi) - self.begin_note
        if not 0 <= class_index < HIGHRES_CLASSES:
            return 0.0, 0.0
        center = min(
            self.frame_count - 1,
            int(round(max(0.0, onset_s) * self.frames_per_second)),
        )
        onset_start = max(0, center - 1)
        onset_stop = min(self.frame_count, center + 2)
        frame_stop = min(self.frame_count, center + 6)
        onset = float(np.max(self.reg_onset_output[onset_start:onset_stop, class_index]))
        frame = float(np.max(self.frame_output[center:frame_stop, class_index]))
        return onset, frame


@dataclass(frozen=True)
class HighResTranscription:
    """Detected events plus the raw posterior tensors from the same pass."""

    events: tuple[AudioEvent, ...]
    posteriors: HighResPosteriors


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
        filter_config: AudioFilterConfig | None | bool = False,
        include_pitch_logits: bool = False,
    ) -> None:
        if checkpoint not in GUITAR_VARIANTS:
            raise InvalidInputError(
                f"unknown guitar checkpoint: {checkpoint!r}; expected one of {GUITAR_VARIANTS}"
            )
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.hf_repo = hf_repo
        self.include_pitch_logits = include_pitch_logits
        if filter_config is True:
            self.filter_config: AudioFilterConfig | None = AudioFilterConfig(
                min_confidence=0.0,
                min_amplitude=0.0,
                harmonic_filter_enabled=False,
                filter_sub_harmonics=False,
            )
        elif filter_config in (False, None):
            self.filter_config = None
        else:
            self.filter_config = filter_config  # type: ignore[assignment]

        self._model = None  # type: ignore[assignment]  # lazy-loaded

    def _load_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model

        # Resolve the checkpoint first so a misconfigured electric backend fails
        # fast with a clear message — before the (heavy) package import.
        if self.checkpoint == "guitar_electric":
            checkpoint_path = os.environ.get(HIGHRES_ELECTRIC_CKPT_ENV)
            if not checkpoint_path:
                raise BackendError(
                    "highres-electric: the electric backbone is not trained yet "
                    "(v2 — see docs/plans/2026-06-02-electric-backbone-finetune-design.md). "
                    f"Set {HIGHRES_ELECTRIC_CKPT_ENV} to a guitar-electric.pth (local "
                    "path or HF repo/file), or use the acoustic backend (--backend highres)."
                )
        else:
            checkpoint_path = _CHECKPOINT_FILE[self.checkpoint]

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

        # instrument="guitar" selects the guitar architecture; checkpoint_path
        # overrides the weights (None → package default guitar-gaps.pth).
        _ensure_utf8_stdio_for_dependency_logs()
        self._model = MidiTranscriptionModel(
            instrument="guitar",
            checkpoint_path=checkpoint_path,
            device=self.device,
            batch_size=self.batch_size,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshold,
            frame_threshold=self.frame_threshold,
        )
        # hf_midi_transcription's MidiTranscriptionModel.__init__ accepts
        # onset_threshold/offset_threshold/frame_threshold above but drops them:
        # it calls self._init_transcriptor(instrument) with only the instrument
        # name, so the underlying piano_transcription_inference.PianoTranscription
        # always falls back to its own hard-coded defaults (0.3/0.3/0.1) — our
        # constructor kwargs are silently inert (confirmed by an eval probe that
        # found onset_threshold/frame_threshold changes produced bit-identical
        # output). PianoTranscription DOES read these as plain instance attributes
        # fresh on every transcribe() call (it rebuilds RegressionPostProcessor
        # from self.onset_threshold / self.offset_threshod [sic — upstream typo,
        # not ours] / self.frame_threshold each time), so set them directly on
        # the underlying transcriptor to make our constructor args actually take
        # effect. No-op today since our defaults equal the library's.
        self._model.transcriptor.onset_threshold = self.onset_threshold
        self._model.transcriptor.offset_threshod = self.offset_threshold
        self._model.transcriptor.frame_threshold = self.frame_threshold
        return self._model

    def transcribe(self, wav: np.ndarray, sr: int, session: SessionConfig) -> Sequence[AudioEvent]:
        result = self.transcribe_with_posteriors(wav, sr, session)
        if self.include_pitch_logits:
            return result.events
        return [_without_pitch_logits(event) for event in result.events]

    def close(self) -> None:
        """Release the lazily loaded checkpoint and its inference graph."""

        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe_with_posteriors(
        self,
        wav: np.ndarray,
        sr: int,
        session: SessionConfig,
    ) -> HighResTranscription:
        """Transcribe once and retain the package's real raw posterior matrices."""

        del session
        if wav.ndim != 1:
            raise InvalidInputError(f"expected mono wav, got shape {wav.shape}")

        wav_16k = _resample_if_needed(wav, sr, HIGHRES_SAMPLE_RATE)
        model = self._load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "in.wav"
            out_path = Path(tmpdir) / "out.mid"
            sf.write(str(in_path), wav_16k, HIGHRES_SAMPLE_RATE, subtype="PCM_16")

            raw_result = model.transcribe(str(in_path), str(out_path), activations=True)
            output_dict = _activation_output_dict(raw_result)
            posteriors = _posteriors_from_output(
                output_dict,
                checkpoint=self.checkpoint,
                duration_s=len(wav_16k) / HIGHRES_SAMPLE_RATE,
            )
            events = _parse_midi(out_path, posteriors=posteriors)

        if self.filter_config is not None:
            events = apply_default_filters(events, self.filter_config)

        return HighResTranscription(tuple(events), posteriors)


def _resample_if_needed(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear-phase resampling via scipy when sample rates differ."""
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise BackendError("scipy is required to resample audio for the highres backend") from exc

    # Use rational up/down to avoid integer-rate artifacts.
    from math import gcd

    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return resample_poly(wav, up=up, down=down).astype(np.float32, copy=False)


def _ensure_utf8_stdio_for_dependency_logs() -> None:
    """Avoid Windows cp1252 crashes from dependency status glyphs."""

    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8":
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _activation_output_dict(raw_result: object) -> dict[str, np.ndarray]:
    """Normalize the pinned package's ``(path, result)`` activation response."""

    payload: object = raw_result
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        payload = raw_result[1]
    if not isinstance(payload, dict) or not isinstance(payload.get("output_dict"), dict):
        raise BackendError("highres backend did not return an activation output_dict")
    output = payload["output_dict"]
    missing = [key for key in _POSTERIOR_KEYS if key not in output]
    if missing:
        raise BackendError(f"highres activation output is missing: {', '.join(missing)}")
    return {key: np.asarray(output[key]) for key in _POSTERIOR_KEYS}


def _posteriors_from_output(
    output_dict: dict[str, np.ndarray],
    *,
    checkpoint: str,
    duration_s: float | None = None,
) -> HighResPosteriors:
    try:
        arrays = {key: np.asarray(output_dict[key], dtype=np.float32) for key in _POSTERIOR_KEYS}
        if duration_s is not None:
            if duration_s <= 0.0:
                raise ValueError("highres posterior duration must be positive")
            frame_extent = duration_s * HIGHRES_FRAMES_PER_SECOND
            # Upstream returns samples [0, duration) after overlap/deframe for
            # exact segment boundaries, but includes the endpoint frame for
            # shorter single-segment inputs. Both cover the complete source.
            minimum_frames = max(1, int(math.floor(frame_extent)))
            target_frames = int(math.ceil(frame_extent)) + 1
            too_short = [key for key, value in arrays.items() if len(value) < minimum_frames]
            if too_short:
                raise ValueError(
                    "posterior output is shorter than the source duration for "
                    + ", ".join(too_short)
                )
            arrays = {key: value[:target_frames] for key, value in arrays.items()}
        return HighResPosteriors(
            checkpoint=checkpoint,
            frames_per_second=HIGHRES_FRAMES_PER_SECOND,
            begin_note=HIGHRES_BEGIN_NOTE,
            reg_onset_output=arrays["reg_onset_output"],
            reg_offset_output=arrays["reg_offset_output"],
            frame_output=arrays["frame_output"],
            velocity_output=arrays["velocity_output"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"invalid highres posterior output: {exc}") from exc


def _parse_midi(
    midi_path: Path,
    *,
    posteriors: HighResPosteriors | None = None,
) -> list[AudioEvent]:
    """Parse a MIDI file produced by hf_midi_transcription into AudioEvents.

    The model writes one MIDI track with note-on / note-off events keyed
    by absolute seconds (via pretty_midi's note objects).
    """
    try:
        import pretty_midi
    except ImportError as exc:
        raise BackendError("pretty_midi is required to parse highres-backend MIDI output") from exc

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
                    pitch_logits=(
                        posteriors.pitch_logits_at(float(note.start))
                        if posteriors is not None
                        else None
                    ),
                )
            )

    out.sort(key=lambda e: e.onset_s)
    return out


def _without_pitch_logits(event: AudioEvent) -> AudioEvent:
    return AudioEvent(
        onset_s=event.onset_s,
        offset_s=event.offset_s,
        pitch_midi=event.pitch_midi,
        velocity=event.velocity,
        confidence=event.confidence,
        pitch_logits=None,
        fret_prior=event.fret_prior,
        tags=event.tags,
    )


__all__ = [
    "HighResBackend",
    "HIGHRES_SAMPLE_RATE",
    "DEFAULT_HF_REPO",
    "GUITAR_VARIANTS",
    "HIGHRES_BEGIN_NOTE",
    "HIGHRES_CLASSES",
    "HIGHRES_FRAMES_PER_SECOND",
    "HighResPosteriors",
    "HighResTranscription",
]
