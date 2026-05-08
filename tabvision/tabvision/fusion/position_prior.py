"""Learned pitch-to-position priors for audio-only tab decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabvision.types import AudioEvent, GuitarConfig, TabEvent

_PRIORS_DIR = Path(__file__).with_name("priors")
_NAMED_PRIORS = {
    "guitarset-v1": _PRIORS_DIR / "guitarset_v1.json",
}


@dataclass(frozen=True)
class PitchPositionPrior:
    """Mapping from MIDI pitch to a normalized ``(string, fret)`` prior."""

    by_pitch: Mapping[int, np.ndarray]

    def matrix_for_pitch(self, pitch_midi: int) -> np.ndarray | None:
        return self.by_pitch.get(int(pitch_midi))


def learn_pitch_position_prior(
    examples: Sequence[TabEvent],
    cfg: GuitarConfig | None = None,
    *,
    alpha: float = 1.0,
    power: float = 2.0,
) -> PitchPositionPrior:
    """Estimate ``P(string, fret | pitch)`` from tab-labelled examples.

    Smoothing is applied only to playable candidates for each pitch. The
    optional ``power`` sharpens observed preferences while preserving zero
    probability for impossible positions.
    """
    if cfg is None:
        cfg = GuitarConfig()
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if power <= 0:
        raise ValueError("power must be positive")

    priors: dict[int, np.ndarray] = {}
    for pitch in range(128):
        arr = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
        for string_idx, open_pitch in enumerate(cfg.tuning_midi):
            fret = pitch - open_pitch
            if cfg.capo <= fret <= cfg.max_fret:
                arr[string_idx, fret] = alpha
        priors[pitch] = arr

    for ev in examples:
        if ev.pitch_midi not in priors:
            continue
        if not (0 <= ev.string_idx < cfg.n_strings):
            continue
        if not (0 <= ev.fret <= cfg.max_fret):
            continue
        priors[ev.pitch_midi][ev.string_idx, ev.fret] += 1.0

    normalized: dict[int, np.ndarray] = {}
    for pitch, arr in priors.items():
        sharpened = arr**power
        total = float(sharpened.sum())
        if total > 0:
            normalized[pitch] = sharpened / total
    return PitchPositionPrior(normalized)


def apply_pitch_position_prior(
    events: Sequence[AudioEvent],
    prior: PitchPositionPrior,
) -> list[AudioEvent]:
    """Return copies of audio events with a pitch-position prior attached."""
    out: list[AudioEvent] = []
    for ev in events:
        matrix = prior.matrix_for_pitch(ev.pitch_midi)
        out.append(
            AudioEvent(
                onset_s=ev.onset_s,
                offset_s=ev.offset_s,
                pitch_midi=ev.pitch_midi,
                velocity=ev.velocity,
                confidence=ev.confidence,
                pitch_logits=ev.pitch_logits,
                fret_prior=matrix if matrix is not None else ev.fret_prior,
                tags=ev.tags,
            )
        )
    return out


def load_pitch_position_prior(
    name_or_path: str | Path,
    *,
    cfg: GuitarConfig | None = None,
) -> PitchPositionPrior:
    """Load a versioned pitch-position prior artifact.

    Named artifacts are checked into ``tabvision.fusion.priors`` so runtime
    transcription never needs raw GuitarSet files. A filesystem path may also
    be supplied for reproducible experiments.
    """
    if cfg is None:
        cfg = GuitarConfig()

    key = str(name_or_path)
    path = _NAMED_PRIORS.get(key)
    if path is None:
        candidate = Path(key)
        if candidate.is_file():
            path = candidate
        else:
            known = ", ".join(sorted(_NAMED_PRIORS))
            raise ValueError(f"unknown pitch-position prior {key!r}; known: {known}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported pitch-position prior schema in {path}")
    counts = payload.get("counts")
    if not isinstance(counts, list):
        raise ValueError(f"pitch-position prior artifact missing counts: {path}")

    examples: list[TabEvent] = []
    for row in counts:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"invalid prior count row in {path}: {row!r}")
        pitch_midi, string_idx, fret, count = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if count < 0:
            raise ValueError(f"invalid negative prior count in {path}: {row!r}")
        examples.extend(
            TabEvent(
                onset_s=0.0,
                duration_s=0.0,
                string_idx=string_idx,
                fret=fret,
                pitch_midi=pitch_midi,
                confidence=1.0,
            )
            for _ in range(count)
        )
    return learn_pitch_position_prior(
        examples,
        cfg=cfg,
        alpha=float(payload.get("alpha", 1.0)),
        power=float(payload.get("power", 2.0)),
    )


__all__ = [
    "PitchPositionPrior",
    "apply_pitch_position_prior",
    "learn_pitch_position_prior",
    "load_pitch_position_prior",
]
