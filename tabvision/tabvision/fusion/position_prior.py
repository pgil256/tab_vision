"""Learned pitch-to-position priors for audio-only tab decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tabvision.types import AudioEvent, GuitarConfig, TabEvent


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


__all__ = [
    "PitchPositionPrior",
    "apply_pitch_position_prior",
    "learn_pitch_position_prior",
]
