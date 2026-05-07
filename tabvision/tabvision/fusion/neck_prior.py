"""Attach coarse hand-neck anchors to audio events as fret priors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import numpy as np

from tabvision.types import AudioEvent, GuitarConfig


class NeckAnchorLike(Protocol):
    center_fret: float
    min_fret: float
    max_fret: float
    confidence: float


TimedNeckAnchor = tuple[float, NeckAnchorLike]


def apply_neck_anchor_priors(
    events: Sequence[AudioEvent],
    anchors: Sequence[TimedNeckAnchor],
    cfg: GuitarConfig,
    *,
    max_time_distance_s: float = 0.15,
) -> list[AudioEvent]:
    """Return events enriched with nearest video-anchor position priors.

    The resulting ``AudioEvent.fret_prior`` has shape
    ``(cfg.n_strings, cfg.max_fret + 1)`` so Phase 5 playability emission can
    consume it as a per-position prior.
    """
    if not anchors:
        return list(events)

    out: list[AudioEvent] = []
    for ev in events:
        nearest = min(anchors, key=lambda item: abs(item[0] - ev.onset_s))
        dt = abs(nearest[0] - ev.onset_s)
        if dt > max_time_distance_s or nearest[1].confidence <= 0.0:
            out.append(ev)
            continue
        prior = anchor_position_prior(nearest[1], cfg)
        if ev.fret_prior is not None:
            prior = _combine_priors(ev.fret_prior, prior, cfg)
        out.append(replace(ev, fret_prior=prior))
    return out


def anchor_position_prior(anchor: NeckAnchorLike, cfg: GuitarConfig) -> np.ndarray:
    """Return a normalized ``(string, fret)`` prior from a hand-neck anchor."""
    frets = np.arange(cfg.max_fret + 1, dtype=np.float64)
    uniform_fret = np.full(cfg.max_fret + 1, 1.0 / (cfg.max_fret + 1), dtype=np.float64)
    if anchor.confidence <= 0.0:
        fret_probs = uniform_fret
    else:
        sigma = max((float(anchor.max_fret) - float(anchor.min_fret)) / 2.0, 1.0)
        logits = -0.5 * ((frets - float(anchor.center_fret)) / sigma) ** 2
        gaussian = np.exp(logits - float(logits.max()))
        gaussian /= float(gaussian.sum())
        weight = min(max(float(anchor.confidence), 0.0), 1.0)
        fret_probs = weight * gaussian + (1.0 - weight) * uniform_fret
        fret_probs /= float(fret_probs.sum())

    prior = np.tile(fret_probs[None, :], (cfg.n_strings, 1))
    prior /= float(prior.sum())
    return prior


def _combine_priors(
    existing: np.ndarray, anchor_prior: np.ndarray, cfg: GuitarConfig
) -> np.ndarray:
    existing_position = _as_position_prior(existing, cfg)
    combined = existing_position * anchor_prior
    denom = float(combined.sum())
    if denom <= 0.0:
        return anchor_prior
    return np.asarray(combined / denom, dtype=np.float64)


def _as_position_prior(prior: np.ndarray, cfg: GuitarConfig) -> np.ndarray:
    arr = np.asarray(prior, dtype=np.float64)
    if arr.shape == (cfg.n_strings, cfg.max_fret + 1):
        denom = float(arr.sum())
        if denom > 0.0:
            return np.asarray(arr / denom, dtype=np.float64)
        return anchor_position_prior(_ZeroAnchor(), cfg)
    if arr.shape == (cfg.max_fret + 1,):
        out = np.tile(arr[None, :], (cfg.n_strings, 1))
        denom = float(out.sum())
        if denom > 0.0:
            return np.asarray(out / denom, dtype=np.float64)
        return anchor_position_prior(_ZeroAnchor(), cfg)
    return anchor_position_prior(_ZeroAnchor(), cfg)


class _ZeroAnchor:
    center_fret = 0.0
    min_fret = 0.0
    max_fret = 0.0
    confidence = 0.0


__all__ = [
    "NeckAnchorLike",
    "TimedNeckAnchor",
    "anchor_position_prior",
    "apply_neck_anchor_priors",
]
