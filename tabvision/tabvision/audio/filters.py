"""Audio-event post-processing filters.

Ports v0's ``audio_pipeline.py`` filter pipeline (sustain redetection,
harmonic, ghost-note merge, low-quality drop) onto the spec's
``AudioEvent`` data type.

Phase 1 polish: applied after Basic Pitch transcription to reduce the
~3× over-detection observed in the bare-Basic-Pitch Phase 1 eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from tabvision.types import AudioEvent

# Harmonic intervals (semitones above the fundamental) where Basic Pitch
# tends to spawn phantom detections. From v0 ``audio_pipeline.py``.
HARMONIC_INTERVALS = (12, 19, 24, 28, 31)
# Octave (12) and octave+fifth (19) are also legitimate musical voicings,
# so we apply a stricter amplitude ratio there to avoid removing real notes.
MUSICAL_INTERVALS = frozenset({12, 19})


@dataclass(frozen=True)
class AudioFilterConfig:
    """Knobs for the v0 filter pipeline. Defaults match v0's tuned values."""

    # Drop events below this confidence.
    min_confidence: float = 0.3
    # Drop events shorter than this (Basic Pitch artifacts).
    min_duration_s: float = 0.03
    # Drop events with raw amplitude below this.
    min_amplitude: float = 0.1

    # Same-pitch merge: events within this gap (sec) are coalesced.
    # Negative ⇒ overlapping ⇒ also merge.
    merge_gap_s: float = 0.02

    # Sustain re-detection: same-pitch event within this window …
    sustain_window_s: float = 0.6
    # … and amplitude < ratio × original ⇒ drop as sustain artifact.
    sustain_amplitude_ratio: float = 0.95

    # Harmonic filter: simultaneous-ish (within tolerance) higher pitch
    # at a harmonic interval below ratio × fundamental amplitude ⇒ drop.
    harmonic_time_tolerance_s: float = 0.15
    harmonic_amplitude_ratio: float = 0.7
    # Stricter ratio for octave / octave+5 (also musical voicings).
    musical_interval_amplitude_ratio: float = 0.35
    # Notes louder than this raw amplitude are protected from harmonic
    # filtering (real notes are rarely this loud as harmonics).
    harmonic_protect_amplitude: float = 0.45
    # Also remove sub-harmonics (low-frequency artifacts).
    filter_sub_harmonics: bool = True


# ---------- individual filters ----------


def filter_low_quality(
    events: Sequence[AudioEvent], cfg: AudioFilterConfig
) -> list[AudioEvent]:
    """Drop events below confidence / duration / amplitude floors."""
    out: list[AudioEvent] = []
    for ev in events:
        if ev.confidence < cfg.min_confidence:
            continue
        if (ev.offset_s - ev.onset_s) < cfg.min_duration_s:
            continue
        if ev.velocity < cfg.min_amplitude:
            continue
        out.append(ev)
    return out


def merge_consecutive(
    events: Sequence[AudioEvent], cfg: AudioFilterConfig
) -> list[AudioEvent]:
    """Coalesce same-pitch overlapping/consecutive events.

    Basic Pitch sometimes splits a single sustained note into multiple
    events. Group by pitch and merge adjacent (or overlapping) events.
    """
    if not events:
        return []

    by_pitch: dict[int, list[AudioEvent]] = {}
    for ev in events:
        by_pitch.setdefault(ev.pitch_midi, []).append(ev)

    merged: list[AudioEvent] = []
    for group in by_pitch.values():
        group.sort(key=lambda e: e.onset_s)
        current = group[0]
        for ev in group[1:]:
            gap = ev.onset_s - current.offset_s
            if gap <= cfg.merge_gap_s:
                current = AudioEvent(
                    onset_s=current.onset_s,
                    offset_s=max(current.offset_s, ev.offset_s),
                    pitch_midi=current.pitch_midi,
                    velocity=max(current.velocity, ev.velocity),
                    confidence=max(current.confidence, ev.confidence),
                    pitch_logits=None,
                    fret_prior=None,
                    tags=tuple(set(current.tags) | set(ev.tags)),
                )
            else:
                merged.append(current)
                current = ev
        merged.append(current)

    merged.sort(key=lambda e: e.onset_s)
    return merged


def filter_sustain_redetections(
    events: Sequence[AudioEvent], cfg: AudioFilterConfig
) -> list[AudioEvent]:
    """Drop quieter same-pitch events that fall while a previous one rings."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.onset_s)
    keep: list[AudioEvent] = []

    for i, ev in enumerate(sorted_events):
        is_redetection = False
        for j in range(i - 1, -1, -1):
            prev = sorted_events[j]
            time_gap = ev.onset_s - prev.onset_s
            if time_gap > cfg.sustain_window_s:
                break
            if prev.pitch_midi != ev.pitch_midi:
                continue
            # Previous event still ringing (end ≥ current start − slop)?
            if prev.offset_s >= ev.onset_s - 0.1:
                if ev.velocity < prev.velocity * cfg.sustain_amplitude_ratio:
                    is_redetection = True
                    break
        if not is_redetection:
            keep.append(ev)

    return keep


def filter_harmonics(
    events: Sequence[AudioEvent], cfg: AudioFilterConfig
) -> list[AudioEvent]:
    """Drop events at harmonic intervals above (or sub-harmonic intervals
    below) a louder near-simultaneous fundamental.

    Loud events (``velocity ≥ harmonic_protect_amplitude``) are protected
    outright — they are unlikely to be overtones.
    """
    if not events:
        return []

    out: list[AudioEvent] = []

    for ev in events:
        if ev.velocity >= cfg.harmonic_protect_amplitude:
            out.append(ev)
            continue

        if _is_harmonic(ev, events, cfg):
            continue

        out.append(ev)

    return out


def _is_harmonic(
    ev: AudioEvent,
    others: Iterable[AudioEvent],
    cfg: AudioFilterConfig,
) -> bool:
    """Return True if ``ev`` is dominated by a near-simultaneous other event
    at a harmonic or (optionally) sub-harmonic interval.
    """
    for other in others:
        if other is ev:
            continue
        if abs(ev.onset_s - other.onset_s) > cfg.harmonic_time_tolerance_s:
            continue

        # ev is a harmonic ABOVE other?
        interval_above = ev.pitch_midi - other.pitch_midi
        if interval_above in HARMONIC_INTERVALS:
            threshold = (
                cfg.musical_interval_amplitude_ratio
                if interval_above in MUSICAL_INTERVALS
                else cfg.harmonic_amplitude_ratio
            )
            if ev.velocity < other.velocity * threshold:
                return True

        # ev is a SUB-harmonic BELOW other?
        if cfg.filter_sub_harmonics:
            interval_below = other.pitch_midi - ev.pitch_midi
            if interval_below in HARMONIC_INTERVALS:
                threshold = (
                    cfg.musical_interval_amplitude_ratio
                    if interval_below in MUSICAL_INTERVALS
                    else cfg.harmonic_amplitude_ratio
                )
                if ev.velocity < other.velocity * threshold:
                    return True

    return False


# ---------- pipeline ----------


def apply_default_filters(
    events: Sequence[AudioEvent], cfg: AudioFilterConfig | None = None
) -> list[AudioEvent]:
    """Run the v0-equivalent filter pipeline in canonical order.

    1. Drop low-confidence / short / quiet events.
    2. Merge same-pitch consecutive/overlapping events.
    3. Drop sustain re-detections.
    4. Drop harmonic / sub-harmonic artifacts.
    """
    if cfg is None:
        cfg = AudioFilterConfig()

    out = list(events)
    out = filter_low_quality(out, cfg)
    out = merge_consecutive(out, cfg)
    out = filter_sustain_redetections(out, cfg)
    out = filter_harmonics(out, cfg)
    out.sort(key=lambda e: e.onset_s)
    return out


__all__ = [
    "AudioFilterConfig",
    "filter_low_quality",
    "merge_consecutive",
    "filter_sustain_redetections",
    "filter_harmonics",
    "apply_default_filters",
    "HARMONIC_INTERVALS",
    "MUSICAL_INTERVALS",
]
