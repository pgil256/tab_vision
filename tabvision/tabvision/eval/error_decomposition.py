"""Tab F1 error decomposition — six-bucket port of the apr-28 7-bucket harness.

Ports the methodology from
``tabvision-server/tools/outputs/errors-2026-04-28_185743.md`` to operate
on §8 ``TabEvent`` lists (the v1 contract) instead of the v0 internal
``Note`` representation.

Six failure buckets (the apr-28 ``muted_undetectable`` bucket needs a
muted/X flag the v1 contract does not yet carry; deferred to a later
phase):

- ``correct``: predicted event matches a gold event on string + fret
  + onset within ``onset_tolerance_s``.
- ``wrong_position_same_pitch``: predicted event matches on
  ``pitch_midi`` + onset within tolerance, but a different
  ``(string_idx, fret)``. This is the bucket that dominated the
  2026-05-08 GuitarSet validation (~35% of loss on personal clips per
  the apr-28 report).
- ``pitch_off``: predicted event aligns in onset but pitch_midi
  differs from the matched gold. Audio-side loss.
- ``timing_only``: predicted event matches on position or pitch but
  the onset is outside ``onset_tolerance_s`` and within
  ``timing_extended_tolerance_s``.
- ``missed_onset``: gold event has no predicted event near it within
  the extended tolerance.
- ``extra_detection``: predicted event that did not match any gold
  event by either rule above.

Per the strategy doc §2 the dominant failure axis is
``wrong_position_same_pitch`` on solos. This module lets us measure
that explicitly per tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields

from tabvision.types import TabEvent

DEFAULT_ONSET_TOLERANCE_S = 0.05
DEFAULT_TIMING_EXTENDED_TOLERANCE_S = 0.15

PITCH_OFF_DELTA_CLASSES: tuple[str, ...] = ("octave", "harmonic", "semitone", "other")
"""Coarse classes for ``pitch_off`` semitone deltas (A10 instrumentation).

Each class points at a different fix:

- ``octave``: |Δ| ≡ 0 (mod 12) — octave errors (2nd/4th-harmonic
  confusions or octave-transposed detections). Fixable by
  octave-disambiguation logic, not by better f0 resolution.
- ``harmonic``: |Δ| ≡ 5 or 7 (mod 12) — fifth/fourth chroma family
  (3rd-harmonic +19, fifth +7, subharmonic/fourth −5/+5, +17…).
  Harmonic-series leakage in the detector.
- ``semitone``: |Δ| ≤ 2 — adjacent-bin / tuning / bend errors.
- ``other``: everything else (usually genuine mis-detections).
"""


def classify_pitch_off_delta(delta: int) -> str:
    """Map a signed semitone delta (predicted − gold) to its coarse class."""
    magnitude = abs(delta)
    if magnitude == 0:
        # Unreachable from decompose_errors (equal pitch buckets earlier),
        # but keep the function total.
        return "other"
    if magnitude % 12 == 0:
        return "octave"
    if magnitude <= 2:
        return "semitone"
    if magnitude % 12 in (5, 7):
        return "harmonic"
    return "other"


def pitch_off_delta_histogram(deltas: Iterable[int]) -> dict[int, int]:
    """Signed-delta → count histogram, sorted by delta."""
    counts: dict[int, int] = {}
    for delta in deltas:
        counts[delta] = counts.get(delta, 0) + 1
    return dict(sorted(counts.items()))


def pitch_off_class_counts(deltas: Iterable[int]) -> dict[str, int]:
    """Per-class counts over all four classes (zero-filled)."""
    counts = dict.fromkeys(PITCH_OFF_DELTA_CLASSES, 0)
    for delta in deltas:
        counts[classify_pitch_off_delta(delta)] += 1
    return counts


@dataclass(frozen=True)
class ErrorDecomposition:
    """Six-bucket failure breakdown for one (predicted, gold) pair.

    Construct via :func:`decompose_errors`; sum across tracks via
    :func:`aggregate_decompositions`. Bucket counts are non-negative
    integers.
    """

    correct: int = 0
    wrong_position_same_pitch: int = 0
    pitch_off: int = 0
    timing_only: int = 0
    missed_onset: int = 0
    extra_detection: int = 0
    pitch_off_deltas: tuple[int, ...] = ()
    """Signed semitone delta (predicted − gold) per ``pitch_off`` event.

    A10 instrumentation: length always equals ``pitch_off``. Summarize
    via :func:`pitch_off_delta_histogram` / :func:`pitch_off_class_counts`.
    """

    @property
    def total_gold(self) -> int:
        """Number of gold events accounted for. Excludes ``extra_detection``."""
        return (
            self.correct
            + self.wrong_position_same_pitch
            + self.pitch_off
            + self.timing_only
            + self.missed_onset
        )

    @property
    def total_predicted(self) -> int:
        """Number of predicted events accounted for. Excludes ``missed_onset``."""
        return (
            self.correct
            + self.wrong_position_same_pitch
            + self.pitch_off
            + self.timing_only
            + self.extra_detection
        )

    @property
    def total_loss(self) -> int:
        """Events contributing to Tab F1 loss (everything except ``correct``)."""
        return (
            self.wrong_position_same_pitch
            + self.pitch_off
            + self.timing_only
            + self.missed_onset
            + self.extra_detection
        )

    def share_of_loss(self) -> dict[str, float]:
        """Per-bucket share of recoverable Tab F1 loss.

        ``correct`` events are not counted as loss; the remaining five
        buckets sum to 1.0 (or all zeros if ``total_loss`` is 0).
        """
        total = self.total_loss
        if total == 0:
            return {
                "wrong_position_same_pitch": 0.0,
                "pitch_off": 0.0,
                "timing_only": 0.0,
                "missed_onset": 0.0,
                "extra_detection": 0.0,
            }
        return {
            "wrong_position_same_pitch": self.wrong_position_same_pitch / total,
            "pitch_off": self.pitch_off / total,
            "timing_only": self.timing_only / total,
            "missed_onset": self.missed_onset / total,
            "extra_detection": self.extra_detection / total,
        }

    def to_dict(self) -> dict[str, int]:
        """Bucket counts only; ``pitch_off_deltas`` is per-event data, not a count."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "pitch_off_deltas"}


def decompose_errors(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = DEFAULT_ONSET_TOLERANCE_S,
    timing_extended_tolerance_s: float = DEFAULT_TIMING_EXTENDED_TOLERANCE_S,
) -> ErrorDecomposition:
    """Bucket the events into the six-bucket Phase 0 schema.

    The matcher is **priority-based** within each tolerance window so
    chord clusters (multiple gold events at the same onset) don't get
    mis-paired by raw onset proximity:

    1. **Strict-tolerance pass.** For each gold event, search unclaimed
       predicted events within ``onset_tolerance_s``. Pick the best in
       priority order:
       - same ``(string_idx, fret)`` → ``correct``
       - same ``pitch_midi`` → ``wrong_position_same_pitch``
       - neither → ``pitch_off``
       Within each priority bucket, ties are broken by closest onset.
    2. **Extended-tolerance pass.** For each gold event still unmatched,
       search within ``timing_extended_tolerance_s`` for a predicted
       event that agrees on position or pitch → ``timing_only``.
       Else → ``missed_onset``.

    Unclaimed predicted events after both passes → ``extra_detection``.

    Priority matters: in a chord cluster with three gold events at the
    same onset and three predicted events with matching pitches but
    different on-the-wire ordering, onset-only greediness would shuffle
    pairings and inflate ``pitch_off``. Priority-based matching tracks
    ``event_f1(match_pitch=True)`` exactly when ``Pitch F1 = 1.0``.
    """
    if onset_tolerance_s <= 0:
        raise ValueError(f"onset_tolerance_s must be positive; got {onset_tolerance_s}")
    if timing_extended_tolerance_s < onset_tolerance_s:
        raise ValueError(
            f"timing_extended_tolerance_s ({timing_extended_tolerance_s}) must be "
            f">= onset_tolerance_s ({onset_tolerance_s})"
        )

    pred_used = [False] * len(predicted)

    correct = 0
    wrong_position = 0
    pitch_off = 0
    timing_only = 0
    missed = 0
    pitch_off_deltas: list[int] = []

    gold_sorted = sorted(gold, key=lambda g: g.onset_s)

    for g in gold_sorted:
        # Pass 1: strict-tolerance, priority-ordered match.
        best_pos_idx = -1
        best_pitch_idx = -1
        best_any_idx = -1
        best_pos_dt = onset_tolerance_s + 1e-9
        best_pitch_dt = onset_tolerance_s + 1e-9
        best_any_dt = onset_tolerance_s + 1e-9

        for pi, p in enumerate(predicted):
            if pred_used[pi]:
                continue
            dt = abs(p.onset_s - g.onset_s)
            if dt > onset_tolerance_s:
                continue
            same_pos = p.string_idx == g.string_idx and p.fret == g.fret
            same_pitch = p.pitch_midi == g.pitch_midi
            if same_pos:
                if dt < best_pos_dt:
                    best_pos_idx = pi
                    best_pos_dt = dt
            elif same_pitch:
                if dt < best_pitch_dt:
                    best_pitch_idx = pi
                    best_pitch_dt = dt
            elif dt < best_any_dt:
                best_any_idx = pi
                best_any_dt = dt

        if best_pos_idx >= 0:
            pred_used[best_pos_idx] = True
            correct += 1
            continue
        if best_pitch_idx >= 0:
            pred_used[best_pitch_idx] = True
            wrong_position += 1
            continue
        if best_any_idx >= 0:
            pred_used[best_any_idx] = True
            pitch_off += 1
            pitch_off_deltas.append(predicted[best_any_idx].pitch_midi - g.pitch_midi)
            continue

        # Pass 2: extended-tolerance match on position OR pitch.
        timing_idx = -1
        timing_dt = timing_extended_tolerance_s + 1e-9
        for pi, p in enumerate(predicted):
            if pred_used[pi]:
                continue
            dt = abs(p.onset_s - g.onset_s)
            if dt > timing_extended_tolerance_s:
                continue
            same_pos = p.string_idx == g.string_idx and p.fret == g.fret
            same_pitch = p.pitch_midi == g.pitch_midi
            if (same_pos or same_pitch) and dt < timing_dt:
                timing_idx = pi
                timing_dt = dt

        if timing_idx >= 0:
            pred_used[timing_idx] = True
            timing_only += 1
            continue

        missed += 1

    extra = sum(1 for used in pred_used if not used)

    return ErrorDecomposition(
        correct=correct,
        wrong_position_same_pitch=wrong_position,
        pitch_off=pitch_off,
        timing_only=timing_only,
        missed_onset=missed,
        extra_detection=extra,
        pitch_off_deltas=tuple(pitch_off_deltas),
    )


def aggregate_decompositions(
    decompositions: Iterable[ErrorDecomposition],
) -> ErrorDecomposition:
    """Sum a sequence of per-track decompositions into an aggregate."""
    items = list(decompositions)
    return ErrorDecomposition(
        correct=sum(d.correct for d in items),
        wrong_position_same_pitch=sum(d.wrong_position_same_pitch for d in items),
        pitch_off=sum(d.pitch_off for d in items),
        timing_only=sum(d.timing_only for d in items),
        missed_onset=sum(d.missed_onset for d in items),
        extra_detection=sum(d.extra_detection for d in items),
        pitch_off_deltas=tuple(delta for d in items for delta in d.pitch_off_deltas),
    )


__all__ = [
    "DEFAULT_ONSET_TOLERANCE_S",
    "DEFAULT_TIMING_EXTENDED_TOLERANCE_S",
    "PITCH_OFF_DELTA_CLASSES",
    "ErrorDecomposition",
    "aggregate_decompositions",
    "classify_pitch_off_delta",
    "decompose_errors",
    "pitch_off_class_counts",
    "pitch_off_delta_histogram",
]
