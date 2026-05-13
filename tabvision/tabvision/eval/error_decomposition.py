"""Tab F1 error decomposition — Phase 0 port of the apr-28 7-bucket harness.

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
        return {f.name: getattr(self, f.name) for f in fields(self)}


def decompose_errors(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = DEFAULT_ONSET_TOLERANCE_S,
    timing_extended_tolerance_s: float = DEFAULT_TIMING_EXTENDED_TOLERANCE_S,
) -> ErrorDecomposition:
    """Bucket the events into the six-bucket Phase 0 schema.

    The matcher is greedy by onset proximity, in two passes:

    1. For each gold event, find the closest unclaimed predicted event
       within ``onset_tolerance_s``. If found, bucket by
       ``(string, fret)`` / ``pitch_midi`` agreement.
    2. For each gold event not matched in pass 1, find the closest
       unclaimed predicted event within ``timing_extended_tolerance_s``
       *that agrees on position or pitch*. If found → ``timing_only``;
       otherwise → ``missed_onset``.

    Unclaimed predicted events after both passes → ``extra_detection``.
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

    gold_sorted = sorted(gold, key=lambda g: g.onset_s)

    for g in gold_sorted:
        # Pass 1: strict-tolerance closest match.
        strict_idx = -1
        strict_dt = onset_tolerance_s + 1e-9
        for pi, p in enumerate(predicted):
            if pred_used[pi]:
                continue
            dt = abs(p.onset_s - g.onset_s)
            if dt <= onset_tolerance_s and dt < strict_dt:
                strict_idx = pi
                strict_dt = dt

        if strict_idx >= 0:
            p = predicted[strict_idx]
            pred_used[strict_idx] = True
            if p.string_idx == g.string_idx and p.fret == g.fret:
                correct += 1
            elif p.pitch_midi == g.pitch_midi:
                wrong_position += 1
            else:
                pitch_off += 1
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
    )


__all__ = [
    "DEFAULT_ONSET_TOLERANCE_S",
    "DEFAULT_TIMING_EXTENDED_TOLERANCE_S",
    "ErrorDecomposition",
    "aggregate_decompositions",
    "decompose_errors",
]
