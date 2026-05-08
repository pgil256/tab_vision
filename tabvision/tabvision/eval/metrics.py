"""Tab F1 + chord-instance accuracy metrics — Phase 5 acceptance.

Definitions follow SPEC.md §9.2:

- **Tab F1**: precision / recall / F1 over (string_idx, fret, onset_s)
  with onset matched within ``onset_tolerance_s`` (default 50 ms).
  Greedy matcher — each predicted event matches at most one gold event,
  picked by closest-onset.
- **Chord instance accuracy**: gold events are grouped into chord
  clusters using the same 80 ms gap rule as
  :mod:`tabvision.fusion.chord`. For each gold cluster, find the closest
  predicted cluster by midpoint onset; the cluster matches if (a) the
  cluster sizes are equal and (b) the multiset of ``(string_idx, fret)``
  tuples matches exactly. Accuracy = matched_chords / total_gold_chords.

These helpers operate on :class:`tabvision.types.TabEvent` sequences so
they can score the output of :func:`tabvision.fusion.fuse` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tabvision.fusion.chord import CHORD_MAX_GAP_S
from tabvision.types import TabEvent


@dataclass(frozen=True)
class TabF1Result:
    """Outcome of :func:`tab_f1`."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def total_predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def total_gold(self) -> int:
        return self.true_positives + self.false_negatives


def tab_f1(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = 0.05,
) -> TabF1Result:
    """Tab F1 over (string, fret, onset)."""
    pred_sorted = sorted(predicted, key=lambda t: t.onset_s)
    gold_sorted = sorted(gold, key=lambda t: t.onset_s)
    gold_used = [False] * len(gold_sorted)
    tp = 0
    fp = 0
    for p in pred_sorted:
        best_j = -1
        best_dt = onset_tolerance_s + 1e-9
        for j, g in enumerate(gold_sorted):
            if gold_used[j]:
                continue
            if g.string_idx != p.string_idx or g.fret != p.fret:
                continue
            dt = abs(g.onset_s - p.onset_s)
            if dt <= onset_tolerance_s and dt < best_dt:
                best_j = j
                best_dt = dt
        if best_j >= 0:
            gold_used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for used in gold_used if not used)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return TabF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


@dataclass(frozen=True)
class ChordAccuracyResult:
    accuracy: float
    matched_chords: int
    total_chords: int


def chord_instance_accuracy(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    cluster_gap_s: float = CHORD_MAX_GAP_S,
    onset_match_tolerance_s: float = 0.05,
) -> ChordAccuracyResult:
    """Fraction of gold chord clusters whose (string, fret) multiset
    matches exactly in the closest predicted cluster.

    A chord cluster is a maximal run of consecutive events whose adjacent
    onset gaps are all ≤ ``cluster_gap_s`` (matches the chord-fusion
    grouping rule). Single-event clusters count toward the metric — a
    correctly transcribed isolated note is a "size-1 chord" instance.
    """
    pred_clusters = _cluster_by_gap(sorted(predicted, key=lambda t: t.onset_s), cluster_gap_s)
    gold_clusters = _cluster_by_gap(sorted(gold, key=lambda t: t.onset_s), cluster_gap_s)

    if not gold_clusters:
        return ChordAccuracyResult(accuracy=0.0, matched_chords=0, total_chords=0)

    matched = 0
    pred_used = [False] * len(pred_clusters)
    for gc in gold_clusters:
        gc_mid = sum(t.onset_s for t in gc) / len(gc)
        best_j = -1
        best_dt = onset_match_tolerance_s + 1e-9
        for j, pc in enumerate(pred_clusters):
            if pred_used[j]:
                continue
            pc_mid = sum(t.onset_s for t in pc) / len(pc)
            dt = abs(pc_mid - gc_mid)
            if dt <= onset_match_tolerance_s and dt < best_dt:
                best_j = j
                best_dt = dt
        if best_j < 0:
            continue
        pc = pred_clusters[best_j]
        if len(pc) != len(gc):
            continue
        gc_set = sorted((t.string_idx, t.fret) for t in gc)
        pc_set = sorted((t.string_idx, t.fret) for t in pc)
        if gc_set == pc_set:
            pred_used[best_j] = True
            matched += 1

    return ChordAccuracyResult(
        accuracy=matched / len(gold_clusters),
        matched_chords=matched,
        total_chords=len(gold_clusters),
    )


def _cluster_by_gap(events: Sequence[TabEvent], gap_s: float) -> list[list[TabEvent]]:
    """Same chain semantics as :func:`tabvision.fusion.chord.cluster_events`,
    but on :class:`TabEvent` (which carries an ``onset_s``). Inlined to
    avoid a sequence-type adapter."""
    if not events:
        return []
    clusters: list[list[TabEvent]] = [[events[0]]]
    for ev in events[1:]:
        if ev.onset_s - clusters[-1][-1].onset_s <= gap_s:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])
    return clusters


__all__ = [
    "TabF1Result",
    "ChordAccuracyResult",
    "tab_f1",
    "chord_instance_accuracy",
]
