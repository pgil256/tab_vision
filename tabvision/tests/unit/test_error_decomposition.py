"""Tests for the Tab F1 error-decomposition module (Phase 0)."""

from __future__ import annotations

import pytest

from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.types import TabEvent


def _ev(onset: float, string_idx: int, fret: int, *, pitch: int | None = None) -> TabEvent:
    """Convenience: TabEvent with default duration, confidence, and derived pitch."""
    # Standard tuning open pitches: low E to high E.
    open_pitches = (40, 45, 50, 55, 59, 64)
    pitch_midi = pitch if pitch is not None else open_pitches[string_idx] + fret
    return TabEvent(
        onset_s=onset,
        duration_s=0.1,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch_midi,
        confidence=1.0,
    )


def test_perfect_match_all_correct() -> None:
    gold = [_ev(0.0, 0, 0), _ev(0.5, 2, 5), _ev(1.0, 4, 3)]
    pred = list(gold)

    r = decompose_errors(pred, gold)

    assert r.correct == 3
    assert r.total_loss == 0
    assert r.wrong_position_same_pitch == 0
    assert r.missed_onset == 0
    assert r.extra_detection == 0


def test_wrong_position_same_pitch_bucket() -> None:
    """E3 (MIDI 64) on high-E open vs MIDI 64 on G string fret 9: same pitch, different position."""
    gold = [_ev(0.0, 5, 0, pitch=64)]  # high E open, MIDI 64
    pred = [_ev(0.0, 2, 9, pitch=64)]  # MIDI 64 placed at G string fret 9 — same pitch

    r = decompose_errors(pred, gold)

    assert r.correct == 0
    assert r.wrong_position_same_pitch == 1
    assert r.pitch_off == 0


def test_pitch_off_bucket() -> None:
    """Onset matches strictly but the predicted pitch is wrong."""
    gold = [_ev(0.0, 0, 0, pitch=40)]
    pred = [_ev(0.01, 0, 1, pitch=41)]  # onset within tolerance, but wrong pitch

    r = decompose_errors(pred, gold)

    assert r.pitch_off == 1
    assert r.correct == 0
    assert r.wrong_position_same_pitch == 0


def test_timing_only_bucket() -> None:
    """Correct position + pitch, but onset just outside strict tolerance, within extended."""
    gold = [_ev(0.0, 0, 0)]
    pred = [_ev(0.10, 0, 0)]  # 100 ms off — outside strict (50 ms), within extended (150 ms)

    r = decompose_errors(pred, gold)

    assert r.timing_only == 1
    assert r.correct == 0
    assert r.missed_onset == 0


def test_missed_onset_bucket() -> None:
    """Gold event with no predicted event nearby at all."""
    gold = [_ev(0.0, 0, 0)]
    pred: list[TabEvent] = []

    r = decompose_errors(pred, gold)

    assert r.missed_onset == 1
    assert r.extra_detection == 0


def test_extra_detection_bucket() -> None:
    """Predicted event with no gold event nearby at all."""
    gold: list[TabEvent] = []
    pred = [_ev(0.0, 0, 0)]

    r = decompose_errors(pred, gold)

    assert r.extra_detection == 1
    assert r.missed_onset == 0


def test_predicted_far_from_gold_yields_missed_and_extra() -> None:
    """Far-apart events should bucket as missed + extra, not pair up."""
    gold = [_ev(0.0, 0, 0)]
    pred = [_ev(10.0, 0, 0)]

    r = decompose_errors(pred, gold)

    assert r.missed_onset == 1
    assert r.extra_detection == 1
    assert r.correct == 0


def test_mixed_buckets() -> None:
    """A mixed scenario across all buckets at once."""
    gold = [
        _ev(0.0, 0, 0),             # correct match
        _ev(0.5, 5, 0, pitch=64),   # wrong-position match (MIDI 64 placed elsewhere)
        _ev(1.0, 2, 5, pitch=55),   # pitch_off (pred at wrong position with wrong pitch)
        _ev(1.5, 3, 7),             # timing_only (pred is 100 ms late)
        _ev(2.0, 4, 3),             # missed_onset
    ]
    pred = [
        _ev(0.01, 0, 0),                  # → correct
        _ev(0.51, 2, 9, pitch=64),        # → wrong_position_same_pitch
        _ev(1.01, 0, 3),                  # → pitch_off (low E fret 3 → MIDI 43, ≠ gold's 55)
        _ev(1.60, 3, 7),                  # → timing_only (100 ms late)
        # Nothing near gold[4] at 2.0 → missed_onset
        _ev(5.0, 0, 0),                   # → extra_detection (far from any gold)
    ]

    r = decompose_errors(pred, gold)

    assert r.correct == 1
    assert r.wrong_position_same_pitch == 1
    assert r.pitch_off == 1
    assert r.timing_only == 1
    assert r.missed_onset == 1
    assert r.extra_detection == 1


def test_share_of_loss_sums_to_one() -> None:
    r = ErrorDecomposition(
        correct=10,
        wrong_position_same_pitch=3,
        pitch_off=2,
        timing_only=1,
        missed_onset=2,
        extra_detection=2,
    )
    shares = r.share_of_loss()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["wrong_position_same_pitch"] == pytest.approx(3 / 10)


def test_share_of_loss_zero_when_no_loss() -> None:
    r = ErrorDecomposition(correct=5)
    shares = r.share_of_loss()
    assert all(v == 0.0 for v in shares.values())


def test_total_gold_excludes_extra_detection() -> None:
    r = ErrorDecomposition(
        correct=10, wrong_position_same_pitch=2, pitch_off=1, missed_onset=3, extra_detection=5
    )
    # total_gold = correct + wrong_pos + pitch_off + timing_only + missed_onset
    assert r.total_gold == 16
    # total_predicted = correct + wrong_pos + pitch_off + timing_only + extra_detection
    assert r.total_predicted == 18


def test_aggregate_decompositions_sums_bucketwise() -> None:
    a = ErrorDecomposition(correct=5, wrong_position_same_pitch=2)
    b = ErrorDecomposition(correct=10, missed_onset=3, extra_detection=1)
    agg = aggregate_decompositions([a, b])
    assert agg.correct == 15
    assert agg.wrong_position_same_pitch == 2
    assert agg.missed_onset == 3
    assert agg.extra_detection == 1
    assert agg.pitch_off == 0


def test_aggregate_empty_returns_zeros() -> None:
    agg = aggregate_decompositions([])
    assert agg == ErrorDecomposition()
    assert agg.total_loss == 0


def test_rejects_invalid_tolerances() -> None:
    with pytest.raises(ValueError, match="onset_tolerance_s"):
        decompose_errors([], [], onset_tolerance_s=0.0)
    with pytest.raises(ValueError, match=">="):
        decompose_errors([], [], onset_tolerance_s=0.1, timing_extended_tolerance_s=0.05)


def test_each_pred_matches_at_most_one_gold() -> None:
    """Two gold events at the same time should not both claim one pred."""
    gold = [_ev(0.0, 0, 0), _ev(0.0, 0, 0)]
    pred = [_ev(0.0, 0, 0)]

    r = decompose_errors(pred, gold)

    assert r.correct == 1
    assert r.missed_onset == 1
    assert r.extra_detection == 0


def test_greedy_picks_closest_onset() -> None:
    """When multiple same-position preds are within tolerance, the closest-by-onset wins."""
    gold = [_ev(0.0, 0, 0)]
    pred = [_ev(0.04, 0, 0), _ev(0.01, 0, 0)]  # both within 50 ms; 0.01 is closer

    r = decompose_errors(pred, gold)

    assert r.correct == 1
    assert r.extra_detection == 1


def test_chord_cluster_priority_pitch_over_onset() -> None:
    """Multi-gold same-onset chord: matcher should pair by pitch, not by onset proximity.

    Two gold events at the same onset with different pitches, paired
    with two preds whose pitches match the gold (but whose on-the-wire
    ordering doesn't). Onset-only greediness would mis-pair them and
    inflate ``pitch_off``. The priority-based matcher must pair on
    pitch.
    """
    gold = [
        _ev(0.0, 0, 0, pitch=40),  # low E
        _ev(0.0, 1, 2, pitch=47),  # A string fret 2
    ]
    pred = [
        # Different on-the-wire order: pitch=47 first.
        _ev(0.01, 1, 2, pitch=47),  # → matches gold[1] (correct)
        _ev(0.01, 0, 0, pitch=40),  # → matches gold[0] (correct)
    ]

    r = decompose_errors(pred, gold)

    assert r.correct == 2
    assert r.pitch_off == 0
    assert r.wrong_position_same_pitch == 0


def test_chord_cluster_priority_falls_back_to_position_match_then_pitch() -> None:
    """When one pred has the right position and another has the right pitch,
    the same-position match wins for ``correct`` accounting.
    """
    gold = [_ev(0.0, 0, 0, pitch=40)]
    pred = [
        # Same pitch as gold but different position
        _ev(0.005, 5, 0, pitch=64),  # noise; nothing in common
        _ev(0.020, 0, 0, pitch=40),  # exact match; further in onset
    ]

    r = decompose_errors(pred, gold)

    assert r.correct == 1  # picked the same-position match even though it's further
