"""Unit tests for ``tabvision.eval.metrics`` (Tab F1 + chord accuracy)."""

from __future__ import annotations

from tabvision.eval.metrics import chord_instance_accuracy, tab_f1
from tabvision.types import TabEvent


def _t(t: float, s: int, f: int, midi: int = 60) -> TabEvent:
    return TabEvent(
        onset_s=t,
        duration_s=0.25,
        string_idx=s,
        fret=f,
        pitch_midi=midi,
        confidence=0.9,
    )


# ---------- tab_f1 ----------


def test_tab_f1_perfect_match():
    gold = [_t(0.0, 5, 5), _t(0.5, 5, 7)]
    pred = [_t(0.0, 5, 5), _t(0.5, 5, 7)]
    r = tab_f1(pred, gold)
    assert r.f1 == 1.0
    assert r.true_positives == 2
    assert r.false_positives == 0
    assert r.false_negatives == 0


def test_tab_f1_extra_prediction_lowers_precision():
    gold = [_t(0.0, 5, 5)]
    pred = [_t(0.0, 5, 5), _t(0.5, 5, 7)]
    r = tab_f1(pred, gold)
    assert r.true_positives == 1
    assert r.false_positives == 1
    assert r.false_negatives == 0
    assert r.recall == 1.0
    assert r.precision == 0.5


def test_tab_f1_missed_gold_lowers_recall():
    gold = [_t(0.0, 5, 5), _t(0.5, 5, 7)]
    pred = [_t(0.0, 5, 5)]
    r = tab_f1(pred, gold)
    assert r.true_positives == 1
    assert r.false_positives == 0
    assert r.false_negatives == 1
    assert r.precision == 1.0
    assert r.recall == 0.5


def test_tab_f1_onset_outside_tolerance_is_a_miss():
    gold = [_t(0.0, 5, 5)]
    pred = [_t(0.10, 5, 5)]  # 100 ms off, tolerance 50 ms
    r = tab_f1(pred, gold)
    assert r.true_positives == 0
    assert r.false_positives == 1
    assert r.false_negatives == 1


def test_tab_f1_wrong_string_or_fret_is_a_miss():
    gold = [_t(0.0, 5, 5)]
    wrong_string = [_t(0.0, 4, 5)]
    wrong_fret = [_t(0.0, 5, 6)]
    assert tab_f1(wrong_string, gold).true_positives == 0
    assert tab_f1(wrong_fret, gold).true_positives == 0


def test_tab_f1_each_gold_matches_at_most_one_predicted():
    """A duplicated predicted event should not double-count against the
    same gold event — the second one is a false positive."""
    gold = [_t(0.0, 5, 5)]
    pred = [_t(0.0, 5, 5), _t(0.01, 5, 5)]  # both within tolerance
    r = tab_f1(pred, gold)
    assert r.true_positives == 1
    assert r.false_positives == 1


# ---------- chord_instance_accuracy ----------


def test_chord_accuracy_perfect_chord_matches():
    gold = [_t(0.0, 5, 0), _t(0.0, 4, 1), _t(0.0, 3, 0)]
    pred = [_t(0.0, 5, 0), _t(0.0, 4, 1), _t(0.0, 3, 0)]
    r = chord_instance_accuracy(pred, gold)
    assert r.accuracy == 1.0
    assert r.matched_chords == 1
    assert r.total_chords == 1


def test_chord_accuracy_wrong_position_in_chord_misses():
    gold = [_t(0.0, 5, 0), _t(0.0, 4, 1), _t(0.0, 3, 0)]
    pred = [_t(0.0, 5, 0), _t(0.0, 4, 1), _t(0.0, 3, 7)]  # one wrong
    r = chord_instance_accuracy(pred, gold)
    assert r.matched_chords == 0
    assert r.total_chords == 1


def test_chord_accuracy_size_mismatch_misses():
    gold = [_t(0.0, 5, 0), _t(0.0, 4, 1), _t(0.0, 3, 0)]
    pred = [_t(0.0, 5, 0), _t(0.0, 4, 1)]  # missing one note
    r = chord_instance_accuracy(pred, gold)
    assert r.matched_chords == 0


def test_chord_accuracy_separates_clusters_by_gap():
    """Two well-separated gold chords should both score independently."""
    gold = [
        _t(0.0, 5, 0), _t(0.0, 4, 1),
        _t(2.0, 5, 7), _t(2.0, 4, 8),
    ]
    pred = [
        _t(0.0, 5, 0), _t(0.0, 4, 1),
        _t(2.0, 5, 7), _t(2.0, 4, 8),
    ]
    r = chord_instance_accuracy(pred, gold)
    assert r.total_chords == 2
    assert r.matched_chords == 2


def test_chord_accuracy_empty_gold_yields_zero():
    r = chord_instance_accuracy([], [])
    assert r.total_chords == 0
    assert r.accuracy == 0.0
