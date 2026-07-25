"""Tests for the residual out-parameter on ``decompose_errors``.

The instrumentation exists so Track D can ask *what kind* of note goes missing
without reimplementing the matcher. Two properties make it trustworthy: the
residuals must agree with the counts the matcher already returns (otherwise the
diagnosis describes a different matching than the one that is scored), and
passing no ``Residuals`` must change nothing.
"""

from __future__ import annotations

from tabvision.eval.error_decomposition import Residuals, decompose_errors
from tabvision.types import TabEvent


def _event(pitch: int, string_idx: int, fret: int, onset: float) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def test_residuals_agree_with_the_counts() -> None:
    gold = [
        _event(64, 1, 0, 0.0),  # matched exactly
        _event(67, 2, 3, 1.0),  # missed entirely
        _event(60, 3, 5, 2.0),  # matched, wrong position
    ]
    predicted = [
        _event(64, 1, 0, 0.0),
        _event(60, 4, 10, 2.0),
        _event(72, 0, 12, 5.0),  # spurious, far from anything
    ]
    residuals = Residuals()
    result = decompose_errors(predicted, gold, residuals=residuals)

    assert result.correct == 1
    assert result.wrong_position_same_pitch == 1
    assert result.missed_onset == 1
    assert result.extra_detection == 1

    # The residuals must be the *same* events those counts refer to.
    assert len(residuals.missed) == result.missed_onset
    assert len(residuals.extra) == result.extra_detection
    assert residuals.missed[0].pitch_midi == 67
    assert residuals.extra[0].pitch_midi == 72


def test_residuals_are_empty_on_a_perfect_decode() -> None:
    gold = [_event(64, 1, 0, 0.0), _event(67, 2, 3, 1.0)]
    residuals = Residuals()
    result = decompose_errors(list(gold), gold, residuals=residuals)
    assert result.correct == 2
    assert residuals.missed == ()
    assert residuals.extra == ()


def test_omitting_residuals_changes_nothing() -> None:
    gold = [_event(64, 1, 0, 0.0), _event(67, 2, 3, 1.0)]
    predicted = [_event(64, 1, 0, 0.0), _event(72, 0, 12, 9.0)]
    with_out = decompose_errors(predicted, gold, residuals=Residuals())
    without = decompose_errors(predicted, gold)
    assert with_out == without


def test_a_fresh_residuals_instance_starts_empty() -> None:
    """Guards against the mutable-default trap on the dataclass."""
    first = Residuals()
    decompose_errors([], [_event(64, 1, 0, 0.0)], residuals=first)
    assert len(first.missed) == 1
    second = Residuals()
    assert second.missed == ()
    assert second.extra == ()
