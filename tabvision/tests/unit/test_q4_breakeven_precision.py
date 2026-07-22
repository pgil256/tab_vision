"""Tests for the second-opinion bench's derived leg-2 threshold.

The formula decides whether a merge candidate is worth building, so its
qualitative behaviour has to be right: a purer stream demands purer
additions, a better decoder demands less, and the closed form at perfect
assignment must fall out exactly.
"""

from __future__ import annotations

import pytest

from scripts.eval.q4_breakeven_precision import breakeven_precision, measured_alpha


def test_perfect_assignment_reduces_to_half_the_f1() -> None:
    # alpha = 1: every rescued note also lands on the right string, and the
    # algebra collapses to p > F1/2.
    for f1 in (0.4, 0.6773, 0.9):
        assert breakeven_precision(f1, 1.0) == pytest.approx(f1 / 2.0)


def test_purer_streams_demand_purer_additions() -> None:
    alpha = 0.458
    thresholds = [breakeven_precision(f1, alpha) for f1 in (0.5, 0.7, 0.9)]
    assert thresholds == sorted(thresholds)
    assert all(0.0 < value <= 1.0 for value in thresholds)


def test_better_assignment_lowers_the_bar() -> None:
    f1 = 0.6773
    thresholds = [breakeven_precision(f1, alpha) for alpha in (0.3, 0.6, 0.9)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_matches_the_banked_n2_calibration() -> None:
    # The N2 pilot's own numbers: baseline Tab F1 0.6773, measured alpha
    # 0.4581 -> 0.5278, which is what put the guessed 0.5 gate on a footing.
    assert breakeven_precision(0.6773, 0.4581) == pytest.approx(0.5278, abs=5e-4)


def test_measured_alpha_uses_the_correct_bucket_gain() -> None:
    variants = {
        "ensemble": {"added_true_notes": 0, "decomposition": {"correct": 1000}},
        "cluster": {"added_true_notes": 40, "decomposition": {"correct": 1020}},
        "empty": {"added_true_notes": 0, "decomposition": {"correct": 1000}},
    }
    alphas = measured_alpha(variants)
    assert alphas == {"cluster": 0.5}  # 20 gained / 40 real additions
    assert "empty" not in alphas  # no additions -> alpha is undefined, not 0
