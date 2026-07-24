"""Tests for the Q6 inharmonicity separability precursor.

The precursor's conclusion — that the physics is separable in principle —
rests on a Gaussian mean-separation argument, so the two properties that
carry it are asserted directly: the decision improves as the estimator gets
sharper, and a zero-separation pair is a coin flip.
"""

from __future__ import annotations

import math

import pytest

from scripts.eval.q6_separability_precursor import LOG2, normal_cdf


def _accuracy(fret_gap: int, sigma: float) -> float:
    return normal_cdf((fret_gap / 6.0) * LOG2 / (2.0 * sigma))


def test_normal_cdf_matches_known_points() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_identical_positions_are_a_coin_flip() -> None:
    # No fret difference means no length-driven B difference to exploit.
    assert _accuracy(0, 0.10) == pytest.approx(0.5)


def test_sharper_estimator_decides_better() -> None:
    accuracies = [_accuracy(5, sigma) for sigma in (0.30, 0.20, 0.10, 0.05)]
    assert accuracies == sorted(accuracies)
    assert accuracies[-1] > 0.99


def test_wider_fret_gaps_are_easier() -> None:
    accuracies = [_accuracy(gap, 0.20) for gap in (4, 5, 9, 14)]
    assert accuracies == sorted(accuracies)


def test_length_term_ratio_is_the_documented_physics() -> None:
    # B ∝ 1/L² and fretting shortens L by 2^(-n/12), so B ∝ 2^(n/6):
    # a 5-fret gap is a 1.78x ratio before any plain-vs-wound difference.
    assert 2 ** (5 / 6.0) == pytest.approx(1.7818, abs=1e-4)
    assert math.isclose(LOG2, math.log(2.0))
