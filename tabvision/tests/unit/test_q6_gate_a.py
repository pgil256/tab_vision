"""Tests for the Q6 Gate A inharmonicity estimator.

The estimator is the whole experiment: if it cannot recover a known B from a
synthetic stiff string, a failed Gate A would say nothing about the physics.
These synthesise partials from the model itself and check recovery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.eval.q6_gate_a import (
    candidates_for_pitch,
    estimate_inharmonicity,
)

SR = 44100


def _stiff_string(f0: float, b_value: float, *, seconds: float = 0.4, partials: int = 10):
    t = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(t)
    for k in range(1, partials + 1):
        freq = k * f0 * math.sqrt(1.0 + b_value * k * k)
        if freq > SR / 2.5:
            break
        signal += np.sin(2 * np.pi * freq * t) / k
    return signal


@pytest.mark.parametrize("b_true", [5e-5, 1e-4, 5e-4])
def test_recovers_known_inharmonicity(b_true: float) -> None:
    f0 = 110.0
    fitted = estimate_inharmonicity(_stiff_string(f0, b_true), SR, f0)
    assert fitted is not None
    f0_hat, b_hat, partials, r2 = fitted
    assert f0_hat == pytest.approx(f0, rel=0.01)
    # Relative error well inside the ~25% the separability precursor needs.
    assert b_hat == pytest.approx(b_true, rel=0.25)
    assert partials >= 4
    assert r2 > 0.9


def test_distinguishes_two_strings_at_the_same_pitch() -> None:
    # Same pitch, 5 frets apart -> B ratio 2^(5/6) = 1.78x. The estimator has
    # to separate these or Gate A cannot work regardless of the physics.
    f0 = 196.0
    low = estimate_inharmonicity(_stiff_string(f0, 1.0e-4), SR, f0)
    high = estimate_inharmonicity(_stiff_string(f0, 1.78e-4), SR, f0)
    assert low is not None and high is not None
    assert high[1] > low[1] * 1.3


def test_rejects_a_segment_with_no_partials() -> None:
    silence = np.zeros(int(SR * 0.4))
    assert estimate_inharmonicity(silence, SR, 110.0) is None


def test_rejects_a_too_short_segment() -> None:
    assert estimate_inharmonicity(_stiff_string(110.0, 1e-4, seconds=0.05), SR, 110.0) is None


def test_candidates_respect_tuning_and_fret_bound() -> None:
    # E4 (64) is open high-E plus positions on lower strings.
    assert (5, 0) in candidates_for_pitch(64)
    assert (0, 24) in candidates_for_pitch(64)
    # Below the low open E nothing is playable.
    assert candidates_for_pitch(39) == []
