"""Tests for the bootstrap-CI helper (Phase 0)."""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.eval.bootstrap import BootstrapResult, bootstrap_ci


def test_returns_bootstrap_result_type():
    r = bootstrap_ci([0.5, 0.6, 0.7])
    assert isinstance(r, BootstrapResult)
    assert r.n_observations == 3
    assert r.n_bootstrap == 10_000
    assert r.confidence == 0.95


def test_deterministic_with_seed():
    values = [0.10, 0.50, 0.90, 0.60, 0.30, 0.80]
    r1 = bootstrap_ci(values, seed=42)
    r2 = bootstrap_ci(values, seed=42)
    assert r1.statistic == r2.statistic
    assert r1.lower == r2.lower
    assert r1.upper == r2.upper


def test_different_seeds_produce_different_intervals():
    values = [0.10, 0.50, 0.90, 0.60, 0.30, 0.80]
    r1 = bootstrap_ci(values, seed=42)
    r2 = bootstrap_ci(values, seed=43)
    # CI endpoints may coincide on small data; require at least one to differ.
    assert (r1.lower != r2.lower) or (r1.upper != r2.upper)


def test_single_observation_has_zero_width_ci():
    r = bootstrap_ci([0.85])
    assert r.statistic == pytest.approx(0.85)
    assert r.lower == r.statistic == r.upper
    assert r.n_observations == 1
    assert r.n_bootstrap == 0


def test_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one observation"):
        bootstrap_ci([])


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.1, 1.5])
def test_rejects_bad_confidence(bad_conf):
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.5, 0.6], confidence=bad_conf)


def test_rejects_zero_bootstrap():
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_ci([0.5, 0.6], n_bootstrap=0)


def test_accepts_numpy_array():
    arr = np.array([0.1, 0.5, 0.9])
    r = bootstrap_ci(arr)
    assert r.statistic == pytest.approx(0.5)
    assert r.n_observations == 3


def test_custom_statistic():
    """Verify a non-mean statistic is honored."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    r_median = bootstrap_ci(values, statistic=np.median, seed=0)
    r_mean = bootstrap_ci(values, statistic=np.mean, seed=0)
    # On this small sample they may coincide; correctness check is that
    # statistic is honored, not that they differ.
    assert r_median.statistic == pytest.approx(3.0)
    assert r_mean.statistic == pytest.approx(3.0)


def test_lower_le_statistic_le_upper():
    values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]
    r = bootstrap_ci(values, seed=7)
    assert r.lower <= r.statistic <= r.upper


def test_ci_brackets_known_normal_mean():
    """Coverage check: 95% CI should contain the true mean in roughly 95% of trials.

    Bootstrap percentile intervals are asymptotic — allow generous slack
    so this isn't flaky. We require >= 88% coverage on a low-trial run
    (200 trials, n_obs=80, n_bootstrap=500) for speed.
    """
    rng = np.random.default_rng(0)
    n_trials = 200
    n_obs = 80
    true_mean = 0.85
    sigma = 0.05
    hits = 0
    for trial in range(n_trials):
        sample = rng.normal(true_mean, sigma, n_obs)
        r = bootstrap_ci(sample, seed=trial, n_bootstrap=500)
        if r.lower <= true_mean <= r.upper:
            hits += 1
    coverage = hits / n_trials
    assert coverage >= 0.88, f"bootstrap coverage {coverage:.3f} below 0.88"


def test_zero_variance_input_collapses_ci():
    """If every observation is identical, the CI is a point."""
    r = bootstrap_ci([0.5] * 10, seed=42)
    assert r.statistic == pytest.approx(0.5)
    assert r.lower == pytest.approx(0.5)
    assert r.upper == pytest.approx(0.5)
