"""Tests for the N3 physics-review probe's scoring primitives.

The AUC statistic and the decoder-margin parser are the two pieces the
probe's conclusion rests on; if either is wrong the whole comparison is
meaningless.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scripts.eval.n3_physics_review_probe import _auc, _decoder_margin

# _auc defers to scipy.stats.rankdata for midranks, and scipy ships in the
# audio-baseline extra rather than the [dev] extra CI installs. Marking only
# the AUC tests keeps the _decoder_margin coverage running everywhere.
requires_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="_auc needs scipy.stats.rankdata; scipy is in the audio-baseline extra.",
)


@requires_scipy
def test_auc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    assert _auc(scores, labels) == 1.0


@requires_scipy
def test_auc_inverted_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    assert _auc(scores, labels) == 0.0


@requires_scipy
def test_auc_orders_correctly_with_distinct_scores() -> None:
    # pos at 0.2 and 0.4 beat 3 of 4 negatives -> 0.75, exercising the rank
    # arithmetic on a non-degenerate arrangement.
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    assert _auc(scores, labels) == 0.75


@requires_scipy
def test_auc_all_ties_is_half() -> None:
    # All-equal scores must give 0.5 via midranks, not an order-dependent
    # value — this is the tie-correction the naive rank version got wrong.
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    assert _auc(scores, labels) == 0.5


@requires_scipy
def test_auc_single_class_is_nan() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    assert np.isnan(_auc(scores, np.zeros(3)))


def test_decoder_margin_reads_the_rank2_cost() -> None:
    # candidate_path is "string:fret:cost_delta" in rank order; the rank-1
    # delta is 0 and the margin is the rank-2 delta.
    assert _decoder_margin("2:9:0.0;1:14:0.42;3:4:1.1") == 0.42


def test_decoder_margin_unambiguous_is_large() -> None:
    assert _decoder_margin("2:9:0.0") == 10.0
