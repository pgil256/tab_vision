"""Tests for the Q7 capo-covariant prior transform.

The one load-bearing claim is the index arithmetic: a capo-``C`` session must
read the capo-0 prior ``C`` frets and ``C`` semitones lower. Get that wrong
and the probe's whole comparison is meaningless, so it is asserted directly
against a prior with a known, asymmetric shape.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.q7_capo_covariant_probe import _covariant_score, _naive_score
from tabvision.fusion.position_prior import PitchPositionPrior
from tabvision.types import GuitarConfig

CFG = GuitarConfig()


def _prior_with_known_cell() -> PitchPositionPrior:
    """A prior where pitch P puts all its mass on (string 2, fret 5)."""
    by_pitch = {}
    for pitch in range(40, 90):
        matrix = np.full((CFG.n_strings, CFG.max_fret + 1), 1e-6, dtype=np.float64)
        matrix[2, 5] = 1.0
        by_pitch[pitch] = matrix / matrix.sum()
    return PitchPositionPrior(by_pitch=by_pitch)


def test_capo_zero_is_a_no_op() -> None:
    prior = _prior_with_known_cell()
    for string in range(6):
        for fret in range(6):
            cov = _covariant_score(prior, 64, string, fret, 0)
            naive = _naive_score(prior, 64, string, fret)
            assert cov == naive


def test_covariant_reads_the_shifted_cell() -> None:
    prior = _prior_with_known_cell()
    # Under capo 3, a note sounding pitch 67 on (string 2, fret 8) has
    # relative position (2, 5) at the capo-0 pitch 64 — the loaded cell.
    hot = _covariant_score(prior, 67, 2, 8, 3)
    cold = _covariant_score(prior, 67, 2, 7, 3)  # relative fret 4, not the peak
    assert hot is not None and cold is not None
    assert hot > cold


def test_naive_ignores_the_capo_and_reads_the_wrong_cell() -> None:
    prior = _prior_with_known_cell()
    # Same note as above, but naive looks up (pitch 67, fret 8) directly. The
    # prior's mass for pitch 67 is at fret 5, so fret 8 is not the peak —
    # the capo-ignorant lookup misses.
    peak = _naive_score(prior, 67, 2, 5)
    off = _naive_score(prior, 67, 2, 8)
    assert peak is not None and off is not None
    assert peak > off


def test_out_of_range_shift_returns_none() -> None:
    prior = _prior_with_known_cell()
    # fret - capo negative is unreachable, and a pitch below the learned
    # range has no matrix — both must decline rather than fabricate a score.
    assert _covariant_score(prior, 41, 0, 0, 5) is None
    assert _covariant_score(prior, 30, 0, 10, 2) is None
