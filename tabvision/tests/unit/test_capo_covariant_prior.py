"""Tests for the capo-covariant position-prior transform.

The transform is pure index arithmetic, and getting it wrong would leave a
plausible-looking prior that silently favours the wrong strings. So the shift
is asserted cell-by-cell against a prior with a known peak, along with the
two invariants that make it safe to apply: capo 0 is identity, and positions
below the capo carry no mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.fusion.position_prior import PitchPositionPrior, capo_covariant_prior
from tabvision.types import GuitarConfig

CFG = GuitarConfig()


def _prior(peak_string: int = 2, peak_fret: int = 5) -> PitchPositionPrior:
    by_pitch = {}
    for pitch in range(40, 90):
        matrix = np.zeros((CFG.n_strings, CFG.max_fret + 1), dtype=np.float64)
        matrix[peak_string, peak_fret] = 1.0
        by_pitch[pitch] = matrix
    return PitchPositionPrior(by_pitch=by_pitch)


def test_capo_zero_is_identity() -> None:
    prior = _prior()
    assert capo_covariant_prior(prior, 0) is prior


def test_peak_moves_up_by_the_capo_on_both_axes() -> None:
    shifted = capo_covariant_prior(_prior(peak_string=2, peak_fret=5), 3)
    # capo-0 mass at (pitch 64, fret 5) must land at (pitch 67, fret 8).
    matrix = shifted.matrix_for_pitch(67)
    assert matrix is not None
    assert matrix[2, 8] == pytest.approx(1.0)
    assert matrix[2, 5] == pytest.approx(0.0)


def test_positions_below_the_capo_have_no_mass() -> None:
    shifted = capo_covariant_prior(_prior(peak_string=0, peak_fret=0), 4)
    matrix = shifted.matrix_for_pitch(48)
    assert matrix is not None
    assert np.all(matrix[:, :4] == 0.0)


def test_open_string_mass_lands_on_the_capo() -> None:
    # An open string at capo 0 is played at the capo itself, not fret 0.
    shifted = capo_covariant_prior(_prior(peak_string=5, peak_fret=0), 2)
    matrix = shifted.matrix_for_pitch(66)
    assert matrix is not None
    assert matrix[5, 2] == pytest.approx(1.0)


def test_total_mass_is_preserved_for_reachable_positions() -> None:
    prior = _prior(peak_string=3, peak_fret=7)
    shifted = capo_covariant_prior(prior, 5)
    source = prior.matrix_for_pitch(60)
    target = shifted.matrix_for_pitch(65)
    assert source is not None and target is not None
    assert target.sum() == pytest.approx(source.sum())


def test_negative_capo_is_rejected() -> None:
    with pytest.raises(ValueError):
        capo_covariant_prior(_prior(), -1)
