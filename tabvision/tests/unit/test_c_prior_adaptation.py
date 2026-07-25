"""Tests for the Track C prior-blending helper.

The blend is the whole mechanism of the adaptive arms, so its two non-obvious
properties are pinned: it must stay a probability distribution (an unnormalised
blend would silently change the evidence weight, not just its shape), and a
pitch the session never saw must keep the population prior *unchanged* rather
than being pulled toward a flat or empty one.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.c_prior_adaptation import blend_priors
from tabvision.fusion.position_prior import PitchPositionPrior


def _prior(pitch: int, cells: dict[tuple[int, int], float]) -> PitchPositionPrior:
    matrix = np.zeros((6, 25), dtype=np.float64)
    for (string, fret), value in cells.items():
        matrix[string, fret] = value
    total = matrix.sum()
    if total:
        matrix /= total
    return PitchPositionPrior(by_pitch={pitch: matrix})


def test_blend_stays_normalised() -> None:
    base = _prior(64, {(1, 0): 0.75, (2, 5): 0.25})
    session = _prior(64, {(1, 0): 0.1, (2, 5): 0.9})
    for weight in (0.0, 0.15, 0.5, 1.0):
        blended = blend_priors(base, session, weight)
        matrix = blended.matrix_for_pitch(64)
        assert matrix is not None
        assert np.isclose(matrix.sum(), 1.0)


def test_blend_endpoints_are_the_inputs() -> None:
    base = _prior(64, {(1, 0): 0.75, (2, 5): 0.25})
    session = _prior(64, {(1, 0): 0.1, (2, 5): 0.9})
    at_zero = blend_priors(base, session, 0.0).matrix_for_pitch(64)
    at_one = blend_priors(base, session, 1.0).matrix_for_pitch(64)
    assert np.allclose(at_zero, base.matrix_for_pitch(64))
    assert np.allclose(at_one, session.matrix_for_pitch(64))


def test_blend_moves_toward_the_session() -> None:
    base = _prior(64, {(1, 0): 0.9, (2, 5): 0.1})
    session = _prior(64, {(1, 0): 0.1, (2, 5): 0.9})
    blended = blend_priors(base, session, 0.5).matrix_for_pitch(64)
    assert blended is not None
    # The session favours (2,5); the blend must move that way but not all the way.
    assert base.matrix_for_pitch(64)[2, 5] < blended[2, 5] < session.matrix_for_pitch(64)[2, 5]


def test_pitches_the_session_never_saw_are_untouched() -> None:
    """A session covering one pitch must not disturb the prior for others."""
    base_matrix = np.zeros((6, 25), dtype=np.float64)
    base_matrix[1, 0] = 0.6
    base_matrix[2, 5] = 0.4
    other = np.zeros((6, 25), dtype=np.float64)
    other[3, 2] = 1.0
    base = PitchPositionPrior(by_pitch={64: base_matrix, 67: other})
    session = _prior(64, {(2, 5): 1.0})

    blended = blend_priors(base, session, 0.5)
    assert np.allclose(blended.matrix_for_pitch(67), other)
    assert not np.allclose(blended.matrix_for_pitch(64), base_matrix)
