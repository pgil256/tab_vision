from __future__ import annotations

import numpy as np
import pytest

from tabvision.eval.adjacent_string_probe import (
    adjacent_candidate_pairs,
    adjacent_gold_pairs,
    candidate_potentials,
    fit_balanced_logistic,
)


def test_adjacent_pair_construction_uses_only_physical_neighbors() -> None:
    assert adjacent_gold_pairs((0, 1, 2, 4), 1) == ((0, 1, True), (1, 2, False))
    assert adjacent_candidate_pairs((0, 1, 2, 4)) == ((0, 1), (1, 2))


def test_adjacent_pair_construction_requires_gold_candidate() -> None:
    with pytest.raises(ValueError, match="gold string"):
        adjacent_gold_pairs((0, 1), 2)


def test_candidate_potentials_integrate_edges_and_reset_at_gap() -> None:
    actual = candidate_potentials(
        (0, 1, 2, 4, 5),
        {(0, 1): 0.5, (1, 2): -0.25, (4, 5): 1.0},
    )

    assert actual == {0: 0.0, 1: 0.5, 2: 0.25, 4: 0.0, 5: 1.0}


def test_balanced_logistic_is_deterministic_and_learns_separation() -> None:
    features = np.asarray([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])

    first = fit_balanced_logistic(features, labels)
    second = fit_balanced_logistic(features, labels)

    assert first.sha256() == second.sha256()
    assert np.all(first.decision_function(features[:3]) < 0.0)
    assert np.all(first.decision_function(features[3:]) > 0.0)
