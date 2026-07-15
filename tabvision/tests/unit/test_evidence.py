from __future__ import annotations

import numpy as np
import pytest

from tabvision.fusion.evidence import combine_candidate_evidence
from tabvision.types import GuitarConfig


def _matrix(values: tuple[float, ...]) -> np.ndarray:
    matrix = np.zeros((6, 25), dtype=np.float64)
    for (string_idx, fret), value in zip(
        ((1, 24), (2, 19), (3, 14), (4, 10), (5, 5)), values, strict=True
    ):
        matrix[string_idx, fret] = value
    return matrix


def test_missing_and_uniform_evidence_are_neutral() -> None:
    uniform = _matrix((1, 1, 1, 1, 1))
    assert combine_candidate_evidence(69, GuitarConfig(), {"missing": (None, 1.0)}) is None
    assert combine_candidate_evidence(69, GuitarConfig(), {"uniform": (uniform, 1.0)}) is None


def test_single_normalized_source_is_byte_preserved() -> None:
    source = _matrix((0.5, 0.2, 0.15, 0.1, 0.05))
    combined = combine_candidate_evidence(69, GuitarConfig(), {"corpus": (source, 1.0)})
    assert combined is source


def test_two_sources_use_weighted_product_and_normalize() -> None:
    corpus = _matrix((0.6, 0.1, 0.1, 0.1, 0.1))
    timbre = _matrix((0.1, 0.7, 0.1, 0.05, 0.05))
    combined = combine_candidate_evidence(
        69,
        GuitarConfig(),
        {"corpus": (corpus, 1.0), "timbre": (timbre, 2.0)},
    )
    assert combined is not None
    assert combined.sum() == pytest.approx(1.0)
    assert combined[2, 19] > combined[1, 24]
    assert np.count_nonzero(combined) == 5


def test_invalid_evidence_is_rejected() -> None:
    bad = _matrix((0.5, 0.2, 0.15, 0.1, -0.05))
    with pytest.raises(ValueError, match="finite and non-negative"):
        combine_candidate_evidence(69, GuitarConfig(), {"bad": (bad, 1.0)})
    with pytest.raises(ValueError, match="non-negative"):
        combine_candidate_evidence(
            69, GuitarConfig(), {"bad-weight": (_matrix((1, 2, 3, 4, 5)), -1.0)}
        )
