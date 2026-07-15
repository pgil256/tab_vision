"""Composition of independent string/fret evidence over playable candidates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig


def combine_candidate_evidence(
    pitch_midi: int,
    cfg: GuitarConfig,
    evidence: Mapping[str, tuple[np.ndarray | None, float]],
) -> np.ndarray | None:
    """Combine candidate distributions as a weighted product of experts.

    Missing, zero-weight, uniform, and low-information sources are neutral.
    Values outside the playable candidate set are ignored. A single informative
    unit-weight source is returned unchanged when already normalized, preserving
    the accepted corpus-prior decode exactly.
    """

    candidates = candidate_positions(pitch_midi, cfg)
    if not candidates:
        return None
    shape = (cfg.n_strings, cfg.max_fret + 1)
    informative: list[tuple[np.ndarray, float]] = []
    for matrix, weight in evidence.values():
        if matrix is None or weight == 0.0:
            continue
        if weight < 0.0:
            raise ValueError("evidence weights must be non-negative")
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"evidence matrix must have shape {shape}, got {array.shape}")
        values = np.asarray([array[item.string_idx, item.fret] for item in candidates])
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("candidate evidence must be finite and non-negative")
        total = float(values.sum())
        if total <= 0.0:
            continue
        probabilities = values / total
        if float(np.max(probabilities) - np.min(probabilities)) <= 1e-12:
            continue
        informative.append((array, weight))

    if not informative:
        return None
    if len(informative) == 1 and informative[0][1] == 1.0:
        matrix = informative[0][0]
        candidate_total = sum(matrix[item.string_idx, item.fret] for item in candidates)
        impossible_total = float(matrix.sum()) - float(candidate_total)
        if abs(float(candidate_total) - 1.0) <= 1e-12 and abs(impossible_total) <= 1e-12:
            return matrix

    log_scores = np.zeros(len(candidates), dtype=np.float64)
    for matrix, weight in informative:
        values = np.asarray([matrix[item.string_idx, item.fret] for item in candidates])
        values = values / float(values.sum())
        log_scores += weight * np.log(np.maximum(values, 1e-12))
    log_scores -= float(np.max(log_scores))
    probabilities = np.exp(log_scores)
    probabilities /= float(probabilities.sum())
    combined = np.zeros(shape, dtype=np.float64)
    for candidate, probability in zip(candidates, probabilities, strict=True):
        combined[candidate.string_idx, candidate.fret] = float(probability)
    return combined


__all__ = ["combine_candidate_evidence"]
