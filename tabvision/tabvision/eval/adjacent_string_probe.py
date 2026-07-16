"""Linear adjacent-string ranking primitives for the sequential Phase 4 probe."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalancedLogisticModel:
    """One deterministic, class-balanced L2 logistic model."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("logistic feature shape does not match the fitted model")
        standardized = (values - self.mean) / self.scale
        return self.weights[0] + standardized @ self.weights[1:]

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.mean, self.scale, self.weights):
            digest.update(np.asarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()


def fit_balanced_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
    iterations: int = 50,
) -> BalancedLogisticModel:
    """Fit fixed class-balanced logistic regression with Newton steps."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("logistic training data must be a non-empty feature matrix")
    if np.any(~np.isfinite(x)) or np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("logistic training data must be finite with binary labels")
    if l2 < 0.0 or iterations < 1:
        raise ValueError("logistic l2 and iterations must be non-negative/positive")
    negative = int(np.sum(y == 0.0))
    positive = int(np.sum(y == 1.0))
    if not negative or not positive:
        raise ValueError("balanced logistic training requires both classes")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-8] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    sample_weights = np.where(y == 1.0, len(x) / (2.0 * positive), len(x) / (2.0 * negative))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probabilities * (1.0 - probabilities), 1.0e-6)
        weighted_residual = sample_weights * (probabilities - y)
        gradient = design.T @ weighted_residual + penalty @ weights
        curvature = sample_weights * variance
        hessian = design.T @ (design * curvature[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
    return BalancedLogisticModel(mean, scale, weights)


def adjacent_gold_pairs(
    candidate_strings: Iterable[int],
    gold_string: int,
) -> tuple[tuple[int, int, bool], ...]:
    """Return physically adjacent gold-vs-alternative training pairs.

    The boolean label is true when the higher-numbered string is gold.
    """

    candidates = {int(value) for value in candidate_strings}
    if int(gold_string) not in candidates:
        raise ValueError("gold string is not in the candidate set")
    pairs = []
    for alternative in sorted(candidates):
        if abs(alternative - int(gold_string)) != 1:
            continue
        lower = min(alternative, int(gold_string))
        higher = max(alternative, int(gold_string))
        pairs.append((lower, higher, int(gold_string) == higher))
    return tuple(pairs)


def adjacent_candidate_pairs(candidate_strings: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return all physically adjacent pairs present in one candidate set."""

    candidates = sorted({int(value) for value in candidate_strings})
    return tuple(
        (lower, higher)
        for lower, higher in zip(candidates, candidates[1:], strict=False)
        if higher - lower == 1
    )


def candidate_potentials(
    candidate_strings: Iterable[int],
    edge_logits: dict[tuple[int, int], float],
) -> dict[int, float]:
    """Integrate adjacent log-odds into per-candidate Bradley-Terry potentials."""

    candidates = sorted({int(value) for value in candidate_strings})
    potentials: dict[int, float] = {}
    previous: int | None = None
    for string_idx in candidates:
        if previous is None or string_idx - previous != 1:
            potentials[string_idx] = 0.0
        else:
            key = (previous, string_idx)
            value = float(edge_logits.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError("adjacent edge logits must be finite")
            potentials[string_idx] = potentials[previous] + value
        previous = string_idx
    return potentials


__all__ = [
    "BalancedLogisticModel",
    "adjacent_candidate_pairs",
    "adjacent_gold_pairs",
    "candidate_potentials",
    "fit_balanced_logistic",
]
