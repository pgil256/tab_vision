"""Fixed learned review queue used by the Phase 6 offline gate."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

FEATURE_COUNT: Final = 10
TRAIN_EPOCHS: Final = 40
BATCH_SIZE: Final = 512
LEARNING_RATE: Final = 3.0e-3
WEIGHT_DECAY: Final = 1.0e-4
PLATT_L2: Final = 1.0
PLATT_ITERATIONS: Final = 50


class ReviewQueueNet(nn.Module):
    """Tiny fixed MLP producing a wrong-position risk logit."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"expected feature shape (batch, {FEATURE_COUNT}), got {tuple(features.shape)}"
            )
        return self.layers(features).squeeze(1)


@dataclass(frozen=True)
class FittedReviewModel:
    network: ReviewQueueNet
    mean: np.ndarray
    scale: np.ndarray
    history: tuple[float, ...]

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = _validate_features(features)
        standardized = (values - self.mean) / self.scale
        self.network.eval()
        with torch.inference_mode():
            return self.network(torch.from_numpy(standardized.astype(np.float32))).numpy()

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.mean, dtype="<f8").tobytes())
        digest.update(np.asarray(self.scale, dtype="<f8").tobytes())
        for name, tensor in sorted(self.network.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().numpy().astype("<f4", copy=False).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class PlattCalibrator:
    intercept: float
    slope: float

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        values = self.intercept + self.slope * np.asarray(logits, dtype=np.float64)
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))


@dataclass(frozen=True)
class BudgetMetrics:
    fraction: float
    reviewed: int
    wrong_reviewed: int
    precision: float
    recall: float


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fit_review_model(
    features: np.ndarray,
    wrong_labels: np.ndarray,
    *,
    seed: int,
) -> FittedReviewModel:
    values = _validate_features(features)
    labels = _validate_labels(wrong_labels, len(values))
    _seed_everything(seed)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = ((values - mean) / scale).astype(np.float32)
    network = ReviewQueueNet()
    if parameter_count(network) >= 500:
        raise AssertionError("fixed review detector exceeds the 500-parameter cap")
    wrong = int(np.sum(labels))
    correct = len(labels) - wrong
    if not wrong or not correct:
        raise ValueError("review model requires both correct and wrong examples")
    positive_weight = torch.as_tensor(correct / wrong, dtype=torch.float32)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tensor_features = torch.from_numpy(standardized)
    tensor_labels = torch.from_numpy(labels.astype(np.float32))
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for _epoch in range(TRAIN_EPOCHS):
        order = rng.permutation(len(labels))
        losses: list[float] = []
        network.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            logits = network(tensor_features[indices])
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                tensor_labels[indices],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    network.eval()
    return FittedReviewModel(network, mean, scale, tuple(history))


def fit_platt(logits: np.ndarray, wrong_labels: np.ndarray) -> PlattCalibrator:
    values = np.asarray(logits, dtype=np.float64)
    labels = _validate_labels(wrong_labels, len(values)).astype(np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("Platt logits must be one finite vector")
    prevalence = float(np.clip(np.mean(labels), 1.0e-6, 1.0 - 1.0e-6))
    weights = np.asarray((math.log(prevalence / (1.0 - prevalence)), 1.0), dtype=np.float64)
    design = np.column_stack((np.ones(len(values)), values))
    penalty = np.diag((0.0, PLATT_L2))
    for _iteration in range(PLATT_ITERATIONS):
        linear = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        variance = np.maximum(probabilities * (1.0 - probabilities), 1.0e-6)
        gradient = design.T @ (probabilities - labels) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1.0e-9:
            break
    return PlattCalibrator(float(weights[0]), float(weights[1]))


def binary_roc_auc(risk: np.ndarray, wrong_labels: np.ndarray) -> float:
    """Tie-aware ROC AUC where higher risk should indicate a wrong position."""

    scores = np.asarray(risk, dtype=np.float64)
    labels = _validate_labels(wrong_labels, len(scores))
    if scores.ndim != 1 or np.any(~np.isfinite(scores)):
        raise ValueError("risk must be one finite vector")
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def budget_metrics(
    risk: np.ndarray,
    wrong_labels: np.ndarray,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> tuple[BudgetMetrics, ...]:
    scores = np.asarray(risk, dtype=np.float64)
    labels = _validate_labels(wrong_labels, len(scores))
    order = np.argsort(-scores, kind="stable")
    wrong_total = int(np.sum(labels))
    output: list[BudgetMetrics] = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("review fractions must lie in (0, 1]")
        reviewed = max(1, int(math.ceil(len(labels) * fraction)))
        wrong_reviewed = int(np.sum(labels[order[:reviewed]]))
        output.append(
            BudgetMetrics(
                fraction,
                reviewed,
                wrong_reviewed,
                wrong_reviewed / reviewed,
                wrong_reviewed / wrong_total if wrong_total else 0.0,
            )
        )
    return tuple(output)


def _validate_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape (rows, {FEATURE_COUNT})")
    if not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("features must be non-empty and finite")
    return values


def _validate_labels(labels: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.bool_)
    if values.shape != (rows,):
        raise ValueError("labels must contain one value per row")
    return values


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


__all__ = [
    "BATCH_SIZE",
    "BudgetMetrics",
    "FEATURE_COUNT",
    "FittedReviewModel",
    "PlattCalibrator",
    "ReviewQueueNet",
    "TRAIN_EPOCHS",
    "binary_roc_auc",
    "budget_metrics",
    "fit_platt",
    "fit_review_model",
    "parameter_count",
]
