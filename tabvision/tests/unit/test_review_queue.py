from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tabvision.eval.review_queue import (  # noqa: E402
    FEATURE_COUNT,
    ReviewQueueNet,
    binary_roc_auc,
    budget_metrics,
    fit_platt,
    fit_review_model,
    parameter_count,
)


def test_fixed_network_is_below_parameter_cap_and_has_expected_shape() -> None:
    model = ReviewQueueNet()
    assert parameter_count(model) < 500
    assert model(torch.zeros((4, FEATURE_COUNT), dtype=torch.float32)).shape == (4,)


def test_auc_and_budget_metrics_are_tie_aware() -> None:
    risk = np.asarray((0.9, 0.8, 0.2, 0.1))
    wrong = np.asarray((True, True, False, False))
    assert binary_roc_auc(risk, wrong) == pytest.approx(1.0)
    budget = budget_metrics(risk, wrong, (0.5,))[0]
    assert budget.precision == pytest.approx(1.0)
    assert budget.recall == pytest.approx(1.0)


def test_platt_probabilities_are_finite_monotonic_and_calibrated_direction() -> None:
    logits = np.asarray((-3.0, -2.0, 2.0, 3.0))
    wrong = np.asarray((False, False, True, True))
    calibrator = fit_platt(logits, wrong)
    probabilities = calibrator.probabilities(logits)
    assert np.all(np.isfinite(probabilities))
    assert np.all(np.diff(probabilities) > 0.0)


def test_model_fit_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(256, FEATURE_COUNT))
    wrong = features[:, 0] + 0.5 * features[:, 1] > 0.0
    first = fit_review_model(features, wrong, seed=9)
    second = fit_review_model(features, wrong, seed=9)
    assert first.sha256() == second.sha256()
    assert first.logits(features) == pytest.approx(second.logits(features))
