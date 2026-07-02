"""Shared pytest options for phase acceptance commands."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_transition_prior_state():
    """Save/restore the process-global A15 transition-prior install.

    ``run_pipeline`` now installs/clears the sequence prior globally
    (coupled to the position prior), so any test driving the pipeline —
    or calling ``set_transition_prior`` directly — would otherwise leak
    decode-shaping state into later fusion tests."""
    from tabvision.fusion import playability

    prior = playability._TRANSITION_PRIOR
    env_read = playability._TRANSITION_PRIOR_ENV_READ
    weight = playability.TRANSITION_PRIOR_WEIGHT
    yield
    playability._TRANSITION_PRIOR = prior
    playability._TRANSITION_PRIOR_ENV_READ = env_read
    playability.TRANSITION_PRIOR_WEIGHT = weight


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ablation",
        action="store_true",
        default=False,
        help="accepted by phase eval commands that perform built-in ablation sweeps",
    )
