"""Shared pytest options for phase acceptance commands."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ablation",
        action="store_true",
        default=False,
        help="accepted by phase eval commands that perform built-in ablation sweeps",
    )
