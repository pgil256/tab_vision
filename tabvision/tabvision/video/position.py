"""Implementation-only coarse fret-position observations.

This module deliberately lives outside :mod:`tabvision.types`: the records
below connect the current FretCam implementation to pipeline orchestration
without changing the immutable SPEC section-8 contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

PositionObservationState = Literal["locked", "holding"]


@dataclass(frozen=True)
class PositionWindowObservation:
    """One confidence-gated fret-position window on the media timeline."""

    timestamp_s: float
    position: int
    window_frets: tuple[int, ...]
    confidence: float
    state: PositionObservationState


class PositionAnalyzer(Protocol):
    """Analyze timestamped BGR frames into stable coarse fret windows."""

    def analyze(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        stride: int = 1,
    ) -> list[PositionWindowObservation]: ...


__all__ = [
    "PositionAnalyzer",
    "PositionObservationState",
    "PositionWindowObservation",
]
