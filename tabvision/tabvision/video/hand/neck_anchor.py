"""Coarse fretting-hand neck-region signal.

Exact fingertip-to-string/fret posteriors are useful, but the most robust
first-order video signal is simpler: where along the neck is the fretting
hand? This module projects MediaPipe hand landmarks through the fretboard
homography and summarizes the occupied fret region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabvision.types import GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS, HandSample


@dataclass(frozen=True)
class HandNeckAnchor:
    """Coarse visual prior over the active fretboard region."""

    center_fret: float
    min_fret: float
    max_fret: float
    confidence: float
    method: str = "mediapipe_homography"

    def contains(self, fret: int | float, *, margin: float = 1.0) -> bool:
        """Return whether ``fret`` falls inside the anchor span plus margin."""
        return (self.min_fret - margin) <= float(fret) <= (self.max_fret + margin)


@dataclass(frozen=True)
class NeckAnchorConfig:
    """Tunable constants for converting landmarks into a fret-region prior."""

    include_wrist: bool = True
    fret_margin: float = 1.0
    min_confidence: float = 0.0


def compute_neck_anchor(
    hand: HandSample | None,
    H: Homography,  # noqa: N803
    cfg: GuitarConfig,
    anchor_cfg: NeckAnchorConfig | None = None,
) -> HandNeckAnchor:
    """Estimate which neck region the fretting hand occupies.

    ``Homography.H`` maps canonical fretboard coordinates to image pixels, so
    this function projects hand landmarks through ``H^-1``. Canonical ``x`` is
    converted to fret coordinates over ``[0, cfg.max_fret]``.
    """
    acfg = anchor_cfg or NeckAnchorConfig()
    if hand is None or H.confidence <= 0.0:
        return _empty_anchor()

    try:
        H_inv = np.linalg.inv(H.H)  # noqa: N806
    except np.linalg.LinAlgError:
        return _empty_anchor()

    xs: list[float] = []
    if acfg.include_wrist:
        xs.append(_project_x(H_inv, *hand.wrist_xy))
    for name in FRETTING_FINGERS:
        finger = hand.fingers.get(name)
        if finger is not None:
            xs.append(_project_x(H_inv, *finger.tip_xy))

    if not xs:
        return _empty_anchor()

    fret_positions = np.clip(np.array(xs, dtype=np.float64), 0.0, 1.0) * cfg.max_fret
    raw_min = float(fret_positions.min())
    raw_max = float(fret_positions.max())
    center = float(np.median(fret_positions))
    min_fret = max(0.0, raw_min - acfg.fret_margin)
    max_fret = min(float(cfg.max_fret), raw_max + acfg.fret_margin)

    spread = max(0.0, raw_max - raw_min)
    span_penalty = min(0.5, spread / max(float(cfg.max_fret), 1.0))
    confidence = max(
        acfg.min_confidence,
        min(1.0, float(hand.confidence) * float(H.confidence) * (1.0 - span_penalty)),
    )

    return HandNeckAnchor(
        center_fret=center,
        min_fret=min_fret,
        max_fret=max_fret,
        confidence=confidence,
    )


def _empty_anchor() -> HandNeckAnchor:
    return HandNeckAnchor(
        center_fret=0.0,
        min_fret=0.0,
        max_fret=0.0,
        confidence=0.0,
    )


def _project_x(H_inv: np.ndarray, x: float, y: float) -> float:  # noqa: N803
    pt = np.array([x, y, 1.0], dtype=np.float64)
    proj = H_inv @ pt
    if abs(proj[2]) < 1e-12:
        return 0.0
    return float(proj[0] / proj[2])


__all__ = [
    "HandNeckAnchor",
    "NeckAnchorConfig",
    "compute_neck_anchor",
]
