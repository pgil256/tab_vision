"""Fingertip → (string, fret) posterior — Phase 4.

Pure-Python core that takes raw MediaPipe-derived landmark samples plus
the per-frame :class:`Homography` and produces a §8
:class:`FrameFingering`.  Splitting this off from
``mediapipe_backend.py`` lets the unit tests exercise the projection +
posterior arithmetic without paying the MediaPipe import cost.

Per ``docs/DECISIONS.md`` 2026-05-05 "Phase 4 entry":

- Distance kernel (Gaussian on canonical fret-cell centroid) drives the
  primary signal.
- Curl prior gates: a curled finger contributes near-zero log-prob
  everywhere — it isn't fretting.
- z-depth prior is a small additive bonus when the fingertip projects
  near the fretboard plane.

Canonical convention (matches ``video.fretboard.keypoint``):

- canonical x ∈ [0, 1], where 0 = nut and 1 = the body end of the
  detected fretboard.  We treat ``x = 1`` as ``fret = max_fret`` and
  divide [0, 1] uniformly into ``max_fret + 1`` cells (cell *k* covers
  ``x ∈ [k / (max_fret+1), (k+1) / (max_fret+1)]``).  This matches the
  uniform-cell partition used elsewhere in the package; the equal-tempered
  rule-of-18 spacing is only used as a *visualization* aid (see
  ``scripts/viz/overlay_fretboard.py``).
- canonical y ∈ [0, 1], where 0 = high-E side and 1 = low-E side.
  ``cfg.tuning_midi`` is ordered low-E to high-E (index 0 = low-E),
  so string index ``s`` sits at canonical y =
  ``(cfg.n_strings - 1 - s + 0.5) / cfg.n_strings``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabvision.types import FrameFingering, GuitarConfig, Homography


@dataclass(frozen=True)
class FingerSample:
    """Per-finger landmark summary, image-normalised coords."""

    name: str
    tip_xy: tuple[float, float]
    tip_z: float
    curl_ratio: float  # ~0.5 fully curled, ~1.0 fully extended


@dataclass(frozen=True)
class HandSample:
    """A single fretting-hand sample (output of MediaPipeHandBackend)."""

    wrist_xy: tuple[float, float]
    wrist_z: float
    is_left_hand: bool
    confidence: float
    fingers: dict[str, FingerSample]


@dataclass(frozen=True)
class PosteriorConfig:
    """Tunable parameters for the per-finger logit construction."""

    # Distance kernel (Gaussian) σ in canonical fret-cell widths.  Smaller
    # = peakier posteriors.  Default tuned against typical fretboard
    # widths so that one-fret-wide span ≈ ±1 σ.
    sigma_fret_cells: float = 1.0
    sigma_string_cells: float = 0.6  # strings are smaller; tighter kernel

    # Curl prior: when curl_ratio < curl_min, the finger is treated as
    # curled and its log-probability is reduced by ``curled_log_penalty``
    # everywhere.  Extended fingers get no boost (penalty = 0).
    curl_min: float = 0.95
    curled_log_penalty: float = -2.0

    # z-depth prior: when the fingertip is within ``z_press_window`` of
    # the wrist-z, add ``pressing_log_bonus`` to the logits everywhere.
    # MediaPipe's z is relative to the wrist so 0 means co-planar.
    z_press_window: float = 0.04
    pressing_log_bonus: float = 0.5

    # Soft floor for finger logits — ensures the posterior is never
    # exactly zero (avoids -inf when later combined log-additively).
    floor_logit: float = -10.0


# Public order of fretting fingers.  Mirrors mediapipe_backend.FRETTING_FINGERS
# but defined locally to keep this module import-free.
FRETTING_FINGERS: tuple[str, ...] = ("index", "middle", "ring", "pinky")


def compute_fingering(
    hand: HandSample,
    H: Homography,  # noqa: N803
    cfg: GuitarConfig,
    posterior_cfg: PosteriorConfig | None = None,
) -> FrameFingering:
    """Build a :class:`FrameFingering` from a hand sample + homography.

    The resulting ``finger_pos_logits`` has shape
    ``(len(FRETTING_FINGERS), cfg.n_strings, cfg.max_fret + 1)``.
    Per-finger softmax of these logits is the position posterior over
    fret-cells; :meth:`FrameFingering.marginal_string_fret` aggregates
    across fingers.
    """
    pcfg = posterior_cfg or PosteriorConfig()

    if H.confidence == 0.0:
        # No usable homography → return a uniform-floor distribution but
        # keep ``homography_confidence`` honest at 0 so consumers know to
        # discount it.
        n_fingers = len(FRETTING_FINGERS)
        logits = np.full(
            (n_fingers, cfg.n_strings, cfg.max_fret + 1),
            pcfg.floor_logit,
            dtype=np.float64,
        )
        return FrameFingering(
            t=0.0, finger_pos_logits=logits, homography_confidence=0.0
        )

    H_inv = np.linalg.inv(H.H)  # noqa: N806 — math-convention name

    # Pre-compute the fret-cell-centre and string-y coordinates in canonical space.
    fret_xs = (np.arange(cfg.max_fret + 1) + 0.5) / (cfg.max_fret + 1)  # (F,)
    # cfg.tuning_midi is low-E -> high-E (idx 0 = low-E).  Canonical y:
    # 0 = high-E side, 1 = low-E side.  So idx 0 → y near 1; idx n-1 → y near 0.
    string_ys = (np.arange(cfg.n_strings - 1, -1, -1) + 0.5) / cfg.n_strings  # (S,)

    n_fingers = len(FRETTING_FINGERS)
    logits = np.full(
        (n_fingers, cfg.n_strings, cfg.max_fret + 1),
        pcfg.floor_logit,
        dtype=np.float64,
    )

    # Spacing constants for the kernel — converting "cells" back to canonical
    # units so callers can think in "1 fret-cell wide".
    fret_cell_size = 1.0 / (cfg.max_fret + 1)
    string_cell_size = 1.0 / cfg.n_strings
    sigma_x = pcfg.sigma_fret_cells * fret_cell_size
    sigma_y = pcfg.sigma_string_cells * string_cell_size

    for fi, name in enumerate(FRETTING_FINGERS):
        sample = hand.fingers.get(name)
        if sample is None:
            continue
        cx, cy = _project_point(H_inv, *sample.tip_xy)

        # Gaussian distance log-prob: -0.5 * ((cx-fret_x)/sigma_x)^2 + ((cy-string_y)/sigma_y)^2
        dx = (cx - fret_xs[None, :]) / sigma_x          # (1, F)
        dy = (cy - string_ys[:, None]) / sigma_y        # (S, 1)
        finger_logit = -0.5 * (dx ** 2 + dy ** 2)       # (S, F)

        # Curl prior: curled fingers can't fret.
        if sample.curl_ratio < pcfg.curl_min:
            finger_logit = finger_logit + pcfg.curled_log_penalty

        # z-depth prior: fingertip near the wrist's z-plane is pressing.
        if abs(sample.tip_z - hand.wrist_z) < pcfg.z_press_window:
            finger_logit = finger_logit + pcfg.pressing_log_bonus

        # Floor: don't let a single -inf cell escape (rare but defensive).
        np.maximum(finger_logit, pcfg.floor_logit, out=finger_logit)
        logits[fi] = finger_logit

    return FrameFingering(
        t=0.0,
        finger_pos_logits=logits,
        homography_confidence=float(H.confidence),
    )


def marginal_string_fret(finger_pos_logits: np.ndarray) -> np.ndarray:
    """Marginal softmax distribution over (string, fret) cells.

    Aggregates across fingers via log-sum-exp ("any finger here?"), then
    softmax-normalises so the output sums to 1.

    Args:
        finger_pos_logits: shape ``(n_fingers, n_strings, max_fret+1)``.

    Returns:
        ``(n_strings, max_fret+1)`` softmax distribution.
    """
    if finger_pos_logits.ndim != 3:
        raise ValueError(
            f"expected (n_fingers, n_strings, max_fret+1), got shape "
            f"{finger_pos_logits.shape}"
        )
    # logsumexp over the finger axis (axis 0).
    m = finger_pos_logits.max(axis=0)
    summed = np.log(np.exp(finger_pos_logits - m).sum(axis=0)) + m  # (S, F)
    # Softmax over the (S, F) plane.
    flat = summed.reshape(-1)
    flat_max = flat.max()
    e = np.exp(flat - flat_max)
    e /= e.sum()
    return e.reshape(summed.shape)


def _project_point(H_inv: np.ndarray, x: float, y: float) -> tuple[float, float]:  # noqa: N803
    """Project an image-pixel ``(x, y)`` through ``H^-1`` to canonical coords.

    ``HandSample`` carries pixel-space landmarks (the MediaPipe backend
    multiplies the normalised landmarks by ``(frame_width, frame_height)``
    before constructing the sample), so this is a straight projective
    matrix multiply; no implicit scaling.
    """
    pt = np.array([x, y, 1.0])
    proj = H_inv @ pt
    if abs(proj[2]) < 1e-12:
        return 0.0, 0.0  # degenerate; canonical (0,0)
    return float(proj[0] / proj[2]), float(proj[1] / proj[2])


__all__ = [
    "FingerSample",
    "HandSample",
    "PosteriorConfig",
    "FRETTING_FINGERS",
    "compute_fingering",
    "marginal_string_fret",
]
