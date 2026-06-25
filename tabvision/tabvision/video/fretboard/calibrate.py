"""Per-clip geometric fretboard calibration — v1.1 chunk-6 WS1/WS2.

Training-free calibration of the canonical fretboard from per-clip **detected**
OBB cues (no per-clip-tuned constants), living on the implementation side of the
§8 ``FretboardBackend`` contract. It targets the systematic string/fret bias the
chunk-5 GAPS chain exhibited.

Why this exists (the diagnosis, grounded in the code). ``compute_fingering``
builds its fret grid with a **uniform** partition of the canonical x-axis
(``fret_xs = (arange(F+1)+0.5)/(F+1)``). But the canonical x-axis is, by
construction, *proportional to physical distance along the neck* — the
homography maps the rectangular neck region to the unit square — and real frets
are **not** uniformly spaced in physical distance: fret wire ``k`` sits at
``D_k = S·(1 − r^k)`` with ``r = 2^(-1/12)`` (the "rule of 18"), compressing
toward the body. So a uniform partition systematically **over-estimates** fret
number for mid/high frets (e.g. a fingertip physically at fret 12 — canonical
x ≈ 0.67 of a nut→fret-24 span — is read as ~fret 16), and the pitch constraint
then drags the predicted string toward the bass. That is exactly the
``+4/+5`` fret / ``−1..−4`` string bias the WS0 diagnostic measured
(``docs/EVAL_REPORTS/v1_1_gaps_chunk6_ws0_2026-06-22.md``).

The fix is parameter-free physics: replace the uniform fret partition with the
rule-of-18 partition, *anchored and scaled per clip* using the detected fret-OBB
sequence (currently used only for a confidence bonus) and the nut anchor. Since
canonical x is affine in physical distance, the wire positions obey

    x_canonical(k) = x0 + b · (1 − r^k)

with ``x0`` the nut's canonical x (the affine intercept) and ``b`` a per-clip
scale. Crucially the *shape* of a finite geometric sequence is invariant to the
absolute fret index, so the scale alone cannot recover which fret is which — the
nut anchor (the homography's canonical origin, optionally refined by a detected
nut OBB) pins the intercept and lets a small search over the first detected
wire's index resolve the rest.

This module is **numpy-only** (no cv2 / mediapipe / ultralytics): WS1 reuses the
*cached* homography and only re-derives the nonlinear fret map, so it runs
cache-only (seconds) in both eval gates. The cross-string axis / orientation /
geometry-aware-confidence work (WS2, which does touch the homography fit) lands
in ``keypoint.py`` where the cv2 dependency already lives.
"""

from __future__ import annotations

import numpy as np

from tabvision.types import GuitarConfig, Homography
from tabvision.video.guitar.yolo_backend import OBBPredictions

# Equal-temperament fret-spacing ratio (consecutive wire distance shrinks by
# this factor toward the body): r = 2^(-1/12) ≈ 0.94387. Physical constant, not
# a tunable — there is nothing per-clip to fit here.
RULE_OF_18_RATIO: float = 2.0 ** (-1.0 / 12.0)

# Robustness knobs. These are *generic fit-quality* gates (not tuned to the
# clean-12): a calibration that does not look like a rule-of-18 wire sequence is
# rejected and the caller falls back to the uniform map for that frame.
_MIN_WIRES: int = 4  # need enough wires to fit + validate a geometric sequence
_K0_MAX: int = 6  # the first visible wire is fret 1..6 in practice
# Robust (inlier-consensus) fit knobs — generic RANSAC-style robustness, not
# values tuned to the clean-12. A detected wire is an *inlier* to a candidate
# rule-of-18 model when it lands within ``_INLIER_TOL_FRAC`` of a median gap of
# its predicted position; the model must explain at least ``_MIN_INLIERS`` wires
# and ``_MIN_INLIER_FRAC`` of all detections, with inlier RMS under
# ``_MAX_RMS_FRAC`` of a median gap. This tolerates the spurious/high-fret
# detections that wreck a least-squares-over-all-wires fit.
_MAX_RMS_FRAC: float = 0.30  # max inlier RMS as a fraction of the median gap
_INLIER_TOL_FRAC: float = 0.5  # a wire within half a gap of its prediction is an inlier
_MIN_INLIERS: int = 4  # need a real consensus, not a 2-point coincidence
_MIN_INLIER_FRAC: float = 0.5  # the model must explain most detections


def project_to_canonical(homography: Homography, points_xy: np.ndarray) -> np.ndarray:
    """Project image-pixel points to canonical fretboard coords via ``H^-1``.

    ``Homography.H`` maps canonical [0,1]² → image px (see
    :func:`tabvision.video.fretboard.keypoint._homography_from_quad`), so the
    inverse maps image → canonical. Pure numpy; mirrors ``compute_fingering``'s
    ``_project_point`` but vectorised over many points.

    Args:
        homography: the fitted board homography (canonical→image).
        points_xy: shape ``(N, 2)`` image-pixel coordinates.

    Returns:
        Shape ``(N, 2)`` canonical coordinates. Points whose projective weight
        is degenerate (≈0) map to ``(0, 0)`` (matching ``_project_point``).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got shape {pts.shape}")
    h_inv = np.linalg.inv(homography.H)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 3)
    proj = homog @ h_inv.T  # (N, 3)
    w = proj[:, 2]
    safe = np.abs(w) >= 1e-12
    out = np.zeros((pts.shape[0], 2), dtype=np.float64)
    out[safe, 0] = proj[safe, 0] / w[safe]
    out[safe, 1] = proj[safe, 1] / w[safe]
    return out


def nut_at_high_canonical_x(wire_xs_sorted: np.ndarray) -> bool:
    """Decide which end of an ascending wire sequence is the nut, by spacing decay.

    Fret wires compress toward the body, so the **nut** side has the larger
    inter-wire gaps. Comparing the mean gap in the low-x half vs the high-x half
    of the sorted wire positions tells us which end is the nut — the per-clip,
    detected-cue replacement for ``keypoint.py``'s ``smaller-X = nut`` and
    ``nut-right → flip`` rig heuristics.

    Args:
        wire_xs_sorted: shape ``(m,)`` canonical-x of detected fret wires,
            ascending.

    Returns:
        ``True`` if the nut is at the **high**-x end (gaps grow with x),
        ``False`` if at the low-x end. Ties / too-few wires default to ``False``
        (nut at low x, the canonical convention).
    """
    gaps = np.diff(np.asarray(wire_xs_sorted, dtype=np.float64))
    if gaps.size < 2:
        return False
    mid = gaps.size // 2
    low_half = float(np.mean(gaps[:mid])) if mid > 0 else float(gaps[0])
    high_half = float(np.mean(gaps[mid:]))
    return high_half > low_half


def _fit_scale_fixed_intercept(u: np.ndarray, x: np.ndarray, x0: float) -> tuple[float, float]:
    """Least-squares ``b`` for ``x ≈ x0 + b·u`` with intercept pinned at ``x0``.

    Returns ``(b, sse)`` where ``sse`` is the residual sum of squares.
    """
    u = np.asarray(u, dtype=np.float64)
    dx = np.asarray(x, dtype=np.float64) - x0
    denom = float(np.dot(u, u))
    if denom <= 1e-18:
        return 0.0, float(np.dot(dx, dx))
    b = float(np.dot(u, dx) / denom)
    resid = dx - b * u
    return b, float(np.dot(resid, resid))


def _ruleof18_u(k: np.ndarray | float) -> np.ndarray:
    """Rule-of-18 normalized distance ``1 − r^k`` (nut at k=0 → 0)."""
    return 1.0 - np.power(RULE_OF_18_RATIO, k)


def fit_fret_map(
    wire_xs_from_nut: np.ndarray,
    x0: float,
    max_fret: int,
    *,
    k0_max: int = _K0_MAX,
    max_rms_frac: float = _MAX_RMS_FRAC,
    inlier_tol_frac: float = _INLIER_TOL_FRAC,
    min_inliers: int = _MIN_INLIERS,
    min_inlier_frac: float = _MIN_INLIER_FRAC,
) -> np.ndarray | None:
    """Robustly fit a rule-of-18 fret-cell-center map from wires + a nut anchor.

    For each candidate first-wire index ``k0`` the wires are fit to
    ``x ≈ x0 + b·(1 − r^k)`` (intercept pinned at the nut), then re-fit on the
    inlier subset — wires within ``inlier_tol_frac`` of a median gap of their
    predicted position. The ``k0`` with the largest inlier consensus wins
    (tie-broken by inlier RMS). A growing-gap / non-fretboard sequence simply
    fails to gather a consensus and is rejected, so no separate physical gate is
    needed.

    Args:
        wire_xs_from_nut: canonical-x of detected fret wires, ordered from the
            nut toward the body (index 0 = wire nearest the nut). Ascending or
            descending in raw canonical x per board orientation; the fitted ``b``
            carries the sign.
        x0: canonical-x of the nut (affine intercept; the homography origin,
            optionally refined by a detected nut OBB).
        max_fret: ``cfg.max_fret`` — the map covers frets ``0..max_fret``.
        k0_max: search the first detected wire's fret index over ``1..k0_max``.
        max_rms_frac / inlier_tol_frac / min_inliers / min_inlier_frac: robust
            consensus knobs (fractions are of the median wire gap).

    Returns:
        Shape ``(max_fret + 1,)`` canonical-x cell centers using the *same*
        ``k + 0.5`` cell-center convention as the uniform map, or ``None`` if no
        rule-of-18 fit is trustworthy (caller falls back to the uniform map).
    """
    wires = np.asarray(wire_xs_from_nut, dtype=np.float64)
    m = wires.size
    if m < _MIN_WIRES:
        return None

    median_gap = float(np.median(np.abs(np.diff(wires))))
    if median_gap <= 1e-9:
        return None
    tol = inlier_tol_frac * median_gap
    dx = wires - x0

    best: tuple[int, float, float] | None = None  # (n_inliers, -rms_in, b)
    for k0 in range(1, k0_max + 1):
        u = _ruleof18_u(np.arange(k0, k0 + m, dtype=np.float64))
        b0, _ = _fit_scale_fixed_intercept(u, wires, x0)
        inliers = np.abs(dx - b0 * u) <= tol
        n_in = int(inliers.sum())
        if n_in < min_inliers or n_in < min_inlier_frac * m:
            continue
        b, sse = _fit_scale_fixed_intercept(u[inliers], wires[inliers], x0)
        rms_in = float(np.sqrt(sse / n_in))
        key = (n_in, -rms_in, b)
        if best is None or (n_in, -rms_in) > (best[0], best[1]):
            best = key

    if best is None:
        return None
    n_in, neg_rms, b = best
    if -neg_rms > max_rms_frac * median_gap:
        return None
    # Orientation/scale sanity: b must move the same direction the wires do.
    wire_dir = np.sign(wires[-1] - wires[0])
    if wire_dir == 0 or np.sign(b) != wire_dir:
        return None

    ks_all = np.arange(max_fret + 1, dtype=np.float64) + 0.5  # cell centers
    fret_xs = x0 + b * _ruleof18_u(ks_all)
    # Must be strictly monotone (a valid 1-D coordinate) for the kernel.
    diffs = np.diff(fret_xs)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        return None
    return fret_xs


def calibrate_fret_xs(
    preds: OBBPredictions,
    homography: Homography,
    cfg: GuitarConfig,
) -> np.ndarray | None:
    """Per-clip nonlinear fret map from detected fret/nut OBBs + the cached H.

    Projects the detected fret-wire centers (and the nut, if present) into the
    cached canonical frame, decides the nut end by fret-spacing decay, anchors
    the affine intercept at the nut, and fits the rule-of-18 cell-center map. The
    cached homography is used **as-is** (WS1 re-derives only the fret partition,
    so this stays numpy-only and cache-only).

    Returns:
        Shape ``(cfg.max_fret + 1,)`` canonical-x fret-cell centers, or ``None``
        when there are too few/garbled fret detections (caller then uses the
        uniform map — the chunk-3 fall-back-to-audio invariant is preserved
        because a uniform-map frame is exactly the pre-WS1 behaviour).
    """
    if homography.confidence <= 0.0 or not preds.frets:
        return None

    wire_centers = np.array([[d.cx, d.cy] for d in preds.frets], dtype=np.float64)
    wire_canon = project_to_canonical(homography, wire_centers)
    wire_x = wire_canon[:, 0]

    # Keep wires inside (a slightly padded) canonical span; drop wild outliers
    # from spurious detections before fitting.
    keep = (wire_x > -0.15) & (wire_x < 1.25)
    wire_x = np.sort(wire_x[keep])
    if wire_x.size < _MIN_WIRES:
        return None

    nut_high = nut_at_high_canonical_x(wire_x)

    # Nut anchor: prefer a detected nut OBB's canonical x when it sits on the
    # correct (nut) side; otherwise use the homography origin (canonical x = 0
    # for the low-x nut convention, x = 1 for the flipped one), i.e. the
    # neck-OBB nut edge the homography already encodes.
    nut = preds.best_nut()
    x0 = 1.0 if nut_high else 0.0
    if nut is not None:
        nut_canon = project_to_canonical(homography, np.array([[nut.cx, nut.cy]]))[0, 0]
        if nut_high and nut_canon >= wire_x[-1]:
            x0 = float(nut_canon)
        elif not nut_high and nut_canon <= wire_x[0]:
            x0 = float(nut_canon)

    # Order wires from the nut toward the body for index assignment.
    wires_from_nut = wire_x[::-1] if nut_high else wire_x
    return fit_fret_map(wires_from_nut, x0, cfg.max_fret)


def calibrate_board(
    preds: OBBPredictions,
    cfg: GuitarConfig,
) -> tuple[Homography, np.ndarray | None]:
    """Full per-clip board calibration (chunk-6 WS2): nut-axis homography + fret map.

    Re-fits the homography with the **cross-string axis anchored to the detected
    nut OBB** (:func:`keypoint.predictions_to_homography_nut_axis`), then derives
    the WS1 nonlinear fret map in that re-fit frame. Returns ``(homography,
    fret_xs)`` ready for the eval ``calibrate`` hook; ``fret_xs`` is ``None`` when
    the fret detections don't support a rule-of-18 fit (the cross-string axis
    still improves, with the uniform fret partition). cv2 is used only via the
    lazy import inside the keypoint homography fit, so this module's top stays
    import-light.
    """
    from tabvision.video.fretboard.keypoint import predictions_to_homography_nut_axis

    homography = predictions_to_homography_nut_axis(preds)
    fret_xs = calibrate_fret_xs(preds, homography, cfg)
    return homography, fret_xs


__all__ = [
    "RULE_OF_18_RATIO",
    "project_to_canonical",
    "nut_at_high_canonical_x",
    "fit_fret_map",
    "calibrate_fret_xs",
    "calibrate_board",
]
