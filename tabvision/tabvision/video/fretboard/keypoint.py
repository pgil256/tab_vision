"""Keypoint-based fretboard detection — Phase 3 primary path.

Delegates to the YOLO-OBB backend (``video.guitar.yolo_backend``) and
converts its multi-class output (``neck`` + ``nut`` + per-fret OBBs)
into a :class:`Homography` mapping the canonical fretboard plane to
image coordinates.

Replaces v0's Hough + RANSAC fretboard detection as the primary path
per ``docs/DECISIONS.md`` (2026-05-05 "Phase 3 dataset reveals 3-class
fretboard-parts annotation"). The geometric backend at
``video.fretboard.geometric`` is retained as the fallback when YOLO
fails or returns low-confidence output.

Canonical convention (matches ``geometric.py`` so consumers can swap
backends without changing their math):

- x-axis: along the neck, ``0 = nut``, ``1 = body end of detected region``.
- y-axis: across the strings, ``0 = top edge (high E side)``,
  ``1 = bottom edge (low E side)``.
- The four corners of the unit square map to ``top_left``, ``top_right``,
  ``bottom_right``, ``bottom_left`` in that order.

The high-E vs low-E assignment uses image-Y (smaller Y = top of frame =
high-E side), which is correct for the standard iPhone-on-lap framing
the spec assumes (§7 Phase 3, §1 user setup). Clips with the guitar
flipped will still produce a valid homography but with the canonical
y-axis inverted; downstream consumers that depend on orientation should
sanity-check with the hand pipeline.
"""

from __future__ import annotations

import numpy as np

from tabvision.errors import BackendError
from tabvision.types import GuitarBBox, Homography
from tabvision.video.guitar.yolo_backend import (
    OBBDetection,
    OBBPredictions,
    YoloOBBBackend,
)


class KeypointFretboardBackend:
    """Spec ``FretboardBackend`` impl backed by the YOLO-OBB detector."""

    name = "keypoint"

    def __init__(self, backend: YoloOBBBackend | None = None) -> None:
        # Caller can inject a preconfigured backend (e.g. with a custom
        # checkpoint path); otherwise use defaults.
        self._yolo = backend or YoloOBBBackend()

    def detect(self, frame: np.ndarray, guitar_box: GuitarBBox) -> Homography:
        """Detect a per-frame :class:`Homography` for ``frame``.

        ``guitar_box`` is currently unused — the YOLO model operates on
        the full frame and the neck class implicitly localizes the
        guitar. We accept it to satisfy the §8 ``FretboardBackend``
        protocol and to leave a hook for future crop-then-detect
        optimization (running detection on the cropped guitar region
        could be faster on large frames).
        """
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise BackendError(f"expected BGR frame, got shape {frame.shape}")
        del guitar_box  # documented unused; reserved for crop-then-detect

        preds = self._yolo.predict_all(frame)
        return predictions_to_homography(preds)


def predictions_to_homography(preds: OBBPredictions) -> Homography:
    """Convert per-frame OBB predictions into a fretboard :class:`Homography`.

    Pure function — split out so unit tests can exercise the geometry
    without touching the YOLO model.

    Returns an identity-fallback ``Homography`` with confidence 0.0 if
    no neck OBB was detected. The geometric backend can be tried as a
    fallback in that case (the higher-level orchestrator decides).
    """
    neck = preds.best_neck()
    if neck is None:
        return Homography(
            H=np.eye(3, dtype=np.float64),
            confidence=0.0,
            method="keypoint",
        )

    corners = _obb_to_corners(neck)             # 4×2, image px
    nut = preds.best_nut()
    nut_xy = (nut.cx, nut.cy) if nut else None
    ordered = _order_corners_by_neck_anatomy(corners, nut_xy)
    H = _homography_from_quad(ordered)  # noqa: N806 — math-convention name

    # Confidence: weight neck heavily, boost a bit if we also pinned the
    # nut side via an explicit nut detection (more accurate orientation).
    base_conf = float(neck.confidence)
    nut_bonus = 0.05 if nut is not None and nut.confidence > 0.25 else 0.0
    fret_bonus = 0.05 if len(preds.frets) >= 4 else 0.0
    confidence = float(min(1.0, base_conf + nut_bonus + fret_bonus))

    return Homography(H=H, confidence=confidence, method="keypoint")


def _obb_to_corners(obb: OBBDetection) -> np.ndarray:
    """Return the 4 corners of an oriented bbox in image coordinates.

    Order: (+w/2,+h/2), (-w/2,+h/2), (-w/2,-h/2), (+w/2,-h/2) before
    rotation — i.e. CCW starting from the +x +y corner. Caller may
    re-order based on anatomy.
    """
    rad = float(np.radians(obb.rotation_deg))
    cos_r = float(np.cos(rad))
    sin_r = float(np.sin(rad))
    half_w = obb.w / 2.0
    half_h = obb.h / 2.0
    local = np.array(
        [
            [+half_w, +half_h],
            [-half_w, +half_h],
            [-half_w, -half_h],
            [+half_w, -half_h],
        ],
        dtype=np.float64,
    )
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)  # noqa: N806
    rotated = local @ R.T
    rotated[:, 0] += obb.cx
    rotated[:, 1] += obb.cy
    return rotated


def _order_corners_by_neck_anatomy(
    corners: np.ndarray, nut_xy: tuple[float, float] | None
) -> np.ndarray:
    """Order 4 OBB corners as (top_left, top_right, bottom_right, bottom_left).

    "top" = high-E side (smaller image Y under the standard framing).
    "left" = nut side (closer to the headstock). When a nut detection
    is available, it disambiguates which short edge is the nut. When
    not, we fall back to the smaller-X short edge (heuristic: typical
    framing places the headstock on the left of the image, but this is
    just a tie-break — the homography stays geometrically valid either
    way, only the canonical orientation may flip).

    Args:
        corners: shape (4, 2) array of image-px corner coordinates,
            output of :func:`_obb_to_corners`.
        nut_xy: optional ``(x, y)`` of the nut detection's center. When
            present, the short-edge midpoint closest to it is the nut
            side.

    Returns:
        Shape (4, 2) array, rows = ``[top_left, top_right, bottom_right,
        bottom_left]``.
    """
    if corners.shape != (4, 2):
        raise ValueError(f"expected (4, 2) corners, got {corners.shape}")

    # The OBB corners came from _obb_to_corners as (+,+),(-,+),(-,-),(+,-)
    # in local coords pre-rotation. With w as the OBB major axis (YOLO-OBB
    # convention) corners 0-1 and 2-3 are the long edges (along the neck);
    # corners 1-2 and 3-0 are the short edges (the nut + body ends).
    short_edges = [(1, 2), (3, 0)]
    short_midpoints = [
        (i, j, np.array([(corners[i, 0] + corners[j, 0]) / 2,
                         (corners[i, 1] + corners[j, 1]) / 2]))
        for (i, j) in short_edges
    ]

    if nut_xy is not None:
        nut_pt = np.array(nut_xy)
        dists = [float(np.linalg.norm(mid - nut_pt)) for _, _, mid in short_midpoints]
        nut_edge_idx = int(np.argmin(dists))
    else:
        # Fall back: pick the short edge whose midpoint has smaller mean X.
        xs = [float(mid[0]) for _, _, mid in short_midpoints]
        nut_edge_idx = int(np.argmin(xs))

    nut_i, nut_j, _ = short_midpoints[nut_edge_idx]
    body_i, body_j, _ = short_midpoints[1 - nut_edge_idx]

    # The two corners on the nut edge are nut_i, nut_j; pick the one with
    # smaller image Y as "top" (high-E side).
    nut_top, nut_bot = _split_top_bottom(corners, nut_i, nut_j)
    body_top, body_bot = _split_top_bottom(corners, body_i, body_j)

    return np.array(
        [
            corners[nut_top],     # top_left
            corners[body_top],    # top_right
            corners[body_bot],    # bottom_right
            corners[nut_bot],     # bottom_left
        ],
        dtype=np.float64,
    )


def _split_top_bottom(corners: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """Return (top_idx, bot_idx) for the two corner indices ``i, j``."""
    if corners[i, 1] <= corners[j, 1]:
        return i, j
    return j, i


def _homography_from_quad(ordered_corners: np.ndarray) -> np.ndarray:
    """Map canonical [0,1]×[0,1] -> image px via a 4-point homography."""
    import cv2

    src = np.array(
        [
            [0.0, 0.0],   # top_left
            [1.0, 0.0],   # top_right
            [1.0, 1.0],   # bottom_right
            [0.0, 1.0],   # bottom_left
        ],
        dtype=np.float32,
    )
    dst = ordered_corners.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)  # noqa: N806
    return H.astype(np.float64)


def detect_keypoint(
    frame: np.ndarray,
    guitar_box: GuitarBBox | None = None,
    *,
    backend: YoloOBBBackend | None = None,
) -> Homography:
    """Module-level convenience wrapper, mirrors ``detect_geometric``."""
    fb = KeypointFretboardBackend(backend=backend)
    bbox = guitar_box or GuitarBBox(0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]), 0.0)
    return fb.detect(frame, bbox)


__all__ = [
    "KeypointFretboardBackend",
    "predictions_to_homography",
    "detect_keypoint",
]
