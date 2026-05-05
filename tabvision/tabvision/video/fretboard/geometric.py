"""Geometric fretboard detection — Phase 3 thin wrapper around v0.

v0's ``tabvision-server/app/fretboard_detection.py`` (1466 lines) implements
Canny + Hough + RANSAC geometric detection that already achieves the
metrics behind v1's 91.6% F1 on the 11-clip set. Rather than verbatim
porting today, this module delegates to v0 and converts the output to the
spec's ``Homography`` type.

A clean port (eliminating the ``tabvision-server`` import) is tracked as
Phase 3 follow-up work; the wrapper boundary keeps the spec contract
(``§8 FretboardBackend``) intact regardless of the underlying impl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError
from tabvision.types import GuitarBBox, Homography

# v0 frozen tree path — see CLAUDE.md "v0 (frozen) reference".
_REPO_ROOT = Path(__file__).resolve().parents[4]
_V0_BACKEND_ROOT = _REPO_ROOT / "tabvision-server"


def _ensure_v0_on_path() -> None:
    p = str(_V0_BACKEND_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _v0_module():  # type: ignore[no-untyped-def]
    """Import v0's fretboard_detection module on-demand."""
    if not _V0_BACKEND_ROOT.exists():
        raise BackendError(
            f"v0 backend not found at {_V0_BACKEND_ROOT}; the geometric "
            "fretboard detector currently delegates to v0. Either keep the "
            "frozen v0 tree in place or wait for the full Phase-3 port."
        )
    _ensure_v0_on_path()
    try:
        from app import fretboard_detection  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendError(
            "could not import v0 fretboard_detection; see "
            "tabvision-server/app/fretboard_detection.py"
        ) from exc
    return fretboard_detection


class GeometricFretboardBackend:
    """Spec ``FretboardBackend`` impl wrapping v0 geometric detection."""

    name = "geometric"

    def detect(self, frame: np.ndarray, guitar_box: GuitarBBox) -> Homography:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise BackendError(f"expected BGR frame, got shape {frame.shape}")

        v0 = _v0_module()
        # v0 detects on the full frame; the spec asks us to operate on the
        # cropped guitar region. Cropping changes the homography reference
        # frame — we crop, detect, then offset the homography back to image
        # coordinates. For Phase 3 first-cut we just run on the full frame
        # to match v0's known-good behavior.
        del guitar_box  # used in a follow-up iteration when crop+offset is wired in

        geometry = v0.detect_fretboard(frame)
        if geometry is None or not geometry.is_valid():
            return Homography(
                H=np.eye(3, dtype=np.float64),
                confidence=0.0,
                method="geometric",
            )

        H = _homography_from_geometry(geometry)
        return Homography(
            H=H,
            confidence=float(geometry.detection_confidence),
            method="geometric",
        )


def _homography_from_geometry(geometry) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Map canonical fretboard plane [0,1]×[0,1] → image pixels.

    Canonical convention:
    - x-axis: along the neck, 0 = nut, 1 = body end of detected region.
    - y-axis: across the strings, 0 = top edge (high E side), 1 = bottom (low E side).

    The four corners of the canonical unit square map to v0's detected
    ``top_left``, ``top_right``, ``bottom_right``, ``bottom_left``.
    """
    import cv2  # local import to keep cv2 out of import-time path

    src = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dst = np.array(
        [
            geometry.top_left,
            geometry.top_right,
            geometry.bottom_right,
            geometry.bottom_left,
        ],
        dtype=np.float64,
    )
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return H.astype(np.float64)


def detect_geometric(frame: np.ndarray, guitar_box: GuitarBBox | None = None) -> Homography:
    """Module-level convenience wrapper."""
    backend = GeometricFretboardBackend()
    bbox = guitar_box or GuitarBBox(0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]), 0.0)
    return backend.detect(frame, bbox)


__all__ = ["GeometricFretboardBackend", "detect_geometric"]
