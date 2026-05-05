"""MediaPipe Hands backend — Phase 4 §8 ``HandBackend`` impl.

Wraps the MediaPipe Tasks API ``HandLandmarker`` (v0.10+) and routes
detected fretting-hand landmarks through the homography-aware
posterior in :mod:`tabvision.video.hand.fingertip_to_fret`.

Per ``docs/DECISIONS.md`` (2026-05-05 "Phase 4 entry") this module is a
hybrid port: the MediaPipe plumbing mirrors v0's
``tabvision-server/app/video_pipeline.py``; the canonical-coord
projection + per-cell posterior is the new layer the §8 contract
requires. v0's HandObservation type is intentionally not re-exported
— v1 consumers see only :class:`FrameFingering`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError
from tabvision.types import FrameFingering, GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import (
    FingerSample,
    HandSample,
    PosteriorConfig,
    compute_fingering,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ENV = "TABVISION_MEDIAPIPE_HAND_MODEL"
DEFAULT_MODEL_NAME = "hand_landmarker.task"

# MediaPipe hand-landmark indices (21 landmarks per hand).
WRIST_IDX = 0
# Per-finger landmarks: (mcp, pip, dip, tip).  Thumb's "MCP/PIP/DIP/TIP"
# are CMC/MCP/IP/TIP since it has fewer joints; tip-vs-base geometry is
# the same shape so we treat it uniformly here, but the thumb is dropped
# from fretting-finger reasoning (see THUMB constant below).
_FINGER_LANDMARKS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
THUMB = "thumb"
FRETTING_FINGERS: tuple[str, ...] = ("index", "middle", "ring", "pinky")


def _default_model_path() -> Path:
    if env := os.environ.get(DEFAULT_MODEL_ENV):
        return Path(env)
    # Match v0's convention so a single download serves both v0 and v1.
    return Path(os.path.expanduser(f"~/.mediapipe/models/{DEFAULT_MODEL_NAME}"))


@dataclass(frozen=True)
class HandBackendConfig:
    """Tunable parameters for the MediaPipe backend.  Kept off the §8
    HandBackend protocol so consumers don't have to care."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    posterior: PosteriorConfig = PosteriorConfig()


class MediaPipeHandBackend:
    """§8 ``HandBackend`` impl using MediaPipe Tasks API."""

    name = "mediapipe"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        config: HandBackendConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else _default_model_path()
        self.config = config or HandBackendConfig()
        self._landmarker = None  # lazy-loaded; closed in close()

    # ----- HandBackend protocol -----

    def detect(
        self, frame: np.ndarray, H: Homography, cfg: GuitarConfig  # noqa: N803
    ) -> FrameFingering:
        """Run MediaPipe + posterior. Returns a degenerate :class:`FrameFingering`
        with ``homography_confidence=0`` if no hand is found in ``frame``."""
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise BackendError(f"expected BGR frame, got shape {frame.shape}")

        landmarks = self._extract_fretting_hand(frame)
        if landmarks is None:
            return _empty_fingering(cfg)

        return compute_fingering(
            landmarks,
            H,
            cfg,
            self.config.posterior,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> MediaPipeHandBackend:
        return self

    def __exit__(self, *_exc) -> None:  # noqa: ANN001
        self.close()

    # ----- private -----

    def _load(self):
        """Lazy-load the MediaPipe HandLandmarker."""
        if self._landmarker is not None:
            return self._landmarker
        if not self.model_path.exists():
            raise BackendError(
                f"MediaPipe HandLandmarker model not found at {self.model_path}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                f"or set {DEFAULT_MODEL_ENV} to point at an existing one."
            )
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise BackendError(
                "mediapipe is not installed. Install with: "
                "pip install '.[vision]' (mediapipe is in the vision extras)."
            ) from exc

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return self._landmarker

    def _extract_fretting_hand(self, frame: np.ndarray) -> HandSample | None:
        """Run MediaPipe and return the fretting hand's landmarks, or None."""
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise BackendError(
                "opencv-python and mediapipe are required. Install with: "
                "pip install '.[vision]'."
            ) from exc

        landmarker = self._load()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks or not result.handedness:
            return None

        idx = _select_fretting_hand(result.hand_landmarks, result.handedness)
        landmarks = result.hand_landmarks[idx]
        handedness = result.handedness[idx]
        h, w = frame.shape[:2]
        return _build_hand_sample(landmarks, handedness, frame_width=w, frame_height=h)


# ----- helpers (importable for testing) -----


def _select_fretting_hand(
    hand_landmarks_list,  # noqa: ANN001 — MediaPipe-typed list
    handedness_list,  # noqa: ANN001
) -> int:
    """Pick the index of the fretting hand. Mirrors v0's logic.

    For a right-handed player, the fretting hand is the player's left
    (which MediaPipe labels "Right" because its handedness convention is
    mirror-image of the camera). When labels are ambiguous (or both
    hands have the same label), we fall back to the hand with the wider
    fingertip-spread, which is empirically the fretting hand because
    spread fingers means each one is on a different fret.
    """
    if len(hand_landmarks_list) == 1:
        return 0

    for i, info in enumerate(handedness_list):
        if info[0].category_name == "Right":
            return i

    # Same-side fallback: pick the hand with widest fingertip-x spread.
    best_idx, best_spread = 0, -1.0
    for i, lms in enumerate(hand_landmarks_list):
        tip_xs = [lms[t].x for _, _, _, t in _FINGER_LANDMARKS.values()]
        spread = max(tip_xs) - min(tip_xs)
        if spread > best_spread:
            best_idx, best_spread = i, spread
    return best_idx


def _build_hand_sample(  # noqa: ANN001 — MediaPipe-typed landmarks
    landmarks,
    handedness,
    *,
    frame_width: int,
    frame_height: int,
) -> HandSample:
    """Convert MediaPipe Tasks-API landmarks into a :class:`HandSample`.

    Tip and wrist positions are stored in **image-pixel coordinates** so
    they share the same coordinate frame as :class:`Homography.H`.  The
    curl ratio is dimensionless (computed from normalised coords) and
    so doesn't need scaling.
    """
    wrist = landmarks[WRIST_IDX]

    fingers: dict[str, FingerSample] = {}
    for name, (mcp_i, pip_i, _dip_i, tip_i) in _FINGER_LANDMARKS.items():
        mcp, pip, tip = landmarks[mcp_i], landmarks[pip_i], landmarks[tip_i]
        # Curl ratio in normalised coords (dimensionless): tip-to-mcp /
        # (tip-to-pip + pip-to-mcp).  Range roughly [0.5, 1.0]; ≈1.0 =
        # fully extended, ≈0.5 = fully curled.
        d_tip_pip = float(np.hypot(tip.x - pip.x, tip.y - pip.y))
        d_pip_mcp = float(np.hypot(pip.x - mcp.x, pip.y - mcp.y))
        d_tip_mcp = float(np.hypot(tip.x - mcp.x, tip.y - mcp.y))
        seg = d_tip_pip + d_pip_mcp
        curl_ratio = (d_tip_mcp / seg) if seg > 1e-6 else 1.0
        fingers[name] = FingerSample(
            name=name,
            tip_xy=(float(tip.x) * frame_width, float(tip.y) * frame_height),
            tip_z=float(tip.z),
            curl_ratio=float(curl_ratio),
        )

    return HandSample(
        wrist_xy=(float(wrist.x) * frame_width, float(wrist.y) * frame_height),
        wrist_z=float(wrist.z),
        is_left_hand=(handedness[0].category_name == "Right"),
        confidence=float(handedness[0].score),
        fingers=fingers,
    )


def _empty_fingering(cfg: GuitarConfig) -> FrameFingering:
    """Degenerate FrameFingering for frames where no hand was detected."""
    n_fingers = len(FRETTING_FINGERS)
    logits = np.zeros((n_fingers, cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    return FrameFingering(t=0.0, finger_pos_logits=logits, homography_confidence=0.0)


__all__ = [
    "MediaPipeHandBackend",
    "HandBackendConfig",
    "FRETTING_FINGERS",
    "DEFAULT_MODEL_ENV",
    "DEFAULT_MODEL_NAME",
]
