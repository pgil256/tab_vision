"""YOLO-OBB guitar detector backend — Phase 3.

Wraps a fine-tuned ultralytics YOLO11-OBB model trained on Roboflow
``b101/guitar-3``. Returns ``GuitarBBox`` instances with ``rotation_deg``
set to the detected oriented-bbox angle.

**License:** ultralytics is AGPL-3.0. Importing this module taints the
combined TabVision distribution as AGPL. See LICENSES.md and
docs/DECISIONS.md (2026-05-05).

Usage:

    backend = YoloOBBBackend()                 # default checkpoint path
    bbox = backend.detect(frame_bgr)           # → GuitarBBox or None
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError
from tabvision.types import GuitarBBox

DEFAULT_CHECKPOINT_ENV = "TABVISION_GUITAR_YOLO_CHECKPOINT"
DEFAULT_CHECKPOINT_NAME = "guitar-yolo-obb-finetuned.pt"


def _default_checkpoint_path() -> Path:
    if env := os.environ.get(DEFAULT_CHECKPOINT_ENV):
        return Path(env)
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))
    return data_root / "models" / DEFAULT_CHECKPOINT_NAME


class YoloOBBBackend:
    """Audio backend wrapping a fine-tuned YOLO11-OBB guitar model."""

    name = "yolo-obb"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path()
        )
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.exists():
            raise BackendError(
                f"YOLO-OBB checkpoint not found at {self.checkpoint_path}. "
                "Train one with: python -m scripts.train.yolo_guitar_obb "
                f"(or set {DEFAULT_CHECKPOINT_ENV} to point at an existing one)."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendError(
                "ultralytics not installed. Install with: pip install '.[vision]' "
                "(AGPL — see LICENSES.md)."
            ) from exc
        self._model = YOLO(str(self.checkpoint_path))
        return self._model

    def detect(self, frame: np.ndarray) -> GuitarBBox | None:
        """Detect the most-confident guitar bbox in ``frame`` (BGR).

        Returns None if no detection meets the confidence threshold.
        """
        model = self._load()
        results = model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return None

        # ultralytics OBB result: .xywhr has shape (N, 5) with (cx, cy, w, h, rad).
        # .conf has shape (N,).
        xywhr = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, "cpu") else np.asarray(obb.xywhr)
        confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.asarray(obb.conf)
        if xywhr.size == 0:
            return None

        # Take the highest-confidence detection.
        idx = int(np.argmax(confs))
        cx, cy, w, h, rad = (float(v) for v in xywhr[idx])
        confidence = float(confs[idx])
        # Convert center-form to top-left-form for spec-compatible (x, y, w, h).
        x = cx - w / 2.0
        y = cy - h / 2.0
        rotation_deg = float(np.degrees(rad))

        return GuitarBBox(
            x=x,
            y=y,
            w=w,
            h=h,
            confidence=confidence,
            rotation_deg=rotation_deg,
        )


__all__ = ["YoloOBBBackend", "DEFAULT_CHECKPOINT_ENV", "DEFAULT_CHECKPOINT_NAME"]
