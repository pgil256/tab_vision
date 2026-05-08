"""YOLO-OBB guitar/fretboard-parts detector backend — Phase 3.

Wraps a fine-tuned ultralytics YOLO11-OBB model trained on Roboflow
``b101/guitar-3``. The dataset has three classes (``fret``, ``neck``,
``nut``) emitted as oriented bounding boxes; per ``docs/DECISIONS.md``
(2026-05-05 "Phase 3 dataset reveals 3-class fretboard-parts annotation"),
the same model serves dual purpose:

- ``neck`` class -> proxy for the spec's :class:`GuitarBBox` (preflight
  + cropping). Returned by :meth:`YoloOBBBackend.detect`.
- ``fret`` + ``nut`` classes -> keypoint anchors consumed by the
  keypoint-based fretboard backend (``video.fretboard.keypoint``).

**License:** ultralytics is AGPL-3.0. Importing this module taints the
combined TabVision distribution as AGPL. See LICENSES.md and
docs/DECISIONS.md (2026-05-05).

Usage::

    backend = YoloOBBBackend()                  # default checkpoint path
    bbox = backend.detect(frame_bgr)            # GuitarBBox (neck) or None

    preds = backend.predict_all(frame_bgr)      # all three classes in one pass
    preds.neck                                  # OBBDetection | None
    preds.frets                                 # list[OBBDetection], by-confidence
    preds.nut                                   # OBBDetection | None
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError
from tabvision.types import GuitarBBox

DEFAULT_CHECKPOINT_ENV = "TABVISION_GUITAR_YOLO_CHECKPOINT"
DEFAULT_CHECKPOINT_NAME = "guitar-yolo-obb-finetuned.pt"

# Class names from the Roboflow b101/guitar-3 v2 data.yaml. Indices verified:
# 0=fret, 1=neck, 2=nut. Hardcoded here so we don't have to round-trip
# through the model's metadata for every prediction.
CLASS_FRET = "fret"
CLASS_NECK = "neck"
CLASS_NUT = "nut"
ALL_CLASSES = (CLASS_FRET, CLASS_NECK, CLASS_NUT)


def _default_checkpoint_path() -> Path:
    if env := os.environ.get(DEFAULT_CHECKPOINT_ENV):
        return Path(env)
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))
    return data_root / "models" / DEFAULT_CHECKPOINT_NAME


@dataclass(frozen=True)
class OBBDetection:
    """A single oriented-bbox detection from the YOLO-OBB model.

    Internal representation — keeps the fret/nut detections (which aren't
    ``GuitarBBox`` per the §8 contract) out of the public type system.
    The ``neck``-class equivalent is converted to :class:`GuitarBBox` by
    :meth:`YoloOBBBackend.detect`.
    """

    class_name: str
    cx: float          # center x (image px)
    cy: float          # center y (image px)
    w: float           # rotated width (along major axis)
    h: float           # rotated height (along minor axis)
    rotation_deg: float  # CCW from +x, in image coords
    confidence: float

    def to_guitar_bbox(self) -> GuitarBBox:
        """Convert to a §8 :class:`GuitarBBox` (axis-aligned (x, y, w, h)
        plus rotation_deg). Caller is responsible for picking the right
        class — typically only valid for ``neck``."""
        return GuitarBBox(
            x=self.cx - self.w / 2.0,
            y=self.cy - self.h / 2.0,
            w=self.w,
            h=self.h,
            confidence=self.confidence,
            rotation_deg=self.rotation_deg,
        )


@dataclass(frozen=True)
class OBBPredictions:
    """Per-class OBB detections from a single frame.

    Each list is sorted by descending confidence so callers that want
    "the best one" can take ``[0]``. ``neck`` and ``nut`` are typically
    one-per-frame; ``frets`` is one per visible fret line.
    """

    frets: list[OBBDetection] = field(default_factory=list)
    neck: list[OBBDetection] = field(default_factory=list)
    nut: list[OBBDetection] = field(default_factory=list)

    def best_neck(self) -> OBBDetection | None:
        return self.neck[0] if self.neck else None

    def best_nut(self) -> OBBDetection | None:
        return self.nut[0] if self.nut else None


class YoloOBBBackend:
    """Backend wrapping a fine-tuned YOLO11-OBB fretboard-parts model."""

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

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.exists():
            raise BackendError(
                f"YOLO-OBB checkpoint not found at {self.checkpoint_path}. "
                "Train one with: modal run tabvision/scripts/train/yolo_guitar_obb_modal.py "
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

    def predict_all(self, frame: np.ndarray) -> OBBPredictions:
        """Run the model once and return detections grouped by class.

        This is the workhorse that the other ``detect_*`` methods delegate
        to — calling them sequentially would mean re-running inference
        per class, which costs ~10× more wallclock per frame.
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
            return OBBPredictions()
        result = results[0]
        names = getattr(result, "names", None) or {}
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return OBBPredictions()

        xywhr = _to_numpy(obb.xywhr)
        confs = _to_numpy(obb.conf)
        clses = _to_numpy(obb.cls).astype(int)
        if xywhr.size == 0:
            return OBBPredictions()

        # Group detections by class, then sort each group by descending conf.
        buckets: dict[str, list[OBBDetection]] = {c: [] for c in ALL_CLASSES}
        for i in range(xywhr.shape[0]):
            cls_idx = int(clses[i])
            cls_name = str(names.get(cls_idx, cls_idx))
            if cls_name not in buckets:
                # Unknown class — skip rather than fall on the floor; the
                # expected vocabulary is the three Roboflow classes.
                continue
            cx, cy, w, h, rad = (float(v) for v in xywhr[i, :5])
            buckets[cls_name].append(
                OBBDetection(
                    class_name=cls_name,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    rotation_deg=float(np.degrees(rad)),
                    confidence=float(confs[i]),
                )
            )
        for k in buckets:
            buckets[k].sort(key=lambda d: d.confidence, reverse=True)

        return OBBPredictions(
            frets=buckets[CLASS_FRET],
            neck=buckets[CLASS_NECK],
            nut=buckets[CLASS_NUT],
        )

    def detect(self, frame: np.ndarray) -> GuitarBBox | None:
        """Detect the most-confident *neck* (= guitar proxy) in ``frame``.

        Returns None if no neck detection meets the confidence threshold.
        Per :file:`docs/DECISIONS.md` the neck class proxies for the
        spec's GuitarBBox; non-neck classes are dropped here.
        """
        preds = self.predict_all(frame)
        best = preds.best_neck()
        return best.to_guitar_bbox() if best is not None else None

    def detect_neck(self, frame: np.ndarray) -> OBBDetection | None:
        """Highest-confidence neck OBB, or None."""
        return self.predict_all(frame).best_neck()

    def detect_frets(self, frame: np.ndarray) -> list[OBBDetection]:
        """All fret OBBs, sorted by descending confidence."""
        return self.predict_all(frame).frets

    def detect_nut(self, frame: np.ndarray) -> OBBDetection | None:
        """Highest-confidence nut OBB, or None."""
        return self.predict_all(frame).best_nut()


def _to_numpy(arr) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Coerce ultralytics tensor-or-array attributes to numpy."""
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


__all__ = [
    "YoloOBBBackend",
    "OBBDetection",
    "OBBPredictions",
    "DEFAULT_CHECKPOINT_ENV",
    "DEFAULT_CHECKPOINT_NAME",
    "CLASS_FRET",
    "CLASS_NECK",
    "CLASS_NUT",
    "ALL_CLASSES",
]
