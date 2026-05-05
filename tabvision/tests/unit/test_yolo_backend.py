"""Unit tests for tabvision.video.guitar.yolo_backend.

Real inference is exercised in a separate (gated) integration test once
a trained checkpoint is available; here we verify construction, error
paths, and the pure-Python multi-class result-parsing logic by injecting
a fake ultralytics-style result object. AGPL-tainted ultralytics doesn't
need to be installed for the core test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tabvision.errors import BackendError
from tabvision.video.guitar.yolo_backend import (
    ALL_CLASSES,
    CLASS_FRET,
    CLASS_NECK,
    CLASS_NUT,
    DEFAULT_CHECKPOINT_ENV,
    OBBDetection,
    OBBPredictions,
    YoloOBBBackend,
    _default_checkpoint_path,
)

# ----- construction / error path -----


def test_default_checkpoint_path_uses_data_root(monkeypatch):
    monkeypatch.setenv("TABVISION_DATA_ROOT", "/tmp/tv-test-root")
    monkeypatch.delenv(DEFAULT_CHECKPOINT_ENV, raising=False)
    p = _default_checkpoint_path()
    assert str(p).endswith("/tmp/tv-test-root/models/guitar-yolo-obb-finetuned.pt")


def test_env_overrides_default_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv(DEFAULT_CHECKPOINT_ENV, str(tmp_path / "custom.pt"))
    p = _default_checkpoint_path()
    assert p == tmp_path / "custom.pt"


def test_missing_checkpoint_raises_helpful_backend_error(tmp_path):
    """detect() without a trained checkpoint should raise BackendError
    with instructions, not a cryptic FileNotFoundError."""
    backend = YoloOBBBackend(checkpoint_path=tmp_path / "does-not-exist.pt")
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(BackendError, match="train"):
        backend.detect(fake_frame)


def test_constructor_does_not_load_model(tmp_path):
    """Construction must be cheap; model load is lazy."""
    backend = YoloOBBBackend(checkpoint_path=tmp_path / "x.pt")
    assert backend._model is None


# ----- multi-class result parsing -----


@dataclass
class _FakeOBB:
    xywhr: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xywhr.shape[0])


@dataclass
class _FakeResult:
    obb: Any
    names: dict[int, str]


class _FakeModel:
    """Minimal stand-in for ``ultralytics.YOLO`` — exposes only ``predict``."""

    def __init__(self, results: list[_FakeResult]):
        self._results = results

    def predict(self, **_kwargs):  # type: ignore[no-untyped-def]
        return self._results


def _backend_with_fake(model) -> YoloOBBBackend:
    """YoloOBBBackend wired to a pre-loaded fake model (skips checkpoint load)."""
    backend = YoloOBBBackend(checkpoint_path="/does/not/matter.pt")
    backend._model = model
    return backend


FakeDetTuple = tuple[str, float, float, float, float, float, float]


def _make_fake_result(detections: list[FakeDetTuple]) -> _FakeResult:
    """Build a fake ultralytics result from list of
    (class_name, cx, cy, w, h, rotation_rad, conf) tuples."""
    names = {0: CLASS_FRET, 1: CLASS_NECK, 2: CLASS_NUT}
    name_to_idx = {v: k for k, v in names.items()}
    if not detections:
        return _FakeResult(
            obb=_FakeOBB(
                xywhr=np.zeros((0, 5)), conf=np.zeros((0,)), cls=np.zeros((0,), dtype=int)
            ),
            names=names,
        )
    xywhr = np.array([[cx, cy, w, h, rad] for _, cx, cy, w, h, rad, _ in detections])
    conf = np.array([c for *_, c in detections])
    cls = np.array([name_to_idx[name] for name, *_ in detections], dtype=int)
    return _FakeResult(obb=_FakeOBB(xywhr=xywhr, conf=conf, cls=cls), names=names)


def test_predict_all_groups_and_sorts_by_confidence():
    """Detections are bucketed by class and each bucket is sorted desc by conf."""
    fake_result = _make_fake_result([
        # class       cx    cy    w     h    rad   conf
        (CLASS_FRET, 100, 200, 30, 5, 0.0, 0.40),
        (CLASS_FRET, 130, 200, 30, 5, 0.0, 0.85),  # higher-conf fret should come first
        (CLASS_NECK, 200, 200, 400, 60, 0.05, 0.95),
        (CLASS_NUT, 50, 200, 8, 60, 0.05, 0.70),
    ])
    backend = _backend_with_fake(_FakeModel([fake_result]))
    preds = backend.predict_all(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(preds, OBBPredictions)
    assert [d.confidence for d in preds.frets] == [0.85, 0.40]
    assert preds.best_neck() is not None and preds.best_neck().confidence == pytest.approx(0.95)
    assert preds.best_nut() is not None and preds.best_nut().confidence == pytest.approx(0.70)
    assert all(d.class_name in ALL_CLASSES for d in (*preds.frets, *preds.neck, *preds.nut))


def test_predict_all_returns_empty_when_no_detections():
    backend = _backend_with_fake(_FakeModel([_make_fake_result([])]))
    preds = backend.predict_all(np.zeros((480, 640, 3), dtype=np.uint8))
    assert preds.frets == []
    assert preds.neck == []
    assert preds.nut == []
    assert preds.best_neck() is None
    assert preds.best_nut() is None


def test_predict_all_no_results_at_all():
    """Some YOLO modes return an empty results list (no frame processed)."""
    backend = _backend_with_fake(_FakeModel([]))
    preds = backend.predict_all(np.zeros((480, 640, 3), dtype=np.uint8))
    assert preds.frets == []
    assert preds.neck == []
    assert preds.nut == []


def test_detect_returns_only_neck_class():
    """detect() must filter to neck — never return a fret or nut as the GuitarBBox."""
    fake_result = _make_fake_result([
        # A fret with HIGHER confidence than the neck — detect() must still pick neck.
        (CLASS_FRET, 100, 200, 30, 5, 0.0, 0.99),
        (CLASS_NECK, 200, 200, 400, 60, 0.05, 0.85),
        (CLASS_NUT, 50, 200, 8, 60, 0.05, 0.95),
    ])
    backend = _backend_with_fake(_FakeModel([fake_result]))
    bbox = backend.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert bbox is not None
    # Center-form -> top-left form: x = cx - w/2, y = cy - h/2.
    assert bbox.x == pytest.approx(200 - 400 / 2)
    assert bbox.y == pytest.approx(200 - 60 / 2)
    assert bbox.w == pytest.approx(400)
    assert bbox.h == pytest.approx(60)
    assert bbox.confidence == pytest.approx(0.85)
    assert bbox.rotation_deg == pytest.approx(np.degrees(0.05))


def test_detect_returns_none_when_no_neck():
    """A frame with frets/nut but no neck must yield None — we can't crop without it."""
    fake_result = _make_fake_result([
        (CLASS_FRET, 100, 200, 30, 5, 0.0, 0.50),
        (CLASS_NUT, 50, 200, 8, 60, 0.05, 0.90),
    ])
    backend = _backend_with_fake(_FakeModel([fake_result]))
    assert backend.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_unknown_class_is_skipped_not_fatal():
    """A model that emits a class outside ALL_CLASSES (e.g. corrupt
    metadata) shouldn't crash — just drop the unknown detection."""
    fake_result = _make_fake_result([(CLASS_NECK, 200, 200, 400, 60, 0.0, 0.9)])
    # Mutate names so cls=99 maps to an unknown label, then add a row at cls=99.
    fake_result.names[99] = "rogue_class"
    fake_result.obb.xywhr = np.vstack([fake_result.obb.xywhr, [[10, 10, 5, 5, 0.0]]])
    fake_result.obb.conf = np.append(fake_result.obb.conf, 0.99)
    fake_result.obb.cls = np.append(fake_result.obb.cls, 99)
    backend = _backend_with_fake(_FakeModel([fake_result]))
    preds = backend.predict_all(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(preds.neck) == 1
    assert len(preds.frets) == 0
    assert len(preds.nut) == 0


def test_detect_frets_returns_all_sorted():
    """detect_frets() exposes the full set, sorted by descending confidence."""
    fake_result = _make_fake_result([
        (CLASS_FRET, 100, 200, 30, 5, 0.0, 0.30),
        (CLASS_FRET, 130, 200, 30, 5, 0.0, 0.95),
        (CLASS_FRET, 160, 200, 30, 5, 0.0, 0.60),
        (CLASS_NECK, 200, 200, 400, 60, 0.0, 0.99),
    ])
    backend = _backend_with_fake(_FakeModel([fake_result]))
    frets = backend.detect_frets(np.zeros((480, 640, 3), dtype=np.uint8))
    assert [round(d.confidence, 2) for d in frets] == [0.95, 0.60, 0.30]
    assert all(d.class_name == CLASS_FRET for d in frets)


def test_obbdetection_to_guitar_bbox_round_trip():
    """The center->top-left conversion in to_guitar_bbox is the canonical map."""
    det = OBBDetection(
        class_name=CLASS_NECK, cx=100.0, cy=50.0, w=80.0, h=20.0,
        rotation_deg=12.5, confidence=0.7,
    )
    bbox = det.to_guitar_bbox()
    assert bbox.x == pytest.approx(100 - 80 / 2)
    assert bbox.y == pytest.approx(50 - 20 / 2)
    assert bbox.w == pytest.approx(80)
    assert bbox.h == pytest.approx(20)
    assert bbox.rotation_deg == pytest.approx(12.5)
    assert bbox.confidence == pytest.approx(0.7)
