"""Phase A crop-then-detect geometry: coordinate mapping + merge/dedupe.

The F2b lesson is that the largest video errors on record were coordinate-system
bugs, not noise — so the crop↔full-frame round trip is pinned here before any
cache is built with it.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.gaps_cv_cache import (
    _dedupe_frets,
    crop_rect_for_neck,
    map_crop_detection,
    merge_crop_predictions,
    obb_corner_bounds,
    rawcv_cache_path,
)
from tabvision.video.guitar.yolo_backend import OBBDetection, OBBPredictions


def _det(
    class_name: str = "fret",
    cx: float = 100.0,
    cy: float = 50.0,
    w: float = 40.0,
    h: float = 10.0,
    rotation_deg: float = 0.0,
    confidence: float = 0.5,
) -> OBBDetection:
    return OBBDetection(
        class_name=class_name,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        rotation_deg=rotation_deg,
        confidence=confidence,
    )


class TestObbCornerBounds:
    def test_axis_aligned(self) -> None:
        bounds = obb_corner_bounds(_det(cx=100, cy=50, w=40, h=10, rotation_deg=0))
        assert np.allclose(bounds, (80, 45, 120, 55))

    def test_rotated_90_swaps_extents(self) -> None:
        bounds = obb_corner_bounds(_det(cx=100, cy=50, w=40, h=10, rotation_deg=90))
        assert np.allclose(bounds, (95, 30, 105, 70))

    def test_rotated_45_grows_both(self) -> None:
        xmin, ymin, xmax, ymax = obb_corner_bounds(_det(cx=0, cy=0, w=40, h=10, rotation_deg=45))
        half = (40 + 10) / 2 / np.sqrt(2)
        assert np.allclose((xmin, ymin, xmax, ymax), (-half, -half, half, half))


class TestCropRectForNeck:
    def test_none_without_neck(self) -> None:
        assert crop_rect_for_neck(OBBPredictions(), 640, 360) is None

    def test_pads_and_clips_to_frame(self) -> None:
        preds = OBBPredictions(neck=[_det("neck", cx=30, cy=20, w=60, h=30)])
        rect = crop_rect_for_neck(preds, 640, 360, pad_frac=0.5)
        assert rect is not None
        x0, y0, x1, y1 = rect
        assert x0 == 0 and y0 == 0  # clipped at the frame origin
        assert x1 <= 640 and y1 <= 360
        assert x1 > 60 and y1 > 35  # padding extends beyond the raw OBB

    def test_contains_rotated_neck_corners(self) -> None:
        neck = _det("neck", cx=320, cy=180, w=300, h=40, rotation_deg=30)
        rect = crop_rect_for_neck(OBBPredictions(neck=[neck]), 640, 360, pad_frac=0.0)
        assert rect is not None
        x0, y0, x1, y1 = rect
        xmin, ymin, xmax, ymax = obb_corner_bounds(neck)
        assert x0 <= xmin and y0 <= ymin and x1 >= xmax and y1 >= ymax


class TestMapCropDetection:
    def test_round_trip(self) -> None:
        original = _det(cx=250.0, cy=140.0, w=8.0, h=60.0, rotation_deg=87.0)
        x0, y0, scale = 200.0, 100.0, 3.0
        crop_coords = OBBDetection(
            class_name=original.class_name,
            cx=(original.cx - x0) * scale,
            cy=(original.cy - y0) * scale,
            w=original.w * scale,
            h=original.h * scale,
            rotation_deg=original.rotation_deg,
            confidence=original.confidence,
        )
        mapped = map_crop_detection(crop_coords, x0, y0, scale, scale)
        assert np.isclose(mapped.cx, original.cx)
        assert np.isclose(mapped.cy, original.cy)
        assert np.isclose(mapped.w, original.w)
        assert np.isclose(mapped.h, original.h)
        assert mapped.rotation_deg == original.rotation_deg

    def test_anisotropic_centers_map_per_axis(self) -> None:
        det = _det(cx=30.0, cy=40.0)
        mapped = map_crop_detection(det, 10.0, 20.0, 3.0, 2.0)
        assert np.isclose(mapped.cx, 10.0 + 30.0 / 3.0)
        assert np.isclose(mapped.cy, 20.0 + 40.0 / 2.0)


class TestDedupeFrets:
    def test_duplicate_resolved_to_higher_confidence(self) -> None:
        dets = [
            _det(cx=100, confidence=0.3),
            _det(cx=120, confidence=0.9),
            _det(cx=101, cy=50.5, confidence=0.6),  # duplicate of the cx=100 wire
            _det(cx=140, confidence=0.5),
            _det(cx=160, confidence=0.4),
        ]
        kept = _dedupe_frets(dets)
        xs = sorted(round(d.cx) for d in kept)
        assert xs == [101, 120, 140, 160]
        by_x = {round(d.cx): d.confidence for d in kept}
        assert by_x[101] == 0.6  # the higher-confidence duplicate wins

    def test_singleton_and_empty(self) -> None:
        assert _dedupe_frets([]) == []
        one = [_det()]
        assert _dedupe_frets(one) == one


class TestMergeCropPredictions:
    def test_neck_from_full_frame_only(self) -> None:
        full = OBBPredictions(neck=[_det("neck", cx=300, confidence=0.8)])
        crop = OBBPredictions(neck=[_det("neck", cx=310, confidence=0.99)])
        merged = merge_crop_predictions(full, crop, x0=0.0, y0=0.0, sx=1.0, sy=1.0)
        assert len(merged.neck) == 1
        assert merged.neck[0].cx == 300

    def test_crop_contributes_frets_and_nut(self) -> None:
        full = OBBPredictions(
            frets=[_det(cx=100, confidence=0.4)],
            neck=[_det("neck", cx=300, confidence=0.8)],
            nut=[_det("nut", cx=40, confidence=0.2)],
        )
        crop = OBBPredictions(
            frets=[_det(cx=130, confidence=0.15), _det(cx=160, confidence=0.12)],
            nut=[_det("nut", cx=41, confidence=0.7)],
        )
        merged = merge_crop_predictions(full, crop, x0=0.0, y0=0.0, sx=1.0, sy=1.0)
        assert sorted(round(d.cx) for d in merged.frets) == [100, 130, 160]
        assert merged.nut[0].confidence == 0.7  # sorted desc, crop nut first
        confs = [d.confidence for d in merged.frets]
        assert confs == sorted(confs, reverse=True)

    def test_mapping_applied_before_merge(self) -> None:
        full = OBBPredictions(neck=[_det("neck", cx=300, confidence=0.8)])
        crop = OBBPredictions(frets=[_det(cx=90.0, cy=120.0, confidence=0.3)])
        merged = merge_crop_predictions(full, crop, x0=200.0, y0=100.0, sx=3.0, sy=3.0)
        assert np.isclose(merged.frets[0].cx, 200.0 + 90.0 / 3.0)
        assert np.isclose(merged.frets[0].cy, 100.0 + 120.0 / 3.0)


class TestCachePathSuffix:
    def test_default_name_unchanged(self, tmp_path) -> None:  # noqa: ANN001
        p = rawcv_cache_path(tmp_path, "027_Zpswc", 0.25)
        assert p.name == "027_Zpswc.rawcv.c0.25.pkl"

    def test_crop_suffix(self, tmp_path) -> None:  # noqa: ANN001
        p = rawcv_cache_path(tmp_path, "027_Zpswc", 0.25, suffix=".crop")
        assert p.name == "027_Zpswc.rawcv.c0.25.crop.pkl"
