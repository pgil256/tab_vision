"""Unit tests for the WS4 string-dataset extraction geometry helper.

The crop-rect math is pure; full extraction (ffmpeg/YOLO/video) is covered by the
script's smoke run, not here.
"""

from __future__ import annotations

import numpy as np

from scripts.train.extract_string_dataset import neck_crop_rect


def _box(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array(
        [
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
        ]
    )


def test_neck_crop_rect_pads_and_clamps():
    # A 100x40 box centred at (200,150); 0.5 pad => +50 x / +20 y each side.
    rect = neck_crop_rect([_box(200, 150, 100, 40)], 640, 480, pad_frac=0.5)
    assert rect == (100, 110, 300, 190)


def test_neck_crop_rect_clamps_to_frame_bounds():
    rect = neck_crop_rect([_box(20, 20, 100, 40)], 640, 480, pad_frac=0.5)
    assert rect is not None
    x0, y0, x1, y1 = rect
    assert x0 == 0 and y0 == 0  # clamped at the top-left edge
    assert x1 <= 640 and y1 <= 480


def test_neck_crop_rect_uses_median_over_samples():
    # One wild outlier box must not move the median crop.
    boxes = [_box(200, 150, 100, 40)] * 5 + [_box(600, 400, 100, 40)]
    rect = neck_crop_rect(boxes, 640, 480, pad_frac=0.0)
    assert rect == (150, 130, 250, 170)  # the (200,150) cluster, outlier ignored


def test_neck_crop_rect_empty_returns_none():
    assert neck_crop_rect([], 640, 480) is None


def test_neck_crop_rect_degenerate_box_returns_none():
    assert neck_crop_rect([_box(10, 10, 2, 2)], 640, 480, pad_frac=0.0) is None
