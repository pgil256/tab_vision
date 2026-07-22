from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fretcam.gaps_anchor_probe import (
    classify_probe,
    corrected_cached_anchor,
    fret_in_position_window,
    nearest_cached_frame,
    position_from_centroid,
    wilson_interval,
)
from tabvision.types import GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import FingerSample, HandSample


def test_position_and_window_keep_open_strings_possible() -> None:
    assert position_from_centroid(5.9) == 5
    assert position_from_centroid(0.2) == 1
    assert fret_in_position_window(0, 12.0)
    assert fret_in_position_window(4, 5.9)
    assert fret_in_position_window(9, 5.9)
    assert not fret_in_position_window(3, 5.9)
    assert not fret_in_position_window(10, 5.9)


def test_position_rejects_non_finite_centroid() -> None:
    with pytest.raises(ValueError, match="finite"):
        position_from_centroid(math.nan)


def test_nearest_cached_frame_targets_pre_onset_and_prefers_earlier_tie() -> None:
    frames = [8, 9, 10, 11]
    assert nearest_cached_frame(frames, target_s=0.95, fps=10.0) == 9
    assert nearest_cached_frame(frames, target_s=1.01, fps=10.0) == 10
    assert nearest_cached_frame(frames, target_s=2.0, fps=10.0) is None


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(387, 1566)
    assert lower == pytest.approx(0.2264, abs=1e-4)
    assert upper == pytest.approx(0.2691, abs=1e-4)
    assert lower < 387 / 1566 < upper


def test_probe_classification_uses_fixed_a14_comparator_interval() -> None:
    assert classify_probe(387, 1566) == "negative"
    assert classify_probe(1195, 1566) == "positive"


def _cached_record(x: float) -> SimpleNamespace:
    hand = HandSample(
        wrist_xy=(x, 25.0),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=1.0,
        fingers={
            name: FingerSample(name, (x, 25.0), 0.0, 0.8)
            for name in ("index", "middle", "ring", "pinky")
        },
    )
    return SimpleNamespace(preds=object(), hand=hand)


def test_corrected_cache_path_uses_descending_calibrated_fret_map() -> None:
    homography = Homography(
        H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
        confidence=1.0,
        method="fixture",
    )
    centers = np.linspace(0.9, 0.1, 25)

    anchor = corrected_cached_anchor(
        _cached_record(20.0),
        GuitarConfig(),
        calibrator=lambda _preds, _cfg: (homography, centers),
    )

    assert anchor.center_fret == pytest.approx(21.0)
    assert anchor.method == "mediapipe_calibrated_fret_map"


def test_corrected_cache_path_fallback_maps_body_joint_to_fret_twelve() -> None:
    homography = Homography(
        H=np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]),
        confidence=1.0,
        method="fixture",
    )

    anchor = corrected_cached_anchor(
        _cached_record(100.0),
        GuitarConfig(),
        calibrator=lambda _preds, _cfg: (homography, None),
    )

    assert anchor.center_fret == pytest.approx(12.0)
    assert anchor.method == "mediapipe_rule18_fret12_fallback"
