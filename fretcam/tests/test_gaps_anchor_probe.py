from __future__ import annotations

import math

import pytest

from fretcam.gaps_anchor_probe import (
    fret_in_position_window,
    nearest_cached_frame,
    position_from_centroid,
    wilson_interval,
)


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
