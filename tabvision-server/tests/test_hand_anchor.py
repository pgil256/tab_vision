"""Tests for hand_anchor module."""
import math
import pytest

from app.video_pipeline import FingerPosition, HandObservation
from app.fretboard_detection import FretboardGeometry
from app.hand_anchor import (
    HandAnchorPoint,
    build_hand_position_timeline,
    get_hand_anchor_at,
    project_palm_to_fret,
    _interpolate_fret_from_rel_x,
)


# --- Fixtures --------------------------------------------------------------


def make_horizontal_fretboard(
    frame_width: int = 640,
    frame_height: int = 480,
    num_frets: int = 12,
    starting_fret: int = 0,
    y_top: float = 180,
    y_bottom: float = 300,
    x_left: float = 60,
    x_right: float = 600,
    confidence: float = 0.9,
) -> FretboardGeometry:
    """Build a clean horizontal fretboard geometry with evenly-spaced frets."""
    # Fret positions normalized [0, 1] along the neck. Evenly-spaced is not
    # physically accurate (real frets follow 1/(2^(n/12))) but it's fine for
    # unit-testing the anchor projection math.
    fret_positions = [i / (num_frets - 1) for i in range(num_frets)]
    actual_fret_numbers = [starting_fret + i for i in range(num_frets)]
    # 6 strings
    string_positions = [i / 5 for i in range(6)]

    return FretboardGeometry(
        top_left=(x_left, y_top),
        top_right=(x_right, y_top),
        bottom_left=(x_left, y_bottom),
        bottom_right=(x_right, y_bottom),
        fret_positions=fret_positions,
        string_positions=string_positions,
        detection_confidence=confidence,
        frame_width=frame_width,
        frame_height=frame_height,
        actual_fret_numbers=actual_fret_numbers,
        starting_fret=starting_fret,
    )


def make_hand_observation(
    timestamp: float,
    palm_x_norm: float,
    palm_y_norm: float,
    hand_confidence: float = 0.9,
    fingers_extended: int = 4,
) -> HandObservation:
    """Build a hand observation where extended non-thumb fingertips cluster around palm_x_norm.

    Places fingertips within a small x-neighborhood of palm_x_norm so the
    centroid equals palm_x_norm.
    """
    fingers = [
        FingerPosition(finger_id=0, x=palm_x_norm - 0.02, y=palm_y_norm + 0.05, z=0.0, is_extended=False),
    ]
    # Non-thumb fingers, centered so their mean x == palm_x_norm.
    offsets = [-0.015, -0.005, 0.005, 0.015][:fingers_extended]
    # Pad with curled fingers if fingers_extended < 4
    for i in range(1, 5):
        if i - 1 < len(offsets):
            fingers.append(FingerPosition(
                finger_id=i,
                x=palm_x_norm + offsets[i - 1],
                y=palm_y_norm,
                z=-0.03,
                is_extended=True,
            ))
        else:
            fingers.append(FingerPosition(
                finger_id=i,
                x=palm_x_norm + 0.1,
                y=palm_y_norm,
                z=0.0,
                is_extended=False,
            ))
    return HandObservation(
        timestamp=timestamp,
        fingers=fingers,
        is_left_hand=True,
        wrist_position=(palm_x_norm - 0.05, palm_y_norm + 0.1, 0.0),
        hand_confidence=hand_confidence,
    )


# --- _interpolate_fret_from_rel_x -----------------------------------------


class TestInterpolateFret:

    def test_exact_on_fret_position_returns_that_fret(self):
        # 4 evenly-spaced detected frets at 0, 0.33, 0.66, 1.0 mapping to frets 0-3
        positions = [0.0, 1 / 3, 2 / 3, 1.0]
        actual = [0, 1, 2, 3]
        result = _interpolate_fret_from_rel_x(1 / 3, positions, actual)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_midpoint_between_frets_returns_fractional(self):
        positions = [0.0, 0.5, 1.0]
        actual = [0, 1, 2]
        # Halfway between fret 1 (0.5) and fret 2 (1.0) → 1.5
        result = _interpolate_fret_from_rel_x(0.75, positions, actual)
        assert result == pytest.approx(1.5, abs=1e-6)

    def test_starting_fret_offset(self):
        # Partial fretboard starting at fret 3
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        actual = [3, 4, 5, 6, 7]
        result = _interpolate_fret_from_rel_x(0.5, positions, actual)
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_past_last_fret_extrapolates(self):
        positions = [0.0, 0.5, 1.0]
        actual = [0, 1, 2]
        # 0.2 past the last position (1.0) with 0.5/fret gap → fret 2 + 0.4 = 2.4
        result = _interpolate_fret_from_rel_x(1.2, positions, actual)
        assert result == pytest.approx(2.4, abs=1e-6)

    def test_before_first_fret_extrapolates(self):
        positions = [0.5, 1.0]
        actual = [1, 2]
        # 0.25 before position 0.5 with 0.5/fret gap → fret 1 - 0.5 = 0.5
        result = _interpolate_fret_from_rel_x(0.25, positions, actual)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_empty_fret_positions_returns_none(self):
        assert _interpolate_fret_from_rel_x(0.5, [], [0]) is None

    def test_single_fret_position_returns_none(self):
        assert _interpolate_fret_from_rel_x(0.5, [0.5], [1]) is None


# --- project_palm_to_fret -------------------------------------------------


class TestProjectPalmToFret:

    def test_palm_centered_on_fret_5(self):
        """Palm at x=0.5 of the fretboard maps to the middle of 12 detected frets."""
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600)
        # Fretboard spans pixel x = 60 to 600 (width 540). Middle = 330.
        # Normalized x in a 640-wide frame = 330/640 ≈ 0.5156.
        palm_x_norm = 330 / 640
        palm_y_norm = 240 / 480  # Middle of fretboard vertically
        obs = make_hand_observation(timestamp=0.0, palm_x_norm=palm_x_norm, palm_y_norm=palm_y_norm)
        fret, conf = project_palm_to_fret(obs, fb)
        # 12 frets evenly spaced over [0, 1]; middle = 5.5
        assert fret == pytest.approx(5.5, abs=0.1)
        assert conf > 0.7

    def test_palm_off_fretboard_returns_none(self):
        fb = make_horizontal_fretboard()
        # Palm way off to the right
        obs = make_hand_observation(timestamp=0.0, palm_x_norm=1.5, palm_y_norm=0.5)
        fret, conf = project_palm_to_fret(obs, fb)
        assert fret is None
        assert conf == 0.0

    def test_fewer_than_min_extended_fingers_falls_back_to_wrist(self):
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600)
        # Only 1 extended non-thumb finger → fallback to wrist
        obs = make_hand_observation(
            timestamp=0.0,
            palm_x_norm=330 / 640,
            palm_y_norm=240 / 480,
            fingers_extended=1,
        )
        # Wrist is offset by -0.05 in x in our fixture
        fret, conf = project_palm_to_fret(obs, fb)
        # Should still return a fret (from wrist), not None
        assert fret is not None
        assert conf > 0.0

    def test_partial_fretboard_uses_actual_fret_numbers(self):
        # Fretboard starts at fret 5, covers frets 5-12 (8 detected frets)
        fb = make_horizontal_fretboard(num_frets=8, starting_fret=5, x_left=60, x_right=600)
        palm_x_norm = 330 / 640  # Middle of fretboard
        palm_y_norm = 240 / 480
        obs = make_hand_observation(timestamp=0.0, palm_x_norm=palm_x_norm, palm_y_norm=palm_y_norm)
        fret, conf = project_palm_to_fret(obs, fb)
        # Middle of 8 frets starting at 5 = fret 8.5
        assert fret == pytest.approx(8.5, abs=0.1)

    def test_confidence_scales_with_hand_and_fretboard(self):
        fb = make_horizontal_fretboard(confidence=0.5)
        obs = make_hand_observation(
            timestamp=0.0,
            palm_x_norm=330 / 640,
            palm_y_norm=240 / 480,
            hand_confidence=0.6,
        )
        _, conf = project_palm_to_fret(obs, fb)
        # Rough upper bound: 0.5 * 0.6 * 1.0 = 0.3
        assert conf == pytest.approx(0.3, abs=0.05)


# --- build_hand_position_timeline ----------------------------------------


class TestBuildTimeline:

    def test_empty_observations(self):
        fb = make_horizontal_fretboard()
        assert build_hand_position_timeline({}, fb) == []

    def test_none_fretboard(self):
        obs = {0.0: make_hand_observation(0.0, 0.5, 0.5)}
        assert build_hand_position_timeline(obs, None) == []

    def test_steady_hand_produces_smooth_anchor(self):
        """5 samples with palm at the same x position → timeline converges on matching fret."""
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600)
        palm_x = 330 / 640  # middle of fretboard → fret ~5.5
        observations = {
            i * 0.1: make_hand_observation(i * 0.1, palm_x, 0.5)
            for i in range(5)
        }
        timeline = build_hand_position_timeline(observations, fb)
        assert len(timeline) == 5
        # After smoothing, every point should be close to the true anchor
        for p in timeline:
            assert p.anchor_fret == pytest.approx(5.5, abs=0.5)

    def test_outlier_spike_is_rejected(self):
        """One bad sample at fret ~11 among stable fret ~5 neighbors is dropped."""
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600)
        palm_steady = 330 / 640        # → fret ~5.5
        palm_spike = 570 / 640         # → fret ~10.4 (way off)

        observations = {}
        # 4 steady samples before the spike
        for i in range(4):
            t = i * 0.1
            observations[t] = make_hand_observation(t, palm_steady, 0.5, hand_confidence=0.6)
        # 1 spike sample with low confidence (triggers outlier rule)
        observations[0.4] = make_hand_observation(0.4, palm_spike, 0.5, hand_confidence=0.55)
        # 4 steady samples after
        for i in range(5, 9):
            t = i * 0.1
            observations[t] = make_hand_observation(t, palm_steady, 0.5, hand_confidence=0.6)

        timeline = build_hand_position_timeline(observations, fb)
        timestamps = [p.timestamp for p in timeline]
        assert 0.4 not in timestamps, "outlier spike should have been dropped"
        # Retained samples should all be near fret 5.5
        for p in timeline:
            assert p.anchor_fret == pytest.approx(5.5, abs=1.0)

    def test_genuine_shift_is_preserved(self):
        """Samples moving from fret ~5 to fret ~8 over ~1s — timeline follows."""
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600)
        # Walk palm x from 330/640 (fret ~5.5) to 480/640 (fret ~8.2) over 10 samples
        palm_values = [330 + 15 * i for i in range(11)]  # 330, 345, ..., 480
        observations = {
            i * 0.1: make_hand_observation(i * 0.1, px / 640, 0.5, hand_confidence=0.9)
            for i, px in enumerate(palm_values)
        }
        timeline = build_hand_position_timeline(observations, fb)
        assert len(timeline) == 11
        # After smoothing over ~1s, the anchor should have moved > 1.5 frets
        assert timeline[-1].anchor_fret - timeline[0].anchor_fret > 1.5

    def test_low_confidence_samples_dropped(self):
        fb = make_horizontal_fretboard(num_frets=12, x_left=60, x_right=600, confidence=0.2)
        obs = {0.0: make_hand_observation(0.0, 0.5, 0.5, hand_confidence=0.3)}
        # fretboard 0.2 * hand 0.3 = 0.06 — well below MIN_FRAME_CONFIDENCE
        timeline = build_hand_position_timeline(obs, fb)
        assert timeline == []


# --- get_hand_anchor_at --------------------------------------------------


class TestGetHandAnchorAt:

    def test_empty_timeline_returns_none(self):
        fret, conf = get_hand_anchor_at([], 1.0)
        assert fret is None
        assert conf == 0.0

    def test_query_between_points_linearly_interpolates(self):
        timeline = [
            HandAnchorPoint(timestamp=0.0, anchor_fret=5.0, confidence=0.8),
            HandAnchorPoint(timestamp=0.2, anchor_fret=7.0, confidence=0.8),
        ]
        # Both bracketing points within default max_gap=0.3 — interpolate.
        fret, conf = get_hand_anchor_at(timeline, 0.1)
        assert fret == pytest.approx(6.0, abs=1e-6)
        assert conf == pytest.approx(0.8, abs=1e-6)

    def test_query_within_gap_of_nearer_point(self):
        """One bracketing point close, the other far — interpolation still happens."""
        timeline = [
            HandAnchorPoint(timestamp=0.0, anchor_fret=5.0, confidence=0.8),
            HandAnchorPoint(timestamp=1.0, anchor_fret=7.0, confidence=0.8),
        ]
        # Query at 0.1: nearer point is 0.1s away (within gap), farther is 0.9s (out of gap).
        # Per our semantics, as long as ONE bracketing point is within max_gap, we interpolate.
        fret, conf = get_hand_anchor_at(timeline, 0.1, max_gap=0.3)
        assert fret == pytest.approx(5.2, abs=1e-6)

    def test_query_outside_gap_returns_none(self):
        timeline = [
            HandAnchorPoint(timestamp=0.0, anchor_fret=5.0, confidence=0.8),
            HandAnchorPoint(timestamp=2.0, anchor_fret=7.0, confidence=0.8),
        ]
        # Query at 1.0 is 1.0s from each bracketing point — beyond default max_gap 0.3.
        fret, conf = get_hand_anchor_at(timeline, 1.0, max_gap=0.3)
        assert fret is None
        assert conf == 0.0

    def test_query_before_first_within_gap(self):
        timeline = [HandAnchorPoint(timestamp=1.0, anchor_fret=5.0, confidence=0.7)]
        fret, conf = get_hand_anchor_at(timeline, 0.9, max_gap=0.3)
        assert fret == 5.0
        assert conf == 0.7

    def test_query_before_first_outside_gap(self):
        timeline = [HandAnchorPoint(timestamp=1.0, anchor_fret=5.0, confidence=0.7)]
        fret, conf = get_hand_anchor_at(timeline, 0.0, max_gap=0.3)
        assert fret is None

    def test_query_after_last_within_gap(self):
        timeline = [HandAnchorPoint(timestamp=1.0, anchor_fret=5.0, confidence=0.7)]
        fret, conf = get_hand_anchor_at(timeline, 1.2, max_gap=0.3)
        assert fret == 5.0

    def test_query_exact_timestamp_match(self):
        timeline = [
            HandAnchorPoint(timestamp=0.0, anchor_fret=5.0, confidence=0.8),
            HandAnchorPoint(timestamp=1.0, anchor_fret=7.0, confidence=0.6),
        ]
        fret, conf = get_hand_anchor_at(timeline, 1.0)
        # bisect_left puts us at idx=1, then interpolation between [0] and [1]
        # with w=1.0 → hi values
        assert fret == pytest.approx(7.0, abs=1e-6)
