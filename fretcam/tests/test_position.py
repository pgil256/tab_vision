from __future__ import annotations

import pytest

from fretcam.position import (
    EstimatorConfig,
    PositionEstimator,
    position_window,
    roman_position,
)


def _feed(
    estimator: PositionEstimator,
    frets: list[float | None],
    *,
    confidence: float = 0.8,
    start: float = 0.0,
) -> list:
    return [
        estimator.update(
            index_fret=fret,
            vision_confidence=confidence if fret is not None else 0.0,
            timestamp_s=start + index * 0.1,
        )
        for index, fret in enumerate(frets)
    ]


def test_roman_position_and_window_always_keep_open_strings() -> None:
    assert [roman_position(value) for value in (1, 4, 9, 12, 13, 24)] == [
        "I",
        "IV",
        "IX",
        "XII",
        "XIII",
        "XXIV",
    ]
    assert position_window(5) == (0, 4, 5, 6, 7, 8, 9)
    assert position_window(1) == (0, 1, 2, 3, 4, 5)


def test_initial_lock_uses_elapsed_time_not_a_fixed_frame_count() -> None:
    estimates = _feed(PositionEstimator(), [5.2] * 5)

    assert [estimate.state for estimate in estimates[:3]] == ["acquiring"] * 3
    assert estimates[3].state == "locked"
    assert estimates[3].label == "Position V"
    assert estimates[3].stable_for_ms == pytest.approx(250.0)
    assert estimates[3].window_frets == (0, 4, 5, 6, 7, 8, 9)
    assert estimates[3].confidence == pytest.approx(0.8)

    fast = PositionEstimator()
    fast_estimates = [
        fast.update(
            index_fret=5.2,
            vision_confidence=0.8,
            timestamp_s=index / 30.0,
        )
        for index in range(10)
    ]
    first_fast_lock = next(item for item in fast_estimates if item.state == "locked")
    assert first_fast_lock.timestamp_s == pytest.approx(8 / 30.0)


def test_replaced_initial_candidate_does_not_poison_stable_acquisition() -> None:
    estimator = PositionEstimator()
    estimates = _feed(
        estimator,
        [6.38, 6.40, 6.55, 6.55, 6.55, 6.55],
        confidence=0.28,
    )

    assert all(estimate.position is None for estimate in estimates[:5])
    assert estimates[-1].state == "locked"
    assert estimates[-1].position == 7
    assert estimates[-1].confidence == pytest.approx(0.28)


def test_brief_single_frame_gaps_do_not_restart_acquisition() -> None:
    estimates = _feed(
        PositionEstimator(),
        [5.2, None, 5.2, None, 5.2],
    )

    assert [estimate.state for estimate in estimates[:4]] == ["acquiring"] * 4
    assert estimates[4].state == "locked"
    assert estimates[4].position == 5


def test_boundary_jitter_does_not_flap_the_locked_position() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [5.05] * 5)
    estimates = _feed(
        estimator,
        [4.98, 5.03, 4.96, 5.04, 4.99, 5.02] * 3,
        start=0.5,
    )

    assert all(estimate.state == "locked" for estimate in estimates)
    assert all(estimate.position == 5 for estimate in estimates)
    assert min(estimate.confidence for estimate in estimates) == pytest.approx(0.8)


def test_sub_fret_projection_drift_does_not_change_a_held_position() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [2.4] * 5)

    estimates = _feed(
        estimator,
        [2.41, 2.46, 2.58, 2.63, 2.78, 2.87, 2.64],
        start=0.5,
    )

    assert all(estimate.state == "locked" for estimate in estimates)
    assert all(estimate.position == 2 for estimate in estimates)


def test_shift_uses_elapsed_time_without_intermediate_labels() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.1] * 5)
    estimates = _feed(estimator, [9.2] * 5, start=1.0)

    assert [estimate.state for estimate in estimates[:3]] == ["shifting"] * 3
    assert all(estimate.position is None for estimate in estimates[:3])
    assert all(estimate.label == "Shifting…" for estimate in estimates[:3])
    assert estimates[3].state == "locked"
    assert estimates[3].position == 9
    assert estimates[3].label == "Position IX"

    fast = PositionEstimator()
    _feed(fast, [1.1] * 5)
    fast_estimates = [
        fast.update(
            index_fret=9.2,
            vision_confidence=0.8,
            timestamp_s=1.0 + index / 30.0,
        )
        for index in range(9)
    ]
    first_fast_lock = next(item for item in fast_estimates if item.state == "locked")
    assert 0.25 <= first_fast_lock.timestamp_s - 1.0 <= 0.28


def test_dropout_after_detected_shift_never_republishes_old_position() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [2.0] * 5)

    transition = _feed(estimator, [9.0, None, None, None, None], start=1.0)

    assert all(estimate.state == "shifting" for estimate in transition)
    assert all(estimate.position is None for estimate in transition)
    assert all(estimate.previous_position == 2 for estimate in transition)


def test_weak_displacement_can_clear_an_old_lock_without_asserting_a_new_one() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [2.0] * 5)

    estimate = estimator.update(
        index_fret=6.0,
        vision_confidence=0.19,
        timestamp_s=0.5,
    )

    assert estimate.state == "shifting"
    assert estimate.position is None
    assert estimate.previous_position == 2
    assert estimate.reason == "low_confidence"


def test_cell_centered_coordinate_rounds_to_nearest_position() -> None:
    estimate = _feed(PositionEstimator(), [5.6] * 5)[-1]

    assert estimate.state == "locked"
    assert estimate.position == 6
    assert estimate.label == "Position VI"


def test_isolated_projection_spikes_do_not_reset_a_real_shift() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [2.4] * 5)

    estimates = _feed(
        estimator,
        [5.51, 5.42, 5.61, 6.02, 5.89, 5.85, 19.35, 5.85],
        start=1.0,
    )

    assert all(estimate.position != 19 for estimate in estimates)
    assert estimates[-1].state == "locked"
    assert estimates[-1].position == 6
    assert estimates[-1].label == "Position VI"
    assert max(estimate.smoothed_index_fret or 0.0 for estimate in estimates) < 7.0


def test_sustained_large_jump_is_accepted_after_confirmation() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.0] * 5)

    estimates = _feed(estimator, [12.0] * 6, start=1.0)

    assert estimates[0].position is None
    assert estimates[0].reason == "implausible_jump"
    assert estimates[-1].state == "locked"
    assert estimates[-1].position == 12


def test_alternating_implausible_jumps_never_refresh_an_old_lock() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.0] * 5)

    estimates = _feed(
        estimator,
        [15.0, 24.0] * 6,
        start=1.0,
    )

    assert all(estimate.position is None for estimate in estimates)
    assert all(estimate.state in {"shifting", "acquiring"} for estimate in estimates)
    assert estimates[0].reason == "implausible_jump"


def test_dropouts_hold_then_lose_and_reacquire() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [7.2] * 5)
    held = _feed(estimator, [None] * 5, start=0.5)
    lost = _feed(estimator, [None], start=1.0)[0]

    assert all(estimate.state == "holding" for estimate in held)
    assert all(estimate.position == 7 for estimate in held)
    assert held[-1].confidence < held[0].confidence
    assert lost.state == "lost"
    assert lost.position is None

    reacquired = _feed(estimator, [7.2] * 5, start=1.1)
    assert [estimate.state for estimate in reacquired[:3]] == ["acquiring"] * 3
    assert reacquired[3].state == "locked"
    assert reacquired[3].position == 7


def test_same_position_after_a_long_evidence_gap_must_restabilize() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [5.0] * 5)
    held = _feed(estimator, [None] * 3, start=0.5)

    reacquired = _feed(estimator, [5.0] * 4, start=0.8)

    assert held[-1].state == "holding"
    assert [estimate.state for estimate in reacquired[:3]] == ["acquiring"] * 3
    assert all(estimate.position is None for estimate in reacquired[:3])
    assert reacquired[0].reason == "evidence_gap"
    assert reacquired[3].state == "locked"
    assert reacquired[3].position == 5


def test_low_confidence_observation_abstains_and_explains_why() -> None:
    estimate = PositionEstimator().update(
        index_fret=5.2,
        vision_confidence=0.19,
        timestamp_s=0.0,
    )

    assert estimate.state == "acquiring"
    assert estimate.position is None
    assert estimate.confidence == 0.0
    assert estimate.observation_confidence == pytest.approx(0.19)
    assert estimate.reason == "low_confidence"


def test_motion_change_point_updates_the_readout_without_bypassing_hysteresis() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.0] * 5)
    shifted = _feed(estimator, [9.0], start=1.0)[0]

    assert shifted.state == "shifting"
    assert shifted.reason == "motion_detected"
    assert shifted.smoothed_index_fret == pytest.approx(9.0)


def test_locked_state_never_publishes_below_the_composite_threshold() -> None:
    config = EstimatorConfig(min_vision_confidence=0.20)
    estimator = PositionEstimator(config)
    _feed(estimator, [1.0] * 5)

    shifted = _feed(
        estimator,
        [5.0] * 13,
        confidence=0.21,
        start=1.0,
    )

    assert all(
        estimate.state != "locked"
        or estimate.confidence >= config.min_vision_confidence
        for estimate in shifted
    )
    assert any(estimate.state == "shifting" for estimate in shifted)
    assert shifted[-1].state == "locked"
    assert shifted[-1].position == 5


def test_temporal_shift_behavior_is_consistent_across_frame_rates() -> None:
    def lock_time(fps: float) -> float:
        estimator = PositionEstimator()
        timestamp = 0.0
        while timestamp <= 0.5 + 1e-9:
            estimator.update(
                index_fret=1.0,
                vision_confidence=0.8,
                timestamp_s=timestamp,
            )
            timestamp += 1.0 / fps

        estimates = []
        timestamp = 1.0
        while timestamp <= 1.8 + 1e-9:
            estimates.append(
                estimator.update(
                    index_fret=5.0,
                    vision_confidence=0.8,
                    timestamp_s=timestamp,
                )
            )
            timestamp += 1.0 / fps
        return next(
            estimate.timestamp_s
            for estimate in estimates
            if estimate.state == "locked" and estimate.position == 5
        )

    slow_lock = lock_time(4.0)
    fast_lock = lock_time(30.0)

    assert 0.25 <= slow_lock - 1.0 <= 0.50
    assert 0.25 <= fast_lock - 1.0 <= 0.35
    assert abs(slow_lock - fast_lock) <= 0.25


def test_timestamp_regression_requires_reset() -> None:
    estimator = PositionEstimator()
    estimator.update(index_fret=3.0, vision_confidence=0.8, timestamp_s=1.0)
    with pytest.raises(ValueError, match="monotonic"):
        estimator.update(index_fret=3.0, vision_confidence=0.8, timestamp_s=0.9)
