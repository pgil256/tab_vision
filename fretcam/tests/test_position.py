from __future__ import annotations

import pytest

from fretcam.position import PositionEstimator, position_window, roman_position


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


def test_initial_lock_requires_five_consecutive_frames() -> None:
    estimates = _feed(PositionEstimator(), [5.2] * 5)

    assert [estimate.state for estimate in estimates[:4]] == ["acquiring"] * 4
    assert estimates[4].state == "locked"
    assert estimates[4].label == "Position V"
    assert estimates[4].window_frets == (0, 4, 5, 6, 7, 8, 9)
    assert estimates[4].confidence == pytest.approx(0.8)


def test_boundary_jitter_does_not_flap_the_locked_position() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [5.05] * 5)
    estimates = _feed(
        estimator,
        [4.98, 5.03, 4.96, 5.04, 4.99, 5.02] * 3,
        start=1.0,
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
        start=1.0,
    )

    assert all(estimate.state == "locked" for estimate in estimates)
    assert all(estimate.position == 2 for estimate in estimates)


def test_shift_uses_five_frame_hysteresis_without_intermediate_labels() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.1] * 5)
    estimates = _feed(estimator, [9.2] * 5, start=1.0)

    assert [estimate.state for estimate in estimates[:4]] == ["shifting"] * 4
    assert all(estimate.position is None for estimate in estimates[:4])
    assert all(estimate.label == "Shifting…" for estimate in estimates[:4])
    assert estimates[4].state == "locked"
    assert estimates[4].position == 9
    assert estimates[4].label == "Position IX"


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

    assert estimates[0].position == 1
    assert estimates[-1].state == "locked"
    assert estimates[-1].position == 12


def test_dropouts_hold_then_lose_and_reacquire() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [7.2] * 5)
    held = _feed(estimator, [None] * 5, start=1.0)
    lost = _feed(estimator, [None], start=2.0)[0]

    assert all(estimate.state == "holding" for estimate in held)
    assert all(estimate.position == 7 for estimate in held)
    assert held[-1].confidence < held[0].confidence
    assert lost.state == "lost"
    assert lost.position is None

    reacquired = _feed(estimator, [7.2] * 5, start=3.0)
    assert reacquired[-1].state == "locked"
    assert reacquired[-1].position == 7


def test_ema_is_reported_but_does_not_delay_large_shift_hysteresis() -> None:
    estimator = PositionEstimator()
    _feed(estimator, [1.0] * 5)
    shifted = _feed(estimator, [9.0], start=1.0)[0]

    assert shifted.state == "shifting"
    assert shifted.smoothed_index_fret == pytest.approx(3.8)


def test_timestamp_regression_requires_reset() -> None:
    estimator = PositionEstimator()
    estimator.update(index_fret=3.0, vision_confidence=0.8, timestamp_s=1.0)
    with pytest.raises(ValueError, match="monotonic"):
        estimator.update(index_fret=3.0, vision_confidence=0.8, timestamp_s=0.9)
