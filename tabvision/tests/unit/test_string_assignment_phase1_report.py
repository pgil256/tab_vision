from __future__ import annotations

from scripts.eval.string_assignment_phase1 import GRID, _prediction_hash, select_grid_row
from tabvision.fusion.segment_decoder import DEFAULT_SEGMENT_CONFIG
from tabvision.types import TabEvent


def _row(
    name: str,
    *,
    delta: float,
    wrong_rate: float,
    wall_seconds: float,
    departure: float,
    comp_delta: float = 0.0,
    comp_lower: float = 0.0,
    eligible: int = 1,
):
    return {
        "name": name,
        "delta": delta,
        "wrong_rate": wrong_rate,
        "wall_seconds": wall_seconds,
        "departure": departure,
        "comp_delta": comp_delta,
        "comp_ci_lower": comp_lower,
        "comp_noninferiority_eligible": eligible,
    }


def test_fixed_grid_covers_every_predeclared_weight_without_repeat_term() -> None:
    assert len(GRID) == 11
    assert {spec.config.zone_weight for spec in GRID} == {0.5, 1.0, 2.0}
    assert {spec.config.offset_weight for spec in GRID} == {0.5, 1.0, 2.0}
    assert {spec.config.state_change_weight for spec in GRID} == {0.5, 1.0, 2.0}
    assert {spec.config.prior_weight for spec in GRID} == {0.0, 0.25, 0.5}
    assert {spec.config.transition_weight for spec in GRID} == {0.75, 1.0, 1.25}
    assert all(spec.config.repeat_weight == 0.0 for spec in GRID)
    assert DEFAULT_SEGMENT_CONFIG.prior_weight == 0.5


def test_grid_selection_applies_metric_then_wrong_rate_runtime_and_departure_ties() -> None:
    rows = [
        _row("lower_metric", delta=0.01, wrong_rate=0.1, wall_seconds=1.0, departure=0.0),
        _row("more_wrong", delta=0.02, wrong_rate=0.3, wall_seconds=1.0, departure=0.0),
        _row("slower", delta=0.02, wrong_rate=0.2, wall_seconds=2.0, departure=0.0),
        _row("winner", delta=0.02, wrong_rate=0.2, wall_seconds=1.0, departure=0.0),
        _row("farther", delta=0.02, wrong_rate=0.2, wall_seconds=1.0, departure=1.0),
    ]
    selected, reason = select_grid_row(rows)
    assert selected["name"] == "winner"
    assert "hard comp" in reason


def test_grid_selection_excludes_comp_ineligible_metric_winner() -> None:
    rows = [
        _row(
            "regressed_comp",
            delta=0.10,
            wrong_rate=0.1,
            wall_seconds=1.0,
            departure=0.0,
            eligible=0,
        ),
        _row("safe", delta=0.01, wrong_rate=0.2, wall_seconds=1.0, departure=0.0),
    ]
    selected, _reason = select_grid_row(rows)
    assert selected["name"] == "safe"


def test_no_eligible_grid_point_uses_predeclared_comp_first_diagnostic_fallback() -> None:
    rows = [
        _row(
            "better_aggregate",
            delta=0.10,
            wrong_rate=0.1,
            wall_seconds=1.0,
            departure=0.0,
            comp_delta=-0.03,
            comp_lower=-0.04,
            eligible=0,
        ),
        _row(
            "least_comp_harm",
            delta=0.00,
            wrong_rate=0.2,
            wall_seconds=1.0,
            departure=0.0,
            comp_delta=-0.01,
            comp_lower=-0.02,
            eligible=0,
        ),
    ]
    selected, reason = select_grid_row(rows)
    assert selected["name"] == "least_comp_harm"
    assert "diagnosis only" in reason


def test_prediction_hash_uses_the_note_tables_declared_onset_precision() -> None:
    first = TabEvent(0.12345641, 0.2, 3, 5, 60, 0.75)
    csv_round_trip = TabEvent(0.123456, 0.0, 3, 5, 60, 0.75)
    changed_position = TabEvent(0.123456, 0.0, 2, 10, 60, 0.75)

    assert _prediction_hash({"clip": [first]}) == _prediction_hash({"clip": [csv_round_trip]})
    assert _prediction_hash({"clip": [first]}) != _prediction_hash({"clip": [changed_position]})
