from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.eval.string_assignment_phase0 import (
    CONDITIONS,
    ConditionResult,
    OracleAggregate,
    _report,
    _write_csv,
)
from tabvision.eval.metrics import TabF1Result


def _condition(name: str, *, player: str = "00") -> ConditionResult:
    result = ConditionResult(name)
    for mode in ("solo", "comp"):
        clip_id = f"{player}_synthetic_{mode}"
        result.clip_scores[clip_id] = 0.5
        result.clip_tab[clip_id] = TabF1Result(0.5, 0.5, 0.5, 1, 1, 1)
        result.strata[clip_id] = f"{player}|{mode}"
        result.note_rows.append(
            {
                "track_id": clip_id,
                "player": player,
                "ambiguous_pitch_match": 1,
                "confidence": "0.50000000",
                "label": "correct",
                "candidate_top1": 1,
                "candidate_top3": 1,
                "reference_string": 4,
                "predicted_string": 4,
            }
        )
    return result


def test_identical_inputs_produce_identical_csv_and_report_hashes(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    _write_csv(first_csv, rows)
    _write_csv(second_csv, rows)

    dev = [_condition(name) for name in CONDITIONS]
    final = [_condition(name, player="05") for name in CONDITIONS]
    checked = _condition("checked_in_production", player="05")
    final.append(checked)
    tab = TabF1Result(0.5, 0.5, 0.5, 1, 1, 1)
    segment = [
        OracleAggregate("baseline", 1, 1, 0, {"05_synthetic_solo": tab}),
        OracleAggregate("joint_4s", 1, 1, 0, {"05_synthetic_solo": tab}),
    ]
    phrase = {
        "phrases": 1.0,
        "infeasible_phrases": 0.0,
        "notes": 1.0,
        "baseline": 1.0,
        "anchored_top1": 1.0,
        "best_of_three": 1.0,
        "anchor_lift": 0.0,
        "top3_lift": 0.0,
        "baseline_tab_f1": 0.5,
        "anchored_top1_tab_f1": 0.5,
        "best_of_three_tab_f1": 0.5,
    }
    report_args = (
        dev,
        final,
        phrase,
        segment,
        {"onset_f1": 0.8, "pitch_f1": 0.7},
        Path("provenance.json"),
    )
    first_report = _report(*report_args)
    second_report = _report(*report_args)

    assert (
        hashlib.sha256(first_csv.read_bytes()).hexdigest()
        == hashlib.sha256(second_csv.read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256(first_report.encode()).hexdigest()
        == hashlib.sha256(second_report.encode()).hexdigest()
    )
