from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval.string_assignment_oracles import (
    apply_gold_oracle,
    fixed_window_groups,
    track_groups,
)
from tabvision.eval.string_assignment import RankedCandidate
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig, TabEvent


def _event(pitch: int, onset: float, string_idx: int, fret: int) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=0.8,
    )


def _ranks(pitch: int, cfg: GuitarConfig | None = None) -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(candidate.string_idx, candidate.fret, float(index))
        for index, candidate in enumerate(candidate_positions(pitch, cfg))
    )


@pytest.mark.parametrize(
    ("baseline_string", "gold_string"),
    [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4)],
)
def test_offset_oracle_handles_every_adjacent_string_direction(
    baseline_string: int,
    gold_string: int,
) -> None:
    pitch = 64
    cfg = GuitarConfig()
    baseline_fret = pitch - cfg.tuning_midi[baseline_string]
    gold_fret = pitch - cfg.tuning_midi[gold_string]
    predicted = [_event(pitch, 0.0, baseline_string, baseline_fret)]
    gold = {0: _event(pitch, 0.0, gold_string, gold_fret)}

    result = apply_gold_oracle(
        predicted,
        [_ranks(pitch)],
        gold,
        track_groups(set(gold)),
        strategy="offset",
        cfg=cfg,
    )

    assert result.ambiguous_accuracy == 1.0
    assert len(result.events) == 1
    assert result.events[0].pitch_midi == pitch
    assert (result.events[0].string_idx, result.events[0].fret) == (gold_string, gold_fret)


def test_impossible_boundary_shift_is_discarded_without_changing_pitch() -> None:
    pitch = 64
    cfg = GuitarConfig()
    # Two notes vote for +1 string; a high-E note cannot move farther and is
    # discarded by that state.  The chosen candidates never change pitch.
    predicted = [
        _event(pitch, 0.0, 3, 9),
        _event(pitch, 0.2, 3, 9),
        _event(pitch, 0.4, 5, 0),
    ]
    gold = {
        0: _event(pitch, 0.0, 4, 5),
        1: _event(pitch, 0.2, 4, 5),
        2: _event(pitch, 0.4, 5, 0),
    }

    result = apply_gold_oracle(
        predicted,
        [_ranks(pitch)] * 3,
        gold,
        track_groups(set(gold)),
        strategy="offset",
        cfg=cfg,
    )

    assert result.ambiguous_correct == 2
    assert result.dropped_impossible == 1
    assert [event.pitch_midi for event in result.events] == [pitch, pitch]


def test_fixed_windows_never_split_an_onset_cluster_at_a_boundary() -> None:
    events = [
        _event(64, 0.20, 5, 0),
        _event(64, 0.99, 5, 0),
        _event(67, 1.01, 4, 8),
        _event(69, 1.80, 4, 10),
    ]

    groups = fixed_window_groups(events, set(range(4)), 1.0)

    assert groups == ((0, 1, 2), (3,))


@pytest.mark.parametrize("strategy", ["offset", "fret_zone", "joint"])
def test_gold_oracle_cannot_score_below_unmodified_baseline(strategy: str) -> None:
    pitch = 64
    cfg = GuitarConfig()
    predicted = [
        _event(pitch, 0.0, 5, 0),
        _event(pitch, 0.2, 3, 9),
        _event(pitch, 0.4, 4, 5),
    ]
    gold = {
        0: _event(pitch, 0.0, 5, 0),
        1: _event(pitch, 0.2, 4, 5),
        2: _event(pitch, 0.4, 4, 5),
    }
    baseline_correct = 2

    result = apply_gold_oracle(
        predicted,
        [_ranks(pitch)] * 3,
        gold,
        track_groups(set(gold)),
        strategy=strategy,  # type: ignore[arg-type]
        cfg=cfg,
    )

    assert result.ambiguous_correct >= baseline_correct


def test_production_package_does_not_import_gold_oracle_module() -> None:
    package_root = Path(__file__).resolve().parents[2] / "tabvision"
    forbidden = "scripts.eval.string_assignment_oracles"
    offenders = [
        path for path in package_root.rglob("*.py") if forbidden in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
