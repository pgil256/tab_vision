from __future__ import annotations

import math

import pytest

from tabvision.eval.string_assignment import (
    decode_with_analysis,
    expected_calibration_error,
    label_prediction_matches,
    paired_stratified_bootstrap,
    phrase_windows,
    wrong_below_correct_auc,
)
from tabvision.fusion import playability
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.viterbi import fuse
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


def _audio(pitch: int, onset: float, *, confidence: float = 0.8) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.25,
        pitch_midi=pitch,
        velocity=confidence,
        confidence=confidence,
    )


def _tab(pitch: int, onset: float, string_idx: int, fret: int) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


@pytest.fixture(autouse=True)
def _clear_sequence_prior() -> None:
    playability.set_transition_prior(None)


@pytest.mark.parametrize(
    "events",
    [
        [_audio(64, 0.0), _audio(65, 0.5), _audio(67, 1.0)],
        [_audio(60, 0.0), _audio(64, 0.02), _audio(67, 0.04), _audio(69, 0.8)],
    ],
)
def test_analysis_top_path_matches_production_fuse(events: list[AudioEvent]) -> None:
    expected = fuse(events, [], GuitarConfig(), lambda_vision=0.0)
    actual = decode_with_analysis(events, cfg=GuitarConfig(), k_paths=3)

    assert len(actual.paths) >= 1
    observed = actual.paths[0].events
    assert [
        (event.onset_s, event.pitch_midi, event.string_idx, event.fret) for event in observed
    ] == [(event.onset_s, event.pitch_midi, event.string_idx, event.fret) for event in expected]
    assert [event.confidence for event in observed] == pytest.approx(
        [event.confidence for event in expected]
    )


def test_hard_constraint_forces_pitch_preserving_position() -> None:
    event = _audio(64, 0.0)
    analysis = decode_with_analysis(
        [event],
        constraints={0: Candidate(string_idx=4, fret=5)},
        k_paths=3,
    )

    assert len(analysis.paths) == 1
    selected = analysis.paths[0].events[0]
    assert (selected.string_idx, selected.fret, selected.pitch_midi) == (4, 5, 64)


def test_infeasible_constraint_returns_no_path() -> None:
    analysis = decode_with_analysis(
        [_audio(64, 0.0)],
        constraints={0: Candidate(string_idx=0, fret=0)},
    )
    assert analysis.paths == ()


def test_k_best_paths_are_distinct_and_sorted() -> None:
    analysis = decode_with_analysis([_audio(64, 0.0), _audio(65, 0.5)], k_paths=3)
    keys = [
        tuple((event.string_idx, event.fret) for event in path.events) for path in analysis.paths
    ]
    assert len(keys) == 3
    assert len(set(keys)) == 3
    assert [path.score_delta_from_best for path in analysis.paths] == sorted(
        path.score_delta_from_best for path in analysis.paths
    )


def test_candidate_rankings_include_every_single_note_position() -> None:
    analysis = decode_with_analysis([_audio(64, 0.0)], k_paths=1)
    ranked = analysis.candidate_ranks[0]
    assert len(ranked) == 6
    assert ranked[0].cost_delta_from_best == pytest.approx(0.0)
    assert {(candidate.string_idx, candidate.fret) for candidate in ranked} == {
        (0, 24),
        (1, 19),
        (2, 14),
        (3, 9),
        (4, 5),
        (5, 0),
    }


def test_prediction_matcher_labels_correct_wrong_pitch_and_extra() -> None:
    gold = [_tab(64, 0.0, 5, 0), _tab(67, 1.0, 4, 8), _tab(69, 2.0, 3, 14)]
    predicted = [
        _tab(64, 0.01, 5, 0),
        _tab(67, 1.01, 3, 12),
        _tab(70, 2.01, 3, 15),
        _tab(72, 3.0, 4, 13),
    ]
    matches = label_prediction_matches(predicted, gold)
    assert [match.label for match in matches] == [
        "correct",
        "wrong_position_same_pitch",
        "pitch_off",
        "extra_detection",
    ]
    assert [match.gold_index for match in matches] == [0, 1, 2, None]


def test_auc_and_ece_have_expected_extremes() -> None:
    perfect = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert wrong_below_correct_auc(perfect) == pytest.approx(1.0)
    assert expected_calibration_error([(1.0, True), (0.0, False)]) == pytest.approx(0.0)
    assert math.isnan(wrong_below_correct_auc([(0.5, True)]))


def test_paired_stratified_bootstrap_is_deterministic_and_paired() -> None:
    baseline = {"s1": 0.4, "s2": 0.5, "c1": 0.7, "c2": 0.8}
    candidate = {clip: value + 0.03 for clip, value in baseline.items()}
    strata = {"s1": "solo", "s2": "solo", "c1": "comp", "c2": "comp"}
    first = paired_stratified_bootstrap(baseline, candidate, strata, n_resamples=500, seed=7)
    second = paired_stratified_bootstrap(baseline, candidate, strata, n_resamples=500, seed=7)
    assert first == second
    assert first.mean_delta == pytest.approx(0.03)
    assert first.lower == pytest.approx(0.03)
    assert first.upper == pytest.approx(0.03)


def test_phrase_windows_keep_onset_clusters_together_and_enforce_note_limit() -> None:
    events = [_tab(60 + index, index * 0.2, 3, 5 + index) for index in range(10)]
    events.insert(2, _tab(72, events[1].onset_s + 0.01, 2, 10))
    ambiguous = set(range(len(events)))
    windows = phrase_windows(events, ambiguous, max_notes=4, max_duration_s=1.0)

    assert windows
    assert all(window.end_index - window.start_index <= 4 for window in windows)
    simultaneous_indices = {1, 2}
    containing = [
        window
        for window in windows
        if any(window.start_index <= index < window.end_index for index in simultaneous_indices)
    ]
    assert len(containing) == 1
    assert all(
        containing[0].start_index <= index < containing[0].end_index
        for index in simultaneous_indices
    )
