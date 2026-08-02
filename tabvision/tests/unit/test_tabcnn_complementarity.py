from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tabvision.eval.bootstrap import BootstrapResult
from tabvision.eval.error_decomposition import ErrorDecomposition
from tabvision.eval.metrics import TabF1Result
from tabvision.eval.tabcnn_complementarity import (
    CONSERVATIVE_BLEND_POLICY,
    POSTERIOR_ONLY_POLICY,
    AggregateEvaluation,
    ComplementarityResult,
    aggregate_by_tier,
    aggregate_clip_evaluations,
    attach_tabcnn_prior,
    attach_tabcnn_priors,
    evaluate_complementarity,
    evaluate_promotion_gates,
    map_prior_to_pitch_candidates,
    match_pitch_correct_events,
    score_clip,
    select_tabcnn_position,
)
from tabvision.types import AudioEvent, TabEvent


def _tab(
    onset: float,
    string_idx: int,
    fret: int,
    *,
    pitch: int = 64,
) -> TabEvent:
    return TabEvent(onset, 0.25, string_idx, fret, pitch, 0.9)


def _prior(*positions: tuple[int, int, float]) -> np.ndarray:
    matrix = np.zeros((6, 25), dtype=np.float64)
    for string_idx, fret, probability in positions:
        matrix[string_idx, fret] = probability
    return matrix


def _result(f1: float) -> TabF1Result:
    return TabF1Result(f1, f1, f1, 1, 0, 0)


def _bootstrap(value: float, lower: float) -> BootstrapResult:
    return BootstrapResult(value, lower, value + 0.01, 10, 10_000, 0.95)


def _aggregate(
    delta: float,
    lower: float,
    *,
    wrong_position_reduction: float = 0.2,
) -> AggregateEvaluation:
    errors = ErrorDecomposition(correct=8, wrong_position_same_pitch=2)
    return AggregateEvaluation(
        clips=10,
        current_macro=_bootstrap(0.60, 0.58),
        candidate_macro=_bootstrap(0.60 + delta, 0.58 + delta),
        paired_delta=_bootstrap(delta, lower),
        current_micro=_result(0.60),
        candidate_micro=_result(0.60 + delta),
        current_errors=errors,
        candidate_errors=errors,
        wrong_position_reduction=1,
        wrong_position_relative_reduction=wrong_position_reduction,
        improved_clips=("a",),
        regressed_clips=(),
        unchanged_clips=(),
    )


def test_prior_mapping_masks_normalizes_and_keeps_canonical_order() -> None:
    matrix = _prior(
        (5, 0, 2.0),
        (4, 5, 1.0),
        (0, 0, 100.0),  # incompatible with MIDI 64
    )

    candidates = map_prior_to_pitch_candidates(matrix, 64)

    assert [(item.string_idx, item.fret) for item in candidates] == [
        (5, 0),
        (4, 5),
        (3, 9),
        (2, 14),
        (1, 19),
        (0, 24),
    ]
    assert candidates[0].probability == pytest.approx(2 / 3)
    assert candidates[1].probability == pytest.approx(1 / 3)
    assert sum(item.probability for item in candidates) == pytest.approx(1.0)


def test_prior_mapping_rejects_wrong_shape_negative_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        map_prior_to_pitch_candidates(np.zeros((6, 21)), 64)
    with pytest.raises(ValueError, match="non-negative"):
        map_prior_to_pitch_candidates(_prior((5, 0, -1.0)), 64)
    invalid = _prior((5, 0, 1.0))
    invalid[4, 5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        map_prior_to_pitch_candidates(invalid, 64)


def test_top_position_tie_breaks_to_lowest_fret_without_confidence_threshold() -> None:
    tied = _prior((5, 0, 1.0), (4, 5, 1.0))

    posterior_only = select_tabcnn_position(
        tied,
        64,
        policy=POSTERIOR_ONLY_POLICY,
    )
    conservative = select_tabcnn_position(
        tied,
        64,
        policy=CONSERVATIVE_BLEND_POLICY,
    )

    assert posterior_only is not None
    assert (posterior_only.string_idx, posterior_only.fret) == (5, 0)
    assert conservative is not None
    assert (conservative.string_idx, conservative.fret) == (5, 0)


def test_pitch_matching_is_within_50ms_pitch_correct_and_one_to_one() -> None:
    current = [
        _tab(0.04, 5, 0),
        _tab(0.01, 5, 0),
        _tab(1.01, 5, 0, pitch=65),
        _tab(2.06, 5, 0),
    ]
    gold = [
        _tab(0.00, 5, 0),
        _tab(1.00, 5, 0, pitch=64),
        _tab(2.00, 5, 0),
    ]

    matches = match_pitch_correct_events(current, gold)

    assert [(item.prediction_index, item.gold_index) for item in matches] == [(1, 0)]
    assert matches[0].onset_delta_s == pytest.approx(0.01)


def test_complementarity_reports_full_2x2_abstention_and_oracle() -> None:
    gold = [
        _tab(0.0, 5, 0),
        _tab(1.0, 5, 0),
        _tab(2.0, 5, 0),
        _tab(3.0, 5, 0),
        _tab(4.0, 5, 0),
        _tab(5.0, 5, 0),
    ]
    current = [
        _tab(0.0, 5, 0),  # both correct
        _tab(1.0, 5, 0),  # current only
        _tab(2.0, 4, 5),  # TabCNN only
        _tab(3.0, 4, 5),  # both wrong
        _tab(4.0, 5, 0),  # abstain, current correct
        _tab(5.0, 4, 5),  # abstain, current wrong
    ]
    priors = [
        _prior((5, 0, 1.0)),
        _prior((4, 5, 1.0)),
        _prior((5, 0, 1.0)),
        _prior((3, 9, 1.0)),
        None,
        np.zeros((6, 25)),
    ]

    result = evaluate_complementarity(current, gold, priors)

    assert result.both_correct == 1
    assert result.current_only == 1
    assert result.tabcnn_only == 1
    assert result.both_wrong == 1
    assert result.abstained_current_correct == 1
    assert result.abstained_current_wrong == 1
    assert result.coverage == pytest.approx(4 / 6)
    assert result.abstention_rate == pytest.approx(2 / 6)
    assert result.p_tabcnn_correct == pytest.approx(2 / 6)
    assert result.p_tabcnn_correct_given_coverage == pytest.approx(2 / 4)
    assert result.p_tabcnn_correct_given_current_wrong == pytest.approx(1 / 3)
    assert result.p_tabcnn_correct_given_current_wrong_and_coverage == pytest.approx(1 / 2)
    assert result.current_position_accuracy == pytest.approx(3 / 6)
    assert result.oracle_ceiling == pytest.approx(4 / 6)
    assert result.oracle_gain == pytest.approx(1 / 6)


def test_attach_posterior_changes_only_fret_prior() -> None:
    logits = np.asarray([0.1, 0.9])
    original_prior = _prior((4, 5, 1.0))
    event = AudioEvent(
        onset_s=1.25,
        offset_s=1.75,
        pitch_midi=64,
        velocity=0.7,
        confidence=0.8,
        pitch_logits=logits,
        fret_prior=original_prior,
        tags=("highres",),
    )

    enriched = attach_tabcnn_prior(
        event,
        _prior((5, 0, 1.0)),
        policy=POSTERIOR_ONLY_POLICY,
    )

    assert enriched is not event
    assert enriched.onset_s == event.onset_s
    assert enriched.offset_s == event.offset_s
    assert enriched.pitch_midi == event.pitch_midi
    assert enriched.velocity == event.velocity
    assert enriched.confidence == event.confidence
    assert enriched.pitch_logits is logits
    assert enriched.tags is event.tags
    assert enriched.fret_prior is not original_prior
    assert enriched.fret_prior is not None
    assert enriched.fret_prior[5, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(event.fret_prior, original_prior)


def test_conservative_blend_uses_frozen_product_weight() -> None:
    original_prior = _prior((4, 5, 0.8), (5, 0, 0.2))
    event = AudioEvent(1.0, 1.2, 64, 0.8, 0.9, fret_prior=original_prior)

    blended = attach_tabcnn_prior(
        event,
        _prior((5, 0, 0.8), (4, 5, 0.2)),
        policy=CONSERVATIVE_BLEND_POLICY,
    )

    assert blended.fret_prior is not None
    assert CONSERVATIVE_BLEND_POLICY.tabcnn_weight == 0.35
    assert CONSERVATIVE_BLEND_POLICY.min_top_probability == 0.0
    assert CONSERVATIVE_BLEND_POLICY.min_margin == 0.0
    current_score = 0.8 * 0.2**0.35
    tabcnn_score = 0.2 * 0.8**0.35
    total = current_score + tabcnn_score
    assert blended.fret_prior[4, 5] == pytest.approx(current_score / total)
    assert blended.fret_prior[5, 0] == pytest.approx(tabcnn_score / total)


def test_attach_priors_preserves_event_count_and_requires_alignment() -> None:
    events = [
        AudioEvent(0.0, 0.2, 64, 0.8, 0.9),
        AudioEvent(1.0, 1.2, 64, 0.8, 0.9),
    ]

    attached = attach_tabcnn_priors(
        events,
        [_prior((5, 0, 1.0)), None],
    )

    assert len(attached) == len(events)
    assert [event.onset_s for event in attached] == [0.0, 1.0]
    assert [event.pitch_midi for event in attached] == [64, 64]
    with pytest.raises(ValueError, match="equal length"):
        attach_tabcnn_priors(events, [None])


def test_clip_aggregation_computes_macro_micro_errors_and_paired_bootstrap() -> None:
    gold_a = [_tab(0.0, 5, 0), _tab(1.0, 5, 0)]
    current_a = [_tab(0.0, 4, 5), _tab(1.0, 5, 0)]
    candidate_a = [_tab(0.0, 5, 0), _tab(1.0, 5, 0)]
    gold_b = [_tab(0.0, 5, 0), _tab(1.0, 5, 0), _tab(2.0, 5, 0)]
    current_b = [_tab(0.0, 5, 0)]
    candidate_b = [_tab(0.0, 5, 0)]
    clips = [
        score_clip("a", "solo", current_a, candidate_a, gold_a),
        score_clip("b", "comp", current_b, candidate_b, gold_b),
    ]

    first = aggregate_clip_evaluations(clips, n_bootstrap=500, seed=42)
    second = aggregate_clip_evaluations(clips, n_bootstrap=500, seed=42)
    tiers = aggregate_by_tier(clips, n_bootstrap=100, seed=42)

    assert first.current_macro.statistic == pytest.approx((0.5 + 0.5) / 2)
    assert first.candidate_macro.statistic == pytest.approx((1.0 + 0.5) / 2)
    assert first.paired_delta.statistic == pytest.approx(0.25)
    assert first.paired_delta == second.paired_delta
    assert first.current_micro.true_positives == 2
    assert first.current_micro.false_positives == 1
    assert first.current_micro.false_negatives == 3
    assert first.candidate_micro.true_positives == 3
    assert first.current_errors.wrong_position_same_pitch == 1
    assert first.candidate_errors.wrong_position_same_pitch == 0
    assert first.wrong_position_relative_reduction == pytest.approx(1.0)
    assert first.improved_clips == ("a",)
    assert first.unchanged_clips == ("b",)
    assert sorted(tiers) == ["comp", "solo"]


def test_promotion_gate_encodes_aggregate_solo_reduction_and_comp_noninferiority() -> None:
    aggregate = _aggregate(0.025, 0.001, wrong_position_reduction=0.099)
    tiers = {
        "solo": _aggregate(0.031, 0.005),
        "comp": _aggregate(-0.004, -0.009),
    }

    passing = evaluate_promotion_gates(aggregate, tiers)
    failing_tiers = {
        **tiers,
        "solo": replace(tiers["solo"], wrong_position_relative_reduction=0.099),
    }
    failing = evaluate_promotion_gates(
        aggregate,
        failing_tiers,
    )

    assert passing.passed
    assert not failing.wrong_position_reduction
    assert not failing.passed


def test_gate_requires_named_solo_and_comp_tiers() -> None:
    with pytest.raises(ValueError, match="solo"):
        evaluate_promotion_gates(_aggregate(0.03, 0.01), {"comp": _aggregate(0.0, 0.0)})


def test_zero_denominator_complementarity_properties_are_defined() -> None:
    empty = ComplementarityResult(0, 0, 0, 0, 0, 0, 0, 0, 0)

    assert empty.coverage == 0.0
    assert empty.p_tabcnn_correct == 0.0
    assert empty.p_tabcnn_correct_given_current_wrong == 0.0
    assert empty.oracle_ceiling == 0.0
