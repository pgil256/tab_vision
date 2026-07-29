"""Unit cover for the Stage 1 segment-window ceiling probe.

The probe's verdict is only meaningful if the oracle really is an oracle
(contract-valid windows, precision 1.0) and the reranker really is capped, so
these pin the frozen §5a behaviour rather than the eventual numbers.
"""

from __future__ import annotations

import math

import pytest

stage1 = pytest.importorskip("scripts.eval.segment_window_stage1")

from tabvision.fusion.position_window_prior import (  # noqa: E402
    _is_valid_observation,
)
from tabvision.fusion.segment_decoder import (  # noqa: E402
    SegmentBoundary,
    SegmentDecodedPath,
)
from tabvision.types import GuitarConfig, TabEvent  # noqa: E402

CFG = GuitarConfig()


def _tab(onset: float, string_idx: int, fret: int) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=60,
        confidence=1.0,
        techniques=(),
    )


# ---------------------------------------------------------- gold-window oracle


def test_synthesized_windows_satisfy_the_validity_contract():
    """Stage 2 swaps in real observations, so Stage 1 must use the same contract."""
    gold = [_tab(t * 0.5, 2, 5) for t in range(20)]
    observations, _stats = stage1.synthesize_gold_windows(gold, CFG)
    assert observations
    for observation in observations:
        assert _is_valid_observation(observation, CFG)
        assert observation.state == "locked"


def test_oracle_precision_is_one_by_construction():
    """Every fretted gold note near an observation lies inside its window."""
    gold = [_tab(0.0, 2, 5), _tab(0.3, 3, 7), _tab(0.6, 1, 6)]
    observations, _stats = stage1.synthesize_gold_windows(gold, CFG)
    for observation in observations:
        near = [
            event
            for event in gold
            if -stage1.LOOKBEHIND_S <= event.onset_s - observation.timestamp_s <= stage1.LOOKAHEAD_S
            and event.fret > 0
        ]
        for event in near:
            assert event.fret in observation.window_frets


def test_coverage_degradation_matches_the_frozen_fraction():
    gold = [_tab(t * 0.25, 2, 5) for t in range(400)]
    observations, stats = stage1.synthesize_gold_windows(gold, CFG)
    eligible = len(observations) + stats["dropped_coverage"]
    assert eligible > 50
    assert abs(len(observations) / eligible - stage1.COVERAGE) < 0.02


def test_moments_wider_than_one_window_are_dropped():
    """Where a single window cannot cover the span, real FretCam destabilises."""
    gold = [_tab(0.0, 5, 1), _tab(0.1, 0, 12)]
    observations, stats = stage1.synthesize_gold_windows(gold, CFG)
    assert stats["span_too_wide"] > 0
    assert observations == []


def test_no_gold_yields_no_observations():
    assert stage1.synthesize_gold_windows([], CFG) == (
        [],
        {"grid": 0, "no_fretted": 0, "span_too_wide": 0, "dropped_coverage": 0},
    )


# ------------------------------------------------------------- segment slices


def test_segment_slices_reject_a_note_count_mismatch():
    segments = [SegmentBoundary(0, 1, 0.0, 0.5, 3)]
    with pytest.raises(ValueError, match="note counts"):
        stage1._segment_slices(segments, 4)


# ------------------------------------------------------------------- rerank


def _paths(costs, frets):
    """One single-segment path per (cost, fret) pair."""
    return [
        SegmentDecodedPath(
            events=(_tab(0.5, 2, fret),),
            latent_states=(),
            cost=cost,
            score_delta_from_best=cost - min(costs),
        )
        for cost, fret in zip(costs, frets, strict=True)
    ]


def _segments():
    return [SegmentBoundary(0, 1, 0.5, 0.5, 1)]


def _observation(position: int, timestamp: float = 0.5):
    window = (0, *range(max(1, position - 1), min(CFG.max_fret, position + 4) + 1))
    return stage1.PositionWindowObservation(
        timestamp_s=timestamp,
        position=position,
        window_frets=window,
        confidence=stage1.OBSERVATION_CONFIDENCE,
        state="locked",
    )


def test_rerank_abstains_without_observations():
    winner, raws, pairs = stage1.rerank(_paths([1.0, 1.1], [5, 9]), _segments(), [], CFG, 1)
    assert (winner, raws, pairs) == (0, [], 0)


def test_rerank_abstains_when_every_path_agrees_equally():
    """Both paths sit inside the observed window — vision cannot discriminate."""
    paths = _paths([1.0, 1.05], [5, 6])
    winner, _raws, _pairs = stage1.rerank(paths, _segments(), [_observation(5)], CFG, 1)
    assert winner == 0


def test_rerank_flips_to_the_agreeing_path_within_the_cap():
    """A 0.1 nat audio margin loses to a full-agreement disagreement."""
    paths = _paths([1.0, 1.1], [12, 5])
    winner, raws, _pairs = stage1.rerank(paths, _segments(), [_observation(5)], CFG, 1)
    assert winner == 1
    assert raws[1] > raws[0]


def test_cap_prevents_vision_from_vetoing_a_strong_audio_preference():
    """Beyond CAP nats of audio margin the window must not flip the decode."""
    paths = _paths([1.0, 1.0 + stage1.CAP + 0.5], [12, 5])
    winner, _raws, _pairs = stage1.rerank(paths, _segments(), [_observation(5)], CFG, 1)
    assert winner == 0


def test_exact_tie_keeps_the_baseline_path():
    paths = _paths([1.0, 1.0], [5, 5])
    winner, _raws, _pairs = stage1.rerank(paths, _segments(), [_observation(5)], CFG, 1)
    assert winner == 0


# --------------------------------------------------------------- raw scoring


def test_raw_score_is_median_agreement_times_log_count():
    segments = _segments()
    slices = [(0, 1)]
    events = (_tab(0.5, 2, 5),)
    observations = [_observation(5, 0.5), _observation(5, 0.45)]
    raw, count = stage1.path_raw_score(events, segments, slices, observations, CFG)
    assert count == 2
    assert raw == pytest.approx(1.0 * math.log(3.0))


def test_open_notes_are_excluded_from_agreement():
    """The open-string exemption is inherited, not relitigated (design §3.2)."""
    segments = _segments()
    slices = [(0, 1)]
    raw, count = stage1.path_raw_score((_tab(0.5, 2, 0),), segments, slices, [_observation(5)], CFG)
    assert (raw, count) == (0.0, 0)
