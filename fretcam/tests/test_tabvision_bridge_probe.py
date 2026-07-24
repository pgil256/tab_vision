from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import fretcam.tabvision_bridge_probe as bridge_probe
from fretcam.tabvision_bridge_probe import (
    AggregateBridgeResult,
    BridgeProbeResult,
    ClipBridgeResult,
    aggregate_results,
    anchor_to_observation,
    cached_position_observations,
    classical_gold_pitch_audio_events,
    format_report,
    score_decoded_against_gold,
)
from tabvision.fusion.position_prior import PitchPositionPrior
from tabvision.fusion.transition_prior import TransitionPrior
from tabvision.types import GuitarConfig, TabEvent


def _tab(
    onset_s: float,
    pitch_midi: int,
    string_idx: int,
    fret: int,
) -> TabEvent:
    return TabEvent(
        onset_s=onset_s,
        duration_s=0.2,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch_midi,
        confidence=1.0,
    )


def test_anchor_maps_video_frame_to_audio_clock_and_fixed_window() -> None:
    observation = anchor_to_observation(
        SimpleNamespace(center_fret=5.9, confidence=0.75),
        frame_index=100,
        fps=25.0,
        offset_s=3.0,
        cfg=GuitarConfig(),
    )

    assert observation is not None
    assert observation.timestamp_s == pytest.approx(1.0)
    assert observation.position == 5
    assert observation.window_frets == (0, 4, 5, 6, 7, 8, 9)
    assert observation.confidence == 0.75
    assert observation.state == "locked"


@pytest.mark.parametrize(
    ("anchor", "frame_index", "expected"),
    [
        (SimpleNamespace(center_fret=5.0, confidence=0.199999), 100, None),
        (SimpleNamespace(center_fret=math.nan, confidence=0.8), 100, None),
        (SimpleNamespace(center_fret=25.0, confidence=0.8), 100, None),
    ],
)
def test_invalid_corrected_anchor_is_not_converted(
    anchor: SimpleNamespace,
    frame_index: int,
    expected: None,
) -> None:
    assert (
        anchor_to_observation(
            anchor,
            frame_index=frame_index,
            fps=25.0,
            offset_s=3.0,
            cfg=GuitarConfig(),
        )
        is expected
    )


def test_pre_audio_cache_anchor_keeps_its_negative_timestamp() -> None:
    observation = anchor_to_observation(
        SimpleNamespace(center_fret=5.0, confidence=0.8),
        frame_index=10,
        fps=25.0,
        offset_s=3.0,
        cfg=GuitarConfig(),
    )

    assert observation is not None
    assert observation.timestamp_s == pytest.approx(-2.6)


def test_cached_observations_are_sorted_and_use_injected_corrector() -> None:
    records = {50: "late", 25: "early", 30: None}
    anchors = {
        "early": SimpleNamespace(center_fret=1.2, confidence=0.2),
        "late": SimpleNamespace(center_fret=7.8, confidence=0.9),
    }

    observations = cached_position_observations(
        records,
        fps=25.0,
        offset_s=0.0,
        cfg=GuitarConfig(),
        anchor_builder=lambda record, _cfg: anchors[record],
    )

    assert [item.timestamp_s for item in observations] == [1.0, 2.0]
    assert [item.position for item in observations] == [1, 7]


def test_alignment_uses_rounded_onset_pitch_queues_without_positional_shift() -> None:
    gold = [
        _tab(1.0, 64, 5, 0),
        _tab(1.0, 64, 4, 5),
        _tab(2.0, 69, 5, 5),
    ]
    decoded = [
        _tab(1.0 + 4e-7, 64, 5, 0),
        _tab(1.0, 64, 3, 9),
        _tab(9.0, 40, 0, 0),
        _tab(2.0, 69, 5, 5),
    ]

    score = score_decoded_against_gold(gold, decoded, GuitarConfig())

    assert score.playable_notes == 3
    assert score.matched_notes == 3
    assert score.correct_notes == 2


def test_classical_gold_pitch_events_receive_gaps_position_policy() -> None:
    cfg = GuitarConfig()
    matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    matrix[4, 10] = 1.0
    prior = PitchPositionPrior({69: matrix})

    (event,) = classical_gold_pitch_audio_events(
        [_tab(1.0, 69, 5, 5)],
        prior,
        cfg,
    )

    assert event.pitch_midi == 69
    assert event.fret_prior is not None
    assert event.fret_prior[4, 10] == pytest.approx(1.0)
    assert event.fret_prior[5, 5] == pytest.approx(0.0)


def test_run_probe_loads_clean_classical_policy_artifacts_once(monkeypatch) -> None:
    position_prior = PitchPositionPrior({})
    sequence_prior = TransitionPrior(scheme="delta", delta_table={})
    position_calls: list[str] = []
    sequence_calls: list[str] = []

    monkeypatch.setattr(bridge_probe, "CLEAN_12", ())
    monkeypatch.setattr(
        bridge_probe,
        "load_pitch_position_prior",
        lambda name, *, cfg: position_calls.append(name) or position_prior,
    )
    monkeypatch.setattr(
        bridge_probe,
        "load_transition_prior",
        lambda name: sequence_calls.append(name) or sequence_prior,
    )

    result = bridge_probe.run_probe()

    assert result.clips == ()
    assert position_calls == ["gaps-v1"]
    assert sequence_calls == ["gaps-seq-v1"]


def test_run_probe_rejects_fusion_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TABVISION_LOW_FRET_BIAS", "0")

    with pytest.raises(RuntimeError, match="TABVISION_LOW_FRET_BIAS"):
        bridge_probe.run_probe()


def test_aggregate_metrics_and_markdown_are_deterministic() -> None:
    clips = (
        ClipBridgeResult("a", 10, 10, 10, 7, 8, 3, 4),
        ClipBridgeResult("b", 10, 9, 9, 8, 7, 2, 5),
    )

    aggregate = aggregate_results(clips)

    assert aggregate == AggregateBridgeResult(20, 19, 19, 15, 15, 5, 9)
    assert aggregate.net == 0
    assert aggregate.assignment_scored_notes == 19
    assert aggregate.excluded_playable_notes == 1
    assert aggregate.baseline_accuracy == pytest.approx(15 / 19)
    assert aggregate.bridge_accuracy == pytest.approx(15 / 19)
    assert aggregate.relative_error_reduction == 0.0

    report = format_report(BridgeProbeResult(clips, aggregate))
    assert "| **Total** | **19** | **15** | **15** | **+0** | **5** |" in report
    assert "0 / 4 = 0.0000%" in report
    assert "19` of `20` individually playable gold notes" in report
    assert "`gaps-v1` position prior" in report
    assert "`gaps-seq-v1` at weight `4.0`" in report
