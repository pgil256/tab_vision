from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval.fretcam_end_to_end import (
    CachedPredictionBackend,
    ClipResult,
    GoldClockPositionAnalyzer,
    _resolve_stems,
    aggregate_results,
    align_observations_to_gold_clock,
    format_report,
    micro_tab_f1,
    production_tab_cache_path,
    tab_events_to_audio_surrogate,
    validate_alignment,
)
from tabvision.eval.error_decomposition import ErrorDecomposition
from tabvision.eval.metrics import TabF1Result
from tabvision.types import AudioEvent, SessionConfig, TabEvent
from tabvision.video.position import PositionWindowObservation


def _score(*, tp: int, fp: int, fn: int) -> TabF1Result:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


def _clip(stem: str, baseline_tp: int, fretcam_tp: int) -> ClipResult:
    baseline = _score(tp=baseline_tp, fp=10 - baseline_tp, fn=10 - baseline_tp)
    fretcam = _score(tp=fretcam_tp, fp=10 - fretcam_tp, fn=10 - fretcam_tp)
    return ClipResult(
        stem=stem,
        media_duration_s=60.0,
        offset_s=0.04,
        offset_peak_ratio=3.0,
        direct_alignment_offset_s=None,
        direct_alignment_peak_ratio=None,
        gold_notes=10,
        audio_events=10,
        accepted_observations=4,
        affected_audio_events=2,
        evaluation_runtime_s=5.0,
        prediction_cache_sha256="a" * 64,
        baseline_tab=baseline,
        fretcam_tab=fretcam,
        baseline_errors=ErrorDecomposition(
            correct=baseline_tp,
            wrong_position_same_pitch=10 - baseline_tp,
        ),
        fretcam_errors=ErrorDecomposition(
            correct=fretcam_tp,
            wrong_position_same_pitch=10 - fretcam_tp,
        ),
    )


def test_align_observations_maps_video_time_to_gold_time() -> None:
    source = [
        PositionWindowObservation(
            timestamp_s=0.02,
            position=5,
            window_frets=(0, 4, 5, 6, 7, 8, 9),
            confidence=0.8,
            state="locked",
        )
    ]

    aligned = align_observations_to_gold_clock(
        source,
        video_minus_gold_offset_s=0.04,
    )

    assert aligned[0].timestamp_s == pytest.approx(-0.02)
    assert source[0].timestamp_s == 0.02


def test_validate_alignment_accepts_sharp_full_coverage_match() -> None:
    validate_alignment(
        offset_s=0.04,
        peak_ratio=2.1,
        audio_duration_s=120.72,
        video_duration_s=120.70,
        latest_gold_onset_s=120.60,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"peak_ratio": 1.9}, "without direct-waveform"),
        ({"video_duration_s": float("nan")}, "must be finite"),
        ({"video_duration_s": 119.0}, "durations differ"),
        ({"latest_gold_onset_s": 121.0}, "outside"),
    ],
)
def test_validate_alignment_rejects_untrustworthy_records(
    overrides: dict[str, float],
    message: str,
) -> None:
    values = {
        "offset_s": 0.04,
        "peak_ratio": 3.0,
        "audio_duration_s": 120.0,
        "video_duration_s": 120.0,
        "latest_gold_onset_s": 119.9,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_alignment(**values)


def test_validate_alignment_accepts_agreeing_direct_waveform_check() -> None:
    validate_alignment(
        offset_s=0.04,
        peak_ratio=1.7,
        audio_duration_s=120.0,
        video_duration_s=120.0,
        latest_gold_onset_s=119.9,
        direct_offset_s=0.03625,
        direct_peak_ratio=3.2,
    )


def test_validate_alignment_rejects_disagreeing_direct_waveform_check() -> None:
    with pytest.raises(ValueError, match="methods differ"):
        validate_alignment(
            offset_s=0.04,
            peak_ratio=1.7,
            audio_duration_s=120.0,
            video_duration_s=120.0,
            latest_gold_onset_s=119.9,
            direct_offset_s=0.02,
            direct_peak_ratio=3.2,
        )


def test_cached_backend_returns_independent_event_arrays() -> None:
    source = AudioEvent(
        onset_s=1.0,
        offset_s=2.0,
        pitch_midi=69,
        velocity=0.8,
        confidence=0.9,
        pitch_logits=np.asarray([0.2, 0.8]),
        fret_prior=np.asarray([0.4, 0.6]),
    )
    backend = CachedPredictionBackend("highres", [source])

    first = backend.transcribe(np.zeros(1), 22_050, SessionConfig())
    second = backend.transcribe(np.zeros(1), 22_050, SessionConfig())

    assert first[0] is not second[0]
    assert first[0].pitch_logits is not source.pitch_logits
    assert second[0].fret_prior is not first[0].fret_prior


def test_gold_clock_analyzer_wraps_live_observations() -> None:
    observation = PositionWindowObservation(
        timestamp_s=1.04,
        position=1,
        window_frets=(0, 1, 2, 3, 4, 5),
        confidence=0.7,
        state="holding",
    )
    inner = SimpleNamespace(analyze=lambda _frames, stride: [observation])
    analyzer = GoldClockPositionAnalyzer(  # type: ignore[arg-type]
        inner,
        video_minus_gold_offset_s=0.04,
    )

    result = analyzer.analyze([], stride=3)

    assert result[0].timestamp_s == pytest.approx(1.0)


def test_production_tab_cache_resolves_by_policy_fields(tmp_path: Path) -> None:
    expected = {
        "backend": "highres",
        "position_prior": "gaps-v1",
        "melodic_prior": False,
        "video": False,
    }
    chosen = tmp_path / "clip.current.json"
    chosen.write_text(
        json.dumps({"key_fields": expected, "events": []}),
        encoding="utf-8",
    )
    (tmp_path / "clip.other.json").write_text(
        json.dumps({"key_fields": {**expected, "position_prior": "none"}, "events": []}),
        encoding="utf-8",
    )

    assert (
        production_tab_cache_path(
            "clip",
            backend_name="highres",
            cache_dir=tmp_path,
        )
        == chosen
    )


def test_tab_event_surrogate_preserves_detected_pitch_and_timing() -> None:
    tab = TabEvent(1.2, 0.4, 4, 7, 71, 0.8, ("slide",))

    audio = tab_events_to_audio_surrogate([tab])

    assert audio[0].onset_s == 1.2
    assert audio[0].offset_s == pytest.approx(1.6)
    assert audio[0].pitch_midi == 71
    assert audio[0].tags == ("slide",)
    assert audio[0].fret_prior is None


def test_micro_tab_f1_sums_clip_confusion_counts() -> None:
    combined = micro_tab_f1(
        [
            _score(tp=8, fp=2, fn=1),
            _score(tp=3, fp=1, fn=2),
        ]
    )

    assert combined.true_positives == 11
    assert combined.false_positives == 3
    assert combined.false_negatives == 3
    assert combined.f1 == pytest.approx(11 / 14)


def test_aggregate_results_uses_paired_clip_deltas() -> None:
    clips = [_clip("improved", 6, 8), _clip("regressed", 8, 7)]

    aggregate = aggregate_results(clips, n_bootstrap=100, bootstrap_seed=42)

    assert aggregate["accepted_observations"] == 8
    assert aggregate["wrong_position_reduction"] == 1
    assert aggregate["improved_clips"] == ["improved"]
    assert aggregate["regressed_clips"] == ["regressed"]
    assert aggregate["paired_delta"]["statistic"] == pytest.approx(0.05)  # type: ignore[index]


def test_format_report_contains_auditable_headline() -> None:
    clips = [_clip("clip", 6, 7)]
    payload = {
        "population_label": "synthetic",
        "generated_at": "2026-07-24T00:00:00Z",
        "video_stride": 3,
        "audio_input": "cached predictions",
        "position_prior": "gaps-v1",
        "sequence_prior": "gaps-seq-v1",
        "assignment_decoder": "baseline",
        "clips": [clips[0].as_dict()],
        "aggregate": aggregate_results(clips, n_bootstrap=100, bootstrap_seed=42),
    }

    report = format_report(payload)

    assert "# FretCam current-solver paired end-to-end Tab F1" in report
    assert "| Macro per-clip Tab F1 |" in report
    assert "Wrong-position/same-pitch" in report


def test_resolve_stems_exposes_fresh_and_full_populations() -> None:
    assert len(_resolve_stems("clean12")) == 12
    assert len(_resolve_stems("source-disjoint10")) == 10
    assert len(_resolve_stems("test22")) == 22
    assert _resolve_stems("a,b") == ("a", "b")
